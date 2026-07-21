"""
image_detector.py
-----------------
Detector de imagenes IA/deepfake para Veritas.

Configuracion:
    DEEPFAKE_MODEL_NAME
    IMAGE_MODEL_NAMES
    VERITAS_DEVICE=auto|cpu|cuda|mps
    VERITAS_USE_FP16=auto|true|false
    IMAGE_DETECTOR_SIZE=224
    IMAGE_UNTRUSTED_THRESHOLD=60

NOTA (21/7/2026):
    El modelo "prithivMLmods/deepfake-detector-model-v1" quedo excluido del
    ensemble por defecto. Se confirmo empiricamente con test_imagen.py que
    da resultados invertidos/no confiables: puntua mas "fake" en fotos
    reales de camara que en imagenes generadas por IA (ej: foto real 94.5%
    vs imagen IA 83.2% en el mismo test). Si en el futuro se corrige el
    mapeo de labels del checkpoint, se puede volver a incluir seteando
    IMAGE_MODEL_NAMES manualmente en el .env.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
DEFAULT_SECONDARY_MODEL_NAME = "capcheck/ai-image-detection"

# Modelos con id2label invertido/no confiable, confirmado con datos reales.
# Se excluyen del ensemble por defecto aunque esten seteados via
# DEEPFAKE_MODEL_NAME, salvo que el usuario los pida explicitamente en
# IMAGE_MODEL_NAMES.
UNRELIABLE_MODELS = {"prithivMLmods/deepfake-detector-model-v1"}

MODEL_NAME = os.getenv("DEEPFAKE_MODEL_NAME", DEFAULT_MODEL_NAME)

# Por defecto el ensemble usa solo el modelo secundario (capcheck), porque
# el primario (prithivMLmods) demostro dar resultados invertidos. Si se
# define IMAGE_MODEL_NAMES explicitamente en el .env, se respeta esa lista
# tal cual la pida el usuario (incluso si incluye un modelo no confiable).
MODEL_NAMES = [
    name.strip()
    for name in os.getenv(
        "IMAGE_MODEL_NAMES",
        DEFAULT_SECONDARY_MODEL_NAME,
    ).split(",")
    if name.strip()
]

IMAGE_SIZE = int(os.getenv("IMAGE_DETECTOR_SIZE", "224"))
UNTRUSTED_THRESHOLD = float(os.getenv("IMAGE_UNTRUSTED_THRESHOLD", "60"))

FAKE_LABEL_HINTS = ("fake", "deepfake", "synthetic", "generated", "ai", "artificial")
REAL_LABEL_HINTS = ("real", "bonafide", "genuine", "human", "authentic")

_model_bundles = None
_device = None
_use_fp16 = False
_load_lock = threading.Lock()


def _select_device() -> torch.device:
    requested = os.getenv("VERITAS_DEVICE", "auto").strip().lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _should_use_fp16(device: torch.device) -> bool:
    value = os.getenv("VERITAS_USE_FP16", "auto").strip().lower()
    if value in {"1", "true", "yes"}:
        return device.type == "cuda"
    if value in {"0", "false", "no"}:
        return False
    return device.type == "cuda"


def _load_model():
    """Carga los modelos una sola vez y evita carreras entre requests."""
    global _model_bundles, _device, _use_fp16
    if _model_bundles is not None:
        return _model_bundles, _device, _use_fp16

    with _load_lock:
        if _model_bundles is None:
            _device = _select_device()
            _use_fp16 = _should_use_fp16(_device)
            bundles = []
            for model_name in _dedupe(MODEL_NAMES):
                if model_name in UNRELIABLE_MODELS:
                    print(
                        f"[image_detector] AVISO: '{model_name}' esta marcado "
                        "como no confiable (ver UNRELIABLE_MODELS) pero fue "
                        "pedido explicitamente en IMAGE_MODEL_NAMES. Se carga "
                        "igual, pero revisa sus resultados con cuidado."
                    )
                print(
                    f"[image_detector] Cargando '{model_name}' en {_device}"
                    f"{' fp16' if _use_fp16 else ''}..."
                )
                processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForImageClassification.from_pretrained(model_name)
                model.eval()
                model.to(_device)
                if _use_fp16:
                    model.half()
                bundles.append(
                    {
                        "name": model_name,
                        "model": model,
                        "processor": processor,
                    }
                )
            _model_bundles = bundles
    return _model_bundles, _device, _use_fp16


def analyze_image(image_path: str) -> dict:
    return analyze_images_batch([image_path], batch_size=1)[0]


def analyze_images_batch(image_paths: Iterable[str], batch_size: int = 8) -> list[dict]:
    paths = [str(path) for path in image_paths]
    if not paths:
        return []

    model_bundles, device, use_fp16 = _load_model()
    results: list[dict] = []

    for start in range(0, len(paths), max(1, batch_size)):
        chunk = paths[start : start + max(1, batch_size)]
        loaded = [_load_image(path) for path in chunk]
        images = [item["image"] for item in loaded]
        predictions_by_path = [
            {
                "raw_model_scores": {},
                "model_probabilities": {},
            }
            for _ in chunk
        ]

        for bundle in model_bundles:
            inputs = bundle["processor"](images=images, return_tensors="pt")
            inputs = _move_inputs(inputs, device, use_fp16)

            with torch.inference_mode():
                outputs = bundle["model"](**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().tolist()

            for index, row in enumerate(probs):
                raw_scores = _scores_from_probs(bundle["model"], row)
                fake_probability = _fake_probability(raw_scores)
                predictions_by_path[index]["raw_model_scores"][bundle["name"]] = raw_scores
                predictions_by_path[index]["model_probabilities"][bundle["name"]] = round(
                    fake_probability,
                    1,
                )

        for path, metadata, prediction in zip(chunk, loaded, predictions_by_path):
            model_probability = max(prediction["model_probabilities"].values(), default=0.0)
            ai_probability = round(model_probability, 1)
            label = "fake" if ai_probability >= UNTRUSTED_THRESHOLD else "real"
            confidence = ai_probability / 100 if label == "fake" else 1 - ai_probability / 100

            results.append(
                {
                    "modality": "image",
                    "label": label,
                    "confidence": round(confidence, 4),
                    "scores": {
                        "fake": round(ai_probability / 100, 4),
                        "real": round(1 - ai_probability / 100, 4),
                    },
                    "raw_model_scores": prediction["raw_model_scores"],
                    "model_probabilities": prediction["model_probabilities"],
                    "ai_probability": ai_probability,
                    "model_probability": round(model_probability, 1),
                    "threshold": UNTRUSTED_THRESHOLD,
                    "image_size": [metadata["width"], metadata["height"]],
                    "device": str(device),
                    "use_fp16": use_fp16,
                    "source": "huggingface_ensemble",
                    "model": ",".join(_dedupe(MODEL_NAMES)),
                    "file_name": Path(path).name,
                }
            )

    return results


def _load_image(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe la imagen: {path}")

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            fmt = image.format or Path(path).suffix.replace(".", "").upper() or "UNKNOWN"
            has_exif = bool(image.getexif())
            rgb = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Formato de imagen no reconocido: {path}") from exc
    except OSError as exc:
        raise ValueError(f"No se pudo abrir la imagen '{path}': {exc}") from exc

    resized = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return {
        "image": resized,
        "array": np.asarray(resized, dtype=np.float32),
        "width": width,
        "height": height,
        "format": fmt,
        "has_exif": has_exif,
        "file_size": os.path.getsize(path),
    }


def _move_inputs(inputs, device: torch.device, use_fp16: bool):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if use_fp16 and value.is_floating_point():
                value = value.half()
        moved[key] = value
    return moved


def _label_for_index(model, index: int) -> str:
    id2label = getattr(model.config, "id2label", None) or {0: "fake", 1: "real"}
    return str(id2label.get(index, id2label.get(str(index), index))).lower()


def _scores_from_probs(model, probs: list[float]) -> dict:
    return {
        _label_for_index(model, index): round(float(prob), 4)
        for index, prob in enumerate(probs)
    }


def _fake_probability(scores: dict[str, float]) -> float:
    fake_scores = [
        value for label, value in scores.items() if _contains_hint(label, FAKE_LABEL_HINTS)
    ]
    real_scores = [
        value for label, value in scores.items() if _contains_hint(label, REAL_LABEL_HINTS)
    ]
    if fake_scores:
        return min(sum(fake_scores) * 100, 100)
    if real_scores:
        return max((1 - max(real_scores)) * 100, 0)
    # Si no reconocemos ninguna etiqueta, NO asumimos que la clase mas
    # confiada del modelo es "fake" (eso causaba falsos positivos del 100%
    # en fotos reales). Devolvemos un valor neutro y avisamos por consola
    # para poder mapear las etiquetas manualmente.
    print(f"[image_detector] WARNING: etiquetas no reconocidas: {list(scores.keys())}")
    return 50.0


def _contains_hint(label: str, hints: tuple[str, ...]) -> bool:
    compact = "".join(ch for ch in label.lower() if ch.isalnum())
    return any(hint in compact for hint in hints)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _forensic_manipulation_score(
    arr: np.ndarray,
    width: int,
    height: int,
    fmt: str,
    has_exif: bool,
    file_size: int,
) -> float:
    score = 4.0
    megapixels = max((width * height) / 1_000_000, 0.001)
    bytes_per_pixel = file_size / max(width * height, 1)
    channel_std = float(np.std(arr, axis=(0, 1)).mean())
    edge_energy = float(
        np.mean(np.abs(np.diff(arr, axis=0))) + np.mean(np.abs(np.diff(arr, axis=1)))
    )

    if not has_exif:
        score += 14
    if fmt.lower() in {"png", "webp"}:
        score += 6
    if megapixels >= 1.5 and bytes_per_pixel < 0.45:
        score += 8
    if width == height and width in {512, 768, 1024, 1536, 2048}:
        score += 5
    if channel_std < 34:
        score += 7
    if edge_energy < 16:
        score += 7

    return min(score, 42.0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python image_detector.py <ruta_a_imagen>")
        sys.exit(1)

    print(analyze_image(sys.argv[1]))