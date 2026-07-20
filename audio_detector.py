"""
audio_detector.py
-----------------
Detector de audio clonado/IA para Veritas.

Configuracion:
    AUDIO_DEEPFAKE_MODEL_NAME
    AUDIO_TARGET_SR=16000
    AUDIO_MAX_DURATION_SEC=60
    AUDIO_MIN_DURATION_SEC=1
    AUDIO_NORMALIZE=false
    VERITAS_DEVICE=auto|cpu|cuda|mps
    VERITAS_USE_FP16=auto|true|false
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

DEFAULT_MODEL_NAME = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
MODEL_NAME = os.getenv("AUDIO_DEEPFAKE_MODEL_NAME", DEFAULT_MODEL_NAME)
TARGET_SR = int(os.getenv("AUDIO_TARGET_SR", "16000"))
MAX_DURATION_SEC = float(os.getenv("AUDIO_MAX_DURATION_SEC", "60"))
MIN_DURATION_SEC = float(os.getenv("AUDIO_MIN_DURATION_SEC", "1"))
NORMALIZE_AUDIO = os.getenv("AUDIO_NORMALIZE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

_model = None
_feature_extractor = None
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
    global _model, _feature_extractor, _device, _use_fp16
    if _model is not None:
        return _model, _feature_extractor, _device, _use_fp16

    with _load_lock:
        if _model is None:
            _device = _select_device()
            _use_fp16 = _should_use_fp16(_device)
            print(
                f"[audio_detector] Cargando '{MODEL_NAME}' en {_device}"
                f"{' fp16' if _use_fp16 else ''}..."
            )
            _feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
            _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
            _model.eval()
            _model.to(_device)
            if _use_fp16:
                _model.half()
    return _model, _feature_extractor, _device, _use_fp16


def analyze_audio(audio_path: str) -> dict:
    """
    Analiza audio comun: wav, mp3, m4a, flac, ogg.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"No existe el audio: {audio_path}")

    model, feature_extractor, device, use_fp16 = _load_model()
    waveform, duration_sec, silence_ratio = _load_waveform(audio_path)

    inputs = feature_extractor(waveform, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = _move_inputs(inputs, device, use_fp16)

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().tolist()

    if probs and isinstance(probs[0], list):
        probs = probs[0]

    id2label = getattr(model.config, "id2label", {})
    scores = {
        str(id2label.get(i, id2label.get(str(i), i))).lower(): round(float(probs[i]), 4)
        for i in range(len(probs))
    }
    label = max(scores, key=scores.get)

    return {
        "modality": "audio",
        "label": label,
        "confidence": scores[label],
        "scores": scores,
        "duration_sec": round(duration_sec, 2),
        "target_sr": TARGET_SR,
        "max_duration_sec": MAX_DURATION_SEC,
        "silence_ratio": round(silence_ratio, 4),
        "device": str(device),
        "use_fp16": use_fp16,
        "source": "huggingface",
        "model": MODEL_NAME,
        "file_name": Path(audio_path).name,
    }


def _load_waveform(audio_path: str) -> tuple[np.ndarray, float, float]:
    try:
        waveform, _ = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True,
            duration=MAX_DURATION_SEC,
            res_type="kaiser_fast",
        )
    except Exception as exc:
        raise ValueError(f"No se pudo leer el audio '{audio_path}': {exc}") from exc

    if waveform.size == 0:
        raise ValueError("El audio no contiene muestras validas.")

    duration_sec = waveform.size / TARGET_SR
    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio demasiado corto: {duration_sec:.2f}s. Minimo: {MIN_DURATION_SEC:.2f}s."
        )

    waveform = waveform.astype(np.float32)
    peak = float(np.max(np.abs(waveform)))
    if peak < 1e-5:
        raise ValueError("El audio parece silencio o tiene volumen demasiado bajo.")

    silence_ratio = float(np.mean(np.abs(waveform) < 0.005))
    if NORMALIZE_AUDIO:
        waveform = waveform / peak

    return waveform, duration_sec, silence_ratio


def _move_inputs(inputs, device: torch.device, use_fp16: bool):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if use_fp16 and value.is_floating_point():
                value = value.half()
        moved[key] = value
    return moved


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python audio_detector.py <ruta_a_audio>")
        sys.exit(1)

    print(analyze_audio(sys.argv[1]))
