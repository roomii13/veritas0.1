"""
image_detector.py
-----------------
Detecta si una imagen fue generada por IA usando un modelo open-source
de Hugging Face.

Modelo: prithivMLmods/deepfake-detector-model-v1
"""

from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

_model = None
_processor = None


def _load_model():
    """Carga el modelo una sola vez y lo cachea en memoria."""
    global _model, _processor
    if _model is None:
        print(f"[image_detector] Descargando/cargando modelo '{MODEL_NAME}'...")
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor


def _label_for_index(model, index: int) -> str:
    id2label = getattr(model.config, "id2label", None) or {0: "fake", 1: "real"}
    return str(id2label.get(index, id2label.get(str(index), index))).lower()


def analyze_image(image_path: str) -> dict:
    """
    Analiza una imagen y devuelve label, confidence y scores.
    """
    model, processor = _load_model()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

    scores = {
        _label_for_index(model, i): round(float(probs[i]), 4)
        for i in range(len(probs))
    }
    label = max(scores, key=scores.get)

    return {
        "modality": "image",
        "label": label,
        "confidence": scores[label],
        "scores": scores,
        "source": "huggingface",
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python image_detector.py <ruta_a_imagen>")
        sys.exit(1)

    print(analyze_image(sys.argv[1]))
