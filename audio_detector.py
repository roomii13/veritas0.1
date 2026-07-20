"""
audio_detector.py
-----------------
Detecta si un audio de voz fue generado o clonado con IA usando un
modelo open-source de Hugging Face basado en Wav2Vec2-XLSR.

Modelo: Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
"""

import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
TARGET_SR = 16000

_model = None
_feature_extractor = None


def _load_model():
    global _model, _feature_extractor
    if _model is None:
        print(f"[audio_detector] Descargando/cargando modelo '{MODEL_NAME}'...")
        _feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _feature_extractor


def analyze_audio(audio_path: str) -> dict:
    """
    Analiza un archivo de audio comun: wav, mp3, m4a, etc.
    """
    model, feature_extractor = _load_model()
    waveform, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    inputs = feature_extractor(waveform, sampling_rate=TARGET_SR, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

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
        "source": "huggingface",
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python audio_detector.py <ruta_a_audio>")
        sys.exit(1)

    print(analyze_audio(sys.argv[1]))
