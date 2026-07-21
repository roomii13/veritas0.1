from __future__ import annotations

from audio_detector import analyze_audio
from image_detector import analyze_image
from video_detector import analyze_video


def analyze_path(path: str, modality: str) -> dict:
    if modality == "image":
        return analyze_image(path)
    if modality == "audio":
        return analyze_audio(path)
    if modality == "video":
        return analyze_video(path)

    raise ValueError(f"Modalidad no soportada: {modality}")
