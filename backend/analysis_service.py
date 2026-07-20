from __future__ import annotations

import os

from audio_detector import analyze_audio
from image_detector import analyze_image
from video_detector import analyze_video

from .fallback_detector import analyze_file_fallback


def analyze_path(path: str, modality: str) -> dict:
    try:
        if modality == "image":
            return analyze_image(path)
        if modality == "audio":
            return analyze_audio(path)
        if modality == "video":
            return analyze_video(path)
    except Exception as exc:
        if _allow_fallback():
            return analyze_file_fallback(path, modality, exc)
        raise

    raise ValueError(f"Modalidad no soportada: {modality}")


def _allow_fallback() -> bool:
    value = os.getenv("VERITAS_ALLOW_HEURISTIC_FALLBACK", "true").strip().lower()
    return value not in {"0", "false", "no"}
