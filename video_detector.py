"""
video_detector.py
-----------------
Detector de deepfake en video por muestreo de frames.

Configuracion:
    VIDEO_MAX_FRAMES=12
    VIDEO_SAMPLE_METHOD=uniform|random
    VIDEO_FAKE_RATIO_THRESHOLD=0.30
    VIDEO_FRAME_WORKERS=4
    VIDEO_FRAME_SIZE=224
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import random
import shutil
import tempfile
from pathlib import Path

import cv2

from image_detector import analyze_image

MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "12"))
SAMPLE_METHOD = os.getenv("VIDEO_SAMPLE_METHOD", "uniform").strip().lower()
FAKE_RATIO_THRESHOLD = float(os.getenv("VIDEO_FAKE_RATIO_THRESHOLD", "0.30"))
FRAME_WORKERS = max(1, int(os.getenv("VIDEO_FRAME_WORKERS", "4")))
FRAME_SIZE = int(os.getenv("VIDEO_FRAME_SIZE", os.getenv("IMAGE_DETECTOR_SIZE", "224")))


def analyze_video(video_path: str, max_frames: int | None = None) -> dict:
    """
    Analiza un video muestreando frames y agregando el veredicto por frame.
    """
    frame_paths, tmp_dir, metadata = _extract_frames(video_path, max_frames or MAX_FRAMES)

    try:
        if not frame_paths:
            raise ValueError("No se pudo extraer ningun frame del video.")

        per_frame_results = _analyze_frames(frame_paths)
        suspicious_frames = [
            result
            for result in per_frame_results
            if result.get("label", "").lower() == "fake"
            or float(result.get("ai_probability", 0)) >= 30
        ]
        fake_ratio = len(suspicious_frames) / len(per_frame_results)
        avg_ai = sum(float(r.get("ai_probability", 0)) for r in per_frame_results) / len(
            per_frame_results
        )
        max_ai = max(float(r.get("ai_probability", 0)) for r in per_frame_results)
        ai_probability = round(max(avg_ai, fake_ratio * 100, max_ai * 0.65), 1)
        label = "fake" if fake_ratio >= FAKE_RATIO_THRESHOLD or ai_probability >= 30 else "real"
        confidence = ai_probability / 100 if label == "fake" else 1 - ai_probability / 100

        return {
            "modality": "video",
            "label": label,
            "confidence": round(confidence, 4),
            "scores": {
                "fake": round(ai_probability / 100, 4),
                "real": round(1 - ai_probability / 100, 4),
            },
            "ai_probability": ai_probability,
            "fake_frame_ratio": round(fake_ratio, 4),
            "fake_ratio_threshold": FAKE_RATIO_THRESHOLD,
            "frames_analyzed": len(per_frame_results),
            "sampling": metadata,
            "per_frame": per_frame_results,
            "source": "huggingface_frames",
            "file_name": Path(video_path).name,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_frames(video_path: str, max_frames: int) -> tuple[list[str], str, dict]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No existe el video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    tmp_dir = tempfile.mkdtemp(prefix="veritas_video_frames_")

    try:
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video con OpenCV: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if total_frames <= 0:
            raise ValueError(f"No se pudieron leer frames de: {video_path}")

        frame_indices = _sample_indices(total_frames, max_frames)
        frame_paths: list[str] = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue

            resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            frame_path = os.path.join(tmp_dir, f"frame_{idx}.jpg")
            cv2.imwrite(frame_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_paths.append(frame_path)

        metadata = {
            "method": SAMPLE_METHOD if SAMPLE_METHOD in {"uniform", "random"} else "uniform",
            "max_frames": max_frames,
            "total_frames": total_frames,
            "fps": round(fps, 3),
            "frame_size": FRAME_SIZE,
        }
        return frame_paths, tmp_dir, metadata
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        cap.release()


def _sample_indices(total_frames: int, max_frames: int) -> list[int]:
    max_frames = max(1, min(max_frames, total_frames))
    if SAMPLE_METHOD == "random" and total_frames > max_frames:
        seed_value = os.getenv("VIDEO_RANDOM_SEED")
        rng = random.Random(int(seed_value)) if seed_value else random.Random()
        return sorted(rng.sample(range(total_frames), max_frames))

    if max_frames == 1:
        return [0]
    step = (total_frames - 1) / (max_frames - 1)
    return sorted({round(index * step) for index in range(max_frames)})


def _analyze_frames(frame_paths: list[str]) -> list[dict]:
    if len(frame_paths) <= 1 or FRAME_WORKERS <= 1:
        return [analyze_image(path) for path in frame_paths]

    worker_count = min(FRAME_WORKERS, len(frame_paths))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(analyze_image, frame_paths))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python video_detector.py <ruta_a_video>")
        sys.exit(1)

    result = analyze_video(sys.argv[1])
    print({k: v for k, v in result.items() if k != "per_frame"})
