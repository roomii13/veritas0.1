"""
video_detector.py
-----------------
Detecta deepfake en video muestreando frames y analizando cada frame con
el detector de imagenes.
"""

import os
import shutil
import tempfile

import cv2

from image_detector import analyze_image


def _extract_frames(video_path: str, max_frames: int = 12) -> tuple[list[str], str]:
    """Extrae hasta max_frames frames distribuidos uniformemente en el video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No se pudieron leer frames de: {video_path}")

    step = max(total_frames // max_frames, 1)
    frame_indices = list(range(0, total_frames, step))[:max_frames]
    tmp_dir = tempfile.mkdtemp(prefix="veritas_video_frames_")
    frame_paths = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue
        frame_path = os.path.join(tmp_dir, f"frame_{idx}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths, tmp_dir


def analyze_video(video_path: str, max_frames: int = 12) -> dict:
    """
    Analiza un video muestreando frames y promediando el veredicto.
    """
    frame_paths, tmp_dir = _extract_frames(video_path, max_frames=max_frames)

    try:
        if not frame_paths:
            raise ValueError("No se pudo extraer ningun frame del video.")

        per_frame_results = []
        fake_count = 0

        for frame_path in frame_paths:
            result = analyze_image(frame_path)
            per_frame_results.append(result)
            if result["label"].lower() == "fake":
                fake_count += 1

        fake_ratio = fake_count / len(per_frame_results)
        avg_confidence = sum(r["confidence"] for r in per_frame_results) / len(
            per_frame_results
        )
        label = "fake" if fake_ratio > 0.30 else "real"

        return {
            "modality": "video",
            "label": label,
            "confidence": round(avg_confidence, 4),
            "fake_frame_ratio": round(fake_ratio, 4),
            "frames_analyzed": len(per_frame_results),
            "per_frame": per_frame_results,
            "source": "huggingface_frames",
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python video_detector.py <ruta_a_video>")
        sys.exit(1)

    result = analyze_video(sys.argv[1])
    print({k: v for k, v in result.items() if k != "per_frame"})
