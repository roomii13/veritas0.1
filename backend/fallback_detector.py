from __future__ import annotations

import os
from pathlib import Path


def analyze_file_fallback(path: str, modality: str, error: Exception) -> dict:
    """
    Fallback local para demo cuando el modelo Hugging Face no esta disponible.
    No afirma deteccion real: solo estima senales estadisticas simples.
    """
    if modality == "image":
        return _image_fallback(path, error)
    if modality == "audio":
        return _audio_fallback(path, error)
    if modality == "video":
        return _video_fallback(path, error)
        return _make_result(path, modality, 50, ["Tipo de archivo no reconocido"], error)


def _image_fallback(path: str, error: Exception) -> dict:
    reasons = []
    score = 22

    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format or "unknown"
            exif = image.getexif()
            small = image.convert("RGB").resize((64, 64))
            arr = np.asarray(small, dtype=np.float32)

        megapixels = (width * height) / 1_000_000
        file_size = os.path.getsize(path)
        channel_std = float(np.std(arr, axis=(0, 1)).mean())
        edge_energy = float(
            np.mean(np.abs(np.diff(arr, axis=0))) + np.mean(np.abs(np.diff(arr, axis=1)))
        )

        if not exif:
            score += 16
            reasons.append("Sin metadatos EXIF")
        if fmt.lower() in {"png", "webp"}:
            score += 7
            reasons.append(f"Formato {fmt} comun en contenido exportado")
        if megapixels > 2 and file_size < 900_000:
            score += 11
            reasons.append("Alta resolucion con compresion inusual")
        if channel_std < 35:
            score += 8
            reasons.append("Baja variacion cromatica")
        if edge_energy < 18:
            score += 8
            reasons.append("Textura muy uniforme")

        reasons.append(f"Imagen {width}x{height}")
    except Exception as caught:
        score += 20
        reasons.append(f"No se pudo extraer estadistica de imagen: {caught.__class__.__name__}")

    return _make_result(path, "image", score, reasons, error)


def _audio_fallback(path: str, error: Exception) -> dict:
    reasons = []
    score = 24

    try:
        import librosa
        import numpy as np

        waveform, sr = librosa.load(path, sr=16000, mono=True, duration=35)
        duration = len(waveform) / sr if sr else 0

        if duration < 2:
            score += 14
            reasons.append("Audio muy corto")

        rms = librosa.feature.rms(y=waveform)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        flatness = librosa.feature.spectral_flatness(y=waveform)[0]

        rms_std = float(np.std(rms))
        zcr_mean = float(np.mean(zcr))
        flatness_mean = float(np.mean(flatness))

        if rms_std < 0.012:
            score += 12
            reasons.append("Energia de voz muy estable")
        if zcr_mean < 0.035 or zcr_mean > 0.18:
            score += 8
            reasons.append("Cruces por cero fuera de rango habitual")
        if flatness_mean > 0.22:
            score += 10
            reasons.append("Ruido espectral elevado")

        reasons.append(f"Audio analizado: {duration:.1f}s")
    except Exception as caught:
        score += 22
        reasons.append(f"No se pudo extraer estadistica de audio: {caught.__class__.__name__}")

    return _make_result(path, "audio", score, reasons, error)


def _video_fallback(path: str, error: Exception) -> dict:
    reasons = []
    score = 26

    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = max(total_frames, 1)
        step = max(frame_count // 8, 1)
        samples = []

        for index in range(0, frame_count, step):
            if len(samples) >= 8:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                samples.append(cv2.resize(gray, (96, 54)))
        cap.release()

        if not samples:
            score += 28
            reasons.append("No se pudieron leer frames")
        else:
            frame_std = float(np.mean([np.std(frame) for frame in samples]))
            diffs = [
                float(np.mean(np.abs(samples[i].astype(float) - samples[i - 1].astype(float))))
                for i in range(1, len(samples))
            ]
            motion = float(np.mean(diffs)) if diffs else 0

            if frame_std < 32:
                score += 9
                reasons.append("Frames con textura uniforme")
            if len(samples) > 2 and motion < 4:
                score += 9
                reasons.append("Movimiento visual muy bajo")
            if fps > 0 and (fps < 10 or fps > 90):
                score += 7
                reasons.append("FPS fuera de rango habitual")

            reasons.append(f"Frames muestreados: {len(samples)}")
    except Exception as caught:
        score += 24
        reasons.append(f"No se pudo extraer estadistica de video: {caught.__class__.__name__}")

    return _make_result(path, "video", score, reasons, error)


def _make_result(
    path: str,
    modality: str,
    score: float,
    reasons: list[str],
    error: Exception,
) -> dict:
    ai_probability = round(min(max(score, 0), 100), 1)
    label = "synthetic" if ai_probability >= 55 else "real"
    confidence = ai_probability / 100 if label == "synthetic" else 1 - ai_probability / 100

    return {
        "modality": modality,
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {
            "synthetic": round(ai_probability / 100, 4),
            "real": round(1 - ai_probability / 100, 4),
        },
        "ai_probability": ai_probability,
        "reasons": reasons,
        "source": "local_heuristic_fallback",
        "model_error": _short_error(error),
        "file_name": Path(path).name,
    }


def _short_error(error: Exception) -> str:
    message = str(error).replace("\n", " ").strip()
    return f"{error.__class__.__name__}: {message[:180]}"
