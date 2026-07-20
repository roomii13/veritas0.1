"""
pipeline.py
-----------
Pipeline combinado del demo Veritas.

Uso:
    python pipeline.py --image foto.jpg --audio llamada.wav --video reunion.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Iterable

from audio_detector import analyze_audio
from image_detector import analyze_image
from video_detector import analyze_video

FAKE_TOKENS = (
    "fake",
    "spoof",
    "deepfake",
    "synthetic",
    "generated",
    "suspicious",
    "phishing",
    "malicious",
)
REAL_TOKENS = ("real", "bonafide", "genuine", "human", "legit", "ok")


def build_report(image_path=None, audio_path=None, video_path=None) -> dict:
    """
    Corre los detectores indicados y arma un reporte combinado.
    """
    results = []

    if image_path:
        _assert_exists(image_path)
        print(f"Analizando imagen: {image_path}")
        results.append(analyze_image(image_path))

    if audio_path:
        _assert_exists(audio_path)
        print(f"Analizando audio: {audio_path}")
        results.append(analyze_audio(audio_path))

    if video_path:
        _assert_exists(video_path)
        print(f"Analizando video: {video_path}")
        results.append(analyze_video(video_path))

    if not results:
        raise ValueError("Debes proveer al menos un archivo (--image, --audio o --video).")

    return aggregate_results(results)


def aggregate_results(results: Iterable[dict]) -> dict:
    """
    Combina modalidades en un unico porcentaje de IA/riesgo.
    """
    normalized_results = []
    per_modality = []
    ai_values = []
    high_confidence_fake = False

    for raw in results:
        result = dict(raw)
        label = str(result.get("label", "")).lower()
        confidence = float(result.get("confidence", 0.0))
        ai_probability = _ai_probability_from_result(result)
        is_fake = _label_has_token(label, FAKE_TOKENS) or ai_probability >= 55

        if is_fake and (confidence > 0.85 or ai_probability >= 85):
            high_confidence_fake = True

        result["ai_probability"] = round(ai_probability, 1)
        normalized_results.append(result)
        ai_values.append(ai_probability)
        per_modality.append(
            {
                "modality": result.get("modality", "unknown"),
                "verdict": "SOSPECHOSO" if is_fake else "OK",
                "confidence": round(confidence, 4),
                "ai_probability": round(ai_probability, 1),
            }
        )

    if not ai_values:
        raise ValueError("No hay resultados para agregar.")

    average_ai = sum(ai_values) / len(ai_values)
    max_ai = max(ai_values)
    ai_percentage = max(average_ai, max_ai * 0.75)
    if high_confidence_fake:
        ai_percentage = max(ai_percentage, 85)
    ai_percentage = round(min(max(ai_percentage, 0), 100), 1)

    if ai_percentage >= 70:
        overall = "ALTO RIESGO"
        recommendation = (
            "NO CONFIAR. Verifica por otro canal, no transfieras dinero, no abras adjuntos "
            "y solicita validacion humana antes de actuar."
        )
    elif ai_percentage >= 40:
        overall = "RIESGO MEDIO"
        recommendation = (
            "Revisar con cautela. Busca otra fuente, confirma identidad y evita decisiones "
            "urgentes basadas solo en este contenido."
        )
    else:
        overall = "BAJO RIESGO"
        recommendation = (
            "No hay senales fuertes de IA en este demo, pero no reemplaza una verificacion "
            "manual si hay dinero, datos personales o presion por urgencia."
        )

    return {
        "overall_verdict": overall,
        "risk_score": ai_percentage,
        "ai_percentage": ai_percentage,
        "modalities_analyzed": len(normalized_results),
        "per_modality_summary": per_modality,
        "recommendation": recommendation,
        "warning": ai_percentage >= 70,
        "raw_results": normalized_results,
    }


def _assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")


def _label_has_token(label: str, tokens: tuple[str, ...]) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", label.lower())
    return any(token in compact for token in tokens)


def _ai_probability_from_result(result: dict) -> float:
    if "ai_probability" in result:
        return _clamp_percent(float(result["ai_probability"]))

    per_frame = result.get("per_frame")
    if isinstance(per_frame, list) and per_frame:
        return _clamp_percent(
            sum(_ai_probability_from_result(frame) for frame in per_frame) / len(per_frame)
        )

    scores = result.get("scores")
    if isinstance(scores, dict) and scores:
        fake_scores = [
            float(value)
            for label, value in scores.items()
            if _label_has_token(str(label), FAKE_TOKENS)
        ]
        real_scores = [
            float(value)
            for label, value in scores.items()
            if _label_has_token(str(label), REAL_TOKENS)
        ]

        if fake_scores:
            return _clamp_percent(sum(fake_scores) * 100)
        if real_scores:
            return _clamp_percent((1 - max(real_scores)) * 100)

    if "fake_frame_ratio" in result:
        return _clamp_percent(float(result["fake_frame_ratio"]) * 100)

    label = str(result.get("label", ""))
    confidence = float(result.get("confidence", 0.0))
    if _label_has_token(label, FAKE_TOKENS):
        return _clamp_percent(confidence * 100)
    if _label_has_token(label, REAL_TOKENS):
        return _clamp_percent((1 - confidence) * 100)
    return _clamp_percent(confidence * 100)


def _clamp_percent(value: float) -> float:
    return min(max(value, 0.0), 100.0)


def _print_pretty(report: dict):
    print("\n" + "=" * 56)
    print(" REPORTE VERITAS - DETECCION DE IA / FRAUDE")
    print("=" * 56)
    print(f" Veredicto general : {report['overall_verdict']}")
    print(f" Porcentaje IA     : {report['ai_percentage']} / 100")
    print(f" Modalidades       : {report['modalities_analyzed']}")
    print("-" * 56)
    for item in report["per_modality_summary"]:
        print(
            f"  - {item['modality']:6s} -> {item['verdict']:10s} "
            f"(IA: {item['ai_probability']:5.1f}%, confianza: {item['confidence']:.2%})"
        )
    print("-" * 56)
    print(f" Recomendacion     : {report['recommendation']}")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Veritas de deteccion de IA/fraude")
    parser.add_argument("--image", help="Ruta a un archivo de imagen")
    parser.add_argument("--audio", help="Ruta a un archivo de audio")
    parser.add_argument("--video", help="Ruta a un archivo de video")
    parser.add_argument("--json", action="store_true", help="Imprimir JSON crudo")
    args = parser.parse_args()

    report = build_report(image_path=args.image, audio_path=args.audio, video_path=args.video)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_pretty(report)
