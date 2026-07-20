"""
link_detector.py
----------------
Analisis local de enlaces para detectar senales comunes de fraude.
No sustituye un motor antiphishing comercial; sirve para la demo cuando
el enlace no apunta directamente a imagen, audio o video.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
}

SUSPICIOUS_TERMS = (
    "login",
    "verify",
    "wallet",
    "airdrop",
    "premio",
    "regalo",
    "urgente",
    "soporte",
    "bloqueada",
    "password",
    "clave",
    "transferencia",
    "token",
)

DANGEROUS_EXTENSIONS = (".exe", ".scr", ".bat", ".cmd", ".msi", ".apk", ".js")


def analyze_link(url: str) -> dict:
    parsed = urlparse(url.strip())
    reasons: list[str] = []
    score = 8

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _result(url, 95, ["URL invalida o sin esquema http/https"])

    host = (parsed.hostname or "").lower()
    path_query = f"{parsed.path}?{parsed.query}".lower()

    if parsed.scheme != "https":
        score += 15
        reasons.append("No usa HTTPS")

    if parsed.username or parsed.password or "@" in parsed.netloc:
        score += 25
        reasons.append("Incluye credenciales o @ en el dominio")

    if _is_ip(host):
        score += 20
        reasons.append("Usa una IP como destino")

    if "xn--" in host:
        score += 20
        reasons.append("Dominio punycode, posible suplantacion visual")

    if host in SHORTENERS:
        score += 15
        reasons.append("Usa acortador de enlaces")

    if len(url) > 120:
        score += 10
        reasons.append("URL muy larga")
    if len(url) > 200:
        score += 10
        reasons.append("URL extremadamente larga")

    if host.count(".") >= 3:
        score += 10
        reasons.append("Muchos subdominios")

    if host.count("-") >= 2:
        score += 6
        reasons.append("Dominio con varios guiones")

    matched_terms = [term for term in SUSPICIOUS_TERMS if term in path_query or term in host]
    if matched_terms:
        score += min(24, len(matched_terms) * 8)
        reasons.append(f"Terminos sensibles: {', '.join(matched_terms[:4])}")

    if parsed.path.lower().endswith(DANGEROUS_EXTENSIONS):
        score += 25
        reasons.append("Apunta a un ejecutable o instalador")

    if not reasons:
        reasons.append("Sin senales fuertes en la URL")

    return _result(url, score, reasons)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _result(url: str, score: int, reasons: list[str]) -> dict:
    ai_probability = round(min(max(score, 0), 100), 1)
    label = "suspicious" if ai_probability >= 40 else "real"
    confidence = ai_probability / 100 if label == "suspicious" else 1 - ai_probability / 100

    return {
        "modality": "link",
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {
            "suspicious": round(ai_probability / 100, 4),
            "real": round(1 - ai_probability / 100, 4),
        },
        "ai_probability": ai_probability,
        "reasons": reasons,
        "url": re.sub(r"[\r\n]", "", url)[:500],
        "source": "local_link_heuristic",
    }
