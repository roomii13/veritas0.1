"""
link_detector.py
----------------
Analisis local de enlaces para detectar senales comunes de phishing/fraude.

Configuracion:
    LINK_TRUSTED_DOMAINS=google.com,youtube.com,wikipedia.org,...
    LINK_EXPAND_SHORTENERS=false
    LINK_SUSPICIOUS_THRESHOLD=45
    LINK_WEIGHT_NO_HTTPS=15
    LINK_WEIGHT_SHORTENER=18
    ...
"""

from __future__ import annotations

from functools import lru_cache
import ipaddress
import os
import re
from urllib.parse import urlparse

TRUSTED_DOMAINS = {
    "google.com",
    "youtube.com",
    "youtu.be",
    "wikipedia.org",
    "wikimedia.org",
    "github.com",
    "microsoft.com",
    "apple.com",
    "openai.com",
    "cloudflare.com",
}

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
    "lnkd.in",
    "shorturl.at",
    "s.id",
}

SUSPICIOUS_TERMS = (
    "login",
    "verify",
    "verification",
    "wallet",
    "airdrop",
    "premio",
    "regalo",
    "urgente",
    "soporte",
    "bloqueada",
    "password",
    "passwd",
    "clave",
    "transferencia",
    "token",
    "seed",
    "recovery",
    "bonus",
    "free",
    "claim",
    "account",
    "secure",
    "security",
    "update",
    "invoice",
    "refund",
    "pago",
    "mercadopago",
    "whatsapp",
)

DANGEROUS_EXTENSIONS = (
    ".exe",
    ".scr",
    ".bat",
    ".cmd",
    ".msi",
    ".apk",
    ".js",
    ".jar",
    ".vbs",
    ".ps1",
    ".iso",
    ".lnk",
)

DEFAULT_WEIGHTS = {
    "base": 6,
    "no_https": 15,
    "credentials": 25,
    "ip_host": 22,
    "punycode": 24,
    "shortener": 18,
    "long_url": 10,
    "very_long_url": 10,
    "many_subdomains": 10,
    "many_hyphens": 7,
    "suspicious_term": 8,
    "dangerous_extension": 28,
    "trusted_domain_discount": -18,
}


def _weights() -> dict[str, int]:
    return {
        key: int(os.getenv(f"LINK_WEIGHT_{key.upper()}", str(value)))
        for key, value in DEFAULT_WEIGHTS.items()
    }


@lru_cache(maxsize=1024)
def analyze_link(url: str) -> dict:
    parsed = urlparse(url.strip())
    reasons: list[str] = []
    weights = _weights()
    score = weights["base"]

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _result(url, 95, ["URL invalida o sin esquema http/https"], False)

    host = (parsed.hostname or "").lower()
    path_query = f"{parsed.path}?{parsed.query}".lower()
    trusted = _is_trusted_domain(host)

    if parsed.scheme != "https":
        score += weights["no_https"]
        reasons.append("No usa HTTPS")

    if parsed.username or parsed.password or "@" in parsed.netloc:
        score += weights["credentials"]
        reasons.append("Incluye credenciales o @ en el dominio")

    if _is_ip(host):
        score += weights["ip_host"]
        reasons.append("Usa una IP como destino")

    if "xn--" in host:
        score += weights["punycode"]
        reasons.append("Dominio punycode, posible suplantacion visual")

    expanded_url = None
    if host in SHORTENERS:
        score += weights["shortener"]
        reasons.append("Usa acortador de enlaces")
        expanded_url = _expand_shortener(url) if _expand_shorteners() else None

    if len(url) > 120:
        score += weights["long_url"]
        reasons.append("URL muy larga")
    if len(url) > 200:
        score += weights["very_long_url"]
        reasons.append("URL extremadamente larga")

    if host.count(".") >= 3:
        score += weights["many_subdomains"]
        reasons.append("Muchos subdominios")

    if host.count("-") >= 2:
        score += weights["many_hyphens"]
        reasons.append("Dominio con varios guiones")

    matched_terms = sorted(
        {term for term in SUSPICIOUS_TERMS if term in path_query or term in host}
    )
    if matched_terms:
        score += min(32, len(matched_terms) * weights["suspicious_term"])
        reasons.append(f"Terminos sensibles: {', '.join(matched_terms[:5])}")

    if parsed.path.lower().endswith(DANGEROUS_EXTENSIONS):
        score += weights["dangerous_extension"]
        reasons.append("Apunta a un ejecutable, script o instalador")

    if trusted:
        score += weights["trusted_domain_discount"]
        reasons.append("Dominio en lista confiable")

    if expanded_url and expanded_url != url:
        reasons.append(f"Redirecciona a: {_sanitize_url(expanded_url)}")

    if not reasons:
        reasons.append("Sin senales fuertes en la URL")

    return _result(url, score, reasons, trusted, expanded_url)


def _is_trusted_domain(host: str) -> bool:
    configured = os.getenv("LINK_TRUSTED_DOMAINS")
    trusted_domains = (
        {domain.strip().lower() for domain in configured.split(",") if domain.strip()}
        if configured
        else TRUSTED_DOMAINS
    )
    return any(host == domain or host.endswith(f".{domain}") for domain in trusted_domains)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _expand_shorteners() -> bool:
    return os.getenv("LINK_EXPAND_SHORTENERS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _expand_shortener(url: str) -> str | None:
    try:
        import requests

        response = requests.head(
            url,
            allow_redirects=True,
            timeout=(3, 5),
            headers={"User-Agent": "VeritasDemo/1.0"},
        )
        return response.url
    except Exception:
        return None


def _result(
    url: str,
    score: int | float,
    reasons: list[str],
    trusted: bool,
    expanded_url: str | None = None,
) -> dict:
    suspicious_threshold = float(os.getenv("LINK_SUSPICIOUS_THRESHOLD", "45"))
    ai_probability = round(min(max(float(score), 0), 100), 1)
    label = "suspicious" if ai_probability >= suspicious_threshold else "real"
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
        "threshold": suspicious_threshold,
        "reasons": reasons,
        "trusted_domain": trusted,
        "expanded_url": _sanitize_url(expanded_url) if expanded_url else None,
        "url": _sanitize_url(url),
        "source": "local_link_heuristic",
    }


def _sanitize_url(url: str) -> str:
    return re.sub(r"[\r\n]", "", url)[:500]
