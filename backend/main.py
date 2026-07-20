from __future__ import annotations

import ipaddress
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from link_detector import analyze_link
from pipeline import aggregate_results

from .analysis_service import analyze_path

MAX_UPLOAD_BYTES = int(os.getenv("VERITAS_MAX_UPLOAD_MB", "120")) * 1024 * 1024
ALLOW_PRIVATE_URLS = os.getenv("VERITAS_ALLOW_PRIVATE_URLS", "false").lower() in {
    "1",
    "true",
    "yes",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(
    title="Veritas API",
    description="Backend local para deteccion demo de IA/fraude en imagen, audio, video y enlaces.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }


@app.middleware("http")
async def ensure_cors_on_errors(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse({"ok": True}, headers=_cors_headers())

    try:
        response = await call_next(request)
    except Exception as exc:
        return JSONResponse(
            {
                "detail": (
                    "Error interno del backend Veritas. "
                    f"{exc.__class__.__name__}: {str(exc)[:220]}"
                )
            },
            status_code=500,
            headers=_cors_headers(),
        )

    for header, value in _cors_headers().items():
        response.headers.setdefault(header, value)
    return response


class UrlPayload(BaseModel):
    url: str


class NotDirectMedia(Exception):
    pass


@app.get("/")
def root():
    return {"name": "Veritas API", "status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "heuristic_fallback": os.getenv("VERITAS_ALLOW_HEURISTIC_FALLBACK", "true"),
    }


@app.post("/analyze-file")
async def analyze_file_endpoint(file: UploadFile = File(...)):
    modality = _infer_modality(file.filename or "", file.content_type or "")
    if modality is None:
        raise HTTPException(
            status_code=400,
            detail="Formato no soportado. Usa imagen, audio o video comun.",
        )

    path = await _save_upload(file)
    try:
        result = await run_in_threadpool(analyze_path, path, modality)
        report = aggregate_results([result])
        report["input"] = {
            "type": "file",
            "name": file.filename,
            "content_type": file.content_type,
            "modality": modality,
        }
        return report
    finally:
        _safe_unlink(path)


@app.post("/analyze-url")
async def analyze_url_endpoint(payload: UrlPayload):
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Debes enviar una URL.")

    try:
        path, modality, content_type = await run_in_threadpool(_download_direct_media, url)
    except NotDirectMedia as exc:
        report = aggregate_results([analyze_link(url)])
        report["input"] = {
            "type": "link",
            "url": url,
            "downloaded": False,
            "note": str(exc),
        }
        return report
    except Exception as exc:
        report = aggregate_results([analyze_link(url)])
        report["input"] = {
            "type": "link",
            "url": url,
            "downloaded": False,
            "download_error": f"{exc.__class__.__name__}: {str(exc)[:160]}",
        }
        return report

    try:
        result = await run_in_threadpool(analyze_path, path, modality)
        report = aggregate_results([result])
        report["input"] = {
            "type": "url-media",
            "url": url,
            "downloaded": True,
            "content_type": content_type,
            "modality": modality,
        }
        return report
    finally:
        _safe_unlink(path)


async def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        total = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close()
                _safe_unlink(tmp.name)
                raise HTTPException(status_code=413, detail="Archivo demasiado grande.")
            tmp.write(chunk)
        return tmp.name


def _download_direct_media(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise NotDirectMedia("URL invalida para descarga directa.")

    if not ALLOW_PRIVATE_URLS and _host_is_private(parsed.hostname or ""):
        raise NotDirectMedia("URL privada o local bloqueada para descarga; se analiza como enlace.")

    response = requests.get(
        url,
        stream=True,
        timeout=(8, 60),
        headers={"User-Agent": "VeritasDemo/1.0"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").split(";")[0].lower()
    modality = _infer_modality(parsed.path, content_type)
    if modality is None:
        response.close()
        raise NotDirectMedia("El enlace no parece apuntar directo a imagen/audio/video.")

    suffix = Path(parsed.path).suffix or _suffix_for_content_type(content_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close()
                response.close()
                _safe_unlink(tmp.name)
                raise HTTPException(status_code=413, detail="Archivo remoto demasiado grande.")
            tmp.write(chunk)
        response.close()
        return tmp.name, modality, content_type


def _infer_modality(name: str, content_type: str) -> str | None:
    content_type = content_type.lower()
    suffix = Path(name.split("?")[0]).suffix.lower()

    if content_type.startswith("image/") or suffix in IMAGE_EXTS:
        return "image"
    if content_type.startswith("audio/") or suffix in AUDIO_EXTS:
        return "audio"
    if content_type.startswith("video/") or suffix in VIDEO_EXTS:
        return "video"
    return None


def _suffix_for_content_type(content_type: str) -> str:
    if content_type == "image/jpeg":
        return ".jpg"
    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    if content_type in {"audio/mpeg", "audio/mp3"}:
        return ".mp3"
    if content_type in {"audio/wav", "audio/x-wav"}:
        return ".wav"
    if content_type == "video/mp4":
        return ".mp4"
    if content_type == "video/quicktime":
        return ".mov"
    return ".bin"


def _host_is_private(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
