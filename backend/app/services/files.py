import os
import re
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import settings

SAFE_EXT = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
}

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "file")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base[:180]


def validate_upload(file: UploadFile, allowed: set[str] | None = None) -> str:
    orig = sanitize_filename(file.filename or "upload.bin")
    ext = Path(orig).suffix.lower()
    allowed = allowed or set(SAFE_EXT)
    if ext not in allowed:
        raise HTTPException(400, f"File type {ext or '(none)'} is not allowed")
    if ext in {".exe", ".js", ".bat", ".cmd", ".sh", ".php"}:
        raise HTTPException(400, "Executable uploads are not permitted")
    return ext


def save_upload(file: UploadFile, folder: str, allowed: set[str] | None = None) -> tuple[str, str, int]:
    ext = validate_upload(file, allowed)
    dest_dir = settings.upload_path / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{ext}"
    path = dest_dir / name
    data = file.file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(400, f"File exceeds {settings.MAX_UPLOAD_MB} MB limit")
    if not data:
        raise HTTPException(400, "Empty file")
    path.write_bytes(data)
    if ext in IMAGE_EXT:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.thumbnail((1200, 1200))
                img.save(path)
        except Exception:
            path.unlink(missing_ok=True)
            raise HTTPException(400, "Invalid or corrupt image file")
    return str(path), file.filename or name, len(data)


def safe_resolve(path: str) -> Path:
    root = settings.upload_path.resolve()
    target = Path(path).resolve()
    if root not in target.parents and target != root:
        if not str(target).startswith(str(root)):
            raise HTTPException(403, "Invalid file path")
    if not target.exists():
        raise HTTPException(404, "File not found")
    return target
