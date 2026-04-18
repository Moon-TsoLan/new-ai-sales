import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings

settings = get_settings()
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _validate_upload(image_file: UploadFile) -> str:
    ext = Path(image_file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only JPG/PNG image allowed.")
    return ext


def save_user_upload(user_id: int, image_file: UploadFile, submission_id: str) -> str:
    ext = _validate_upload(image_file)
    user_dir = settings.upload_root / str(user_id) / submission_id
    user_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}{ext}"
    file_path = user_dir / filename

    raw = image_file.file.read()
    size_limit = settings.max_upload_size_mb * 1024 * 1024
    if len(raw) > size_limit:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image too large (max 5MB).")

    with open(file_path, "wb") as f:
        f.write(raw)
    return str(file_path.relative_to(settings.upload_root)).replace("\\", "/")


def remove_relative_path(path_str: str) -> None:
    path = settings.upload_root / path_str
    if path.exists():
        path.unlink()
    parent = path.parent
    if parent.exists() and parent != settings.upload_root:
        try:
            os.rmdir(parent)
        except OSError:
            pass


def remove_relative_dir(dir_str: str) -> None:
    path = settings.upload_root / dir_str
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)

