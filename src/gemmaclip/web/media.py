from __future__ import annotations

import mimetypes
from pathlib import Path


SUPPORTED_EXTENSIONS = {".mp4", ".webm", ".mov"}
SUPPORTED_MIME_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "application/octet-stream",
}


class UploadValidationError(ValueError):
    pass


def validate_upload_name(filename: str | None, content_type: str | None) -> str:
    suffix = Path(str(filename or "").replace("\\", "/").split("/")[-1]).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError("Supported video formats are MP4, WebM, and MOV.")
    normalized_type = str(content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type and normalized_type not in SUPPORTED_MIME_TYPES:
        raise UploadValidationError("The uploaded file does not have a supported video MIME type.")
    return suffix


def media_type_for_path(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
