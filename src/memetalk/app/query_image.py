from __future__ import annotations

import base64
import binascii
import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

_MEDIA_TYPE_TO_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_PIL_FORMAT_TO_SUFFIX = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}
_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_prefix(filename: str | None) -> str:
    stem = Path(filename or "query").stem.strip()
    filtered = "".join(char if char.isalnum() else "-" for char in stem)
    compact = "-".join(part for part in filtered.split("-") if part)
    return compact[:40] or "query"


def decode_base64_image(payload: str) -> tuple[bytes, str | None]:
    text = payload.strip()
    detected_media_type: str | None = None
    if text.startswith("data:"):
        header, separator, encoded = text.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Invalid data URL image payload.")
        detected_media_type = header[5:].split(";", 1)[0] or None
        text = encoded
    try:
        image_bytes = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload.") from exc
    if not image_bytes:
        raise ValueError("Image payload must not be empty.")
    return image_bytes, detected_media_type


def infer_image_suffix(
    image_bytes: bytes,
    filename: str | None = None,
    media_type: str | None = None,
) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in _SUPPORTED_SUFFIXES:
        return suffix
    if media_type:
        normalized_media_type = media_type.strip().lower()
        if normalized_media_type in _MEDIA_TYPE_TO_SUFFIX:
            return _MEDIA_TYPE_TO_SUFFIX[normalized_media_type]
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format in _PIL_FORMAT_TO_SUFFIX:
                return _PIL_FORMAT_TO_SUFFIX[image.format]
    except Exception as exc:
        raise ValueError("Unsupported or unreadable image payload.") from exc
    raise ValueError("Unsupported image format. Use jpg, jpeg, png, or webp.")


@contextmanager
def temporary_query_image(
    image_bytes: bytes,
    filename: str | None = None,
    media_type: str | None = None,
):
    suffix = infer_image_suffix(image_bytes, filename=filename, media_type=media_type)
    handle, raw_path = tempfile.mkstemp(prefix=f"memetalk-query-{_safe_prefix(filename)}-", suffix=suffix)
    os.close(handle)
    path = Path(raw_path)
    try:
        path.write_bytes(image_bytes)
        yield path
    finally:
        path.unlink(missing_ok=True)
