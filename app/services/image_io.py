"""Read and validate an uploaded image into a Pillow image + metadata."""

from __future__ import annotations

import io

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings
from app.core.exceptions import (
    InvalidImageError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.schemas.common import ImageMetadata

_CHUNK_SIZE = 64 * 1024


async def load_upload(
    file: UploadFile, settings: Settings
) -> tuple[Image.Image, ImageMetadata]:
    """Validate ``file`` and return an RGB :class:`PIL.Image.Image` plus its metadata.

    Raises :class:`UnsupportedMediaTypeError` (415), :class:`PayloadTooLargeError` (413)
    or :class:`InvalidImageError` (422) on bad input.
    """

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in settings.allowed_content_types:
        allowed = ", ".join(sorted(settings.allowed_content_types))
        raise UnsupportedMediaTypeError(
            f"Unsupported content type '{content_type or 'unknown'}'. Allowed: {allowed}.",
            field="file",
        )

    # Read with a running cap so an oversized upload is rejected without buffering it all.
    buffer = io.BytesIO()
    size = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        if size > settings.max_image_bytes:
            raise PayloadTooLargeError(
                f"Image exceeds the maximum allowed size of {settings.max_image_bytes} bytes.",
                field="file",
            )
        buffer.write(chunk)

    if size == 0:
        raise InvalidImageError("Uploaded file is empty.", field="file")

    data = buffer.getvalue()
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # cheap structural check; consumes the image object
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Uploaded file is not a readable image.", field="file") from exc

    # Phones store portrait shots sideways plus an EXIF rotation tag; apply it so detectors see
    # the image upright and returned box coordinates match what a browser displays.
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    metadata = ImageMetadata(
        filename=file.filename,
        content_type=content_type,
        width=rgb.width,
        height=rgb.height,
        size_bytes=size,
    )
    return rgb, metadata
