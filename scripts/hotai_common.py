"""Shared helpers for the Hotai iRent dataset scripts (evaluation + fine-tuning prep).

Most images in this dataset were shot in portrait (phone held vertically) and either
carry an EXIF rotation tag or are stored pre-rotated. :func:`orient_horizontal` corrects
true orientation and rotates any still-portrait image into landscape (car lying
horizontal), since the trained detector expects landscape car photos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif"}

BOX_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
]


@dataclass(frozen=True)
class OrientedImage:
    """Result of :func:`orient_horizontal`."""

    image: Image.Image
    rotated: bool
    """True if the image was rotated 90 degrees (``Image.Transpose.ROTATE_90``) to turn
    a portrait, EXIF-corrected photo into landscape."""
    pre_rotation_size: tuple[int, int]
    """``(width, height)`` after EXIF correction but before the extra rotation above -
    i.e. the coordinate space labelme annotations are defined in."""


def orient_horizontal(image: Image.Image) -> OrientedImage:
    """Undo EXIF rotation, then rotate any still-portrait image into landscape."""

    corrected = ImageOps.exif_transpose(image)
    pre_rotation_size = corrected.size
    if corrected.height > corrected.width:
        rotated = corrected.transpose(Image.Transpose.ROTATE_90)
        return OrientedImage(rotated.convert("RGB"), True, pre_rotation_size)
    return OrientedImage(corrected.convert("RGB"), False, pre_rotation_size)


def find_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def labelme_shapes(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("shapes", [])


def labelme_labels(json_path: Path) -> list[str]:
    return [shape["label"] for shape in labelme_shapes(json_path)]


_CLASS_COLOR_CACHE: dict[str, str] = {}


def _color_for_class(name: str) -> str:
    if name not in _CLASS_COLOR_CACHE:
        _CLASS_COLOR_CACHE[name] = BOX_COLORS[len(_CLASS_COLOR_CACHE) % len(BOX_COLORS)]
    return _CLASS_COLOR_CACHE[name]


def draw_predictions(image: Image.Image, detections: list) -> Image.Image:
    """Draw predicted damage boxes (class + confidence) over a copy of ``image``.

    ``detections`` items only need ``damage_class`` (an enum with ``.value``),
    ``confidence`` and ``xyxy`` attributes - i.e. evaluator.RawDetection, without
    importing the app package into this script-only module.
    """

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        label = getattr(det.damage_class, "value", str(det.damage_class))
        color = _color_for_class(label)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {det.confidence:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color, font=font)
    return annotated
