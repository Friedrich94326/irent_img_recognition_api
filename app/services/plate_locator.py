"""OpenCV-based license-plate localisation.

Finds plate-shaped rectangles (edge map -> contours -> aspect-ratio / area filters) so the OCR
engine only has to read a small, tight crop instead of the whole photo.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels

# Taiwan car plates are ~2.1:1 (32x15 cm); allow slack for perspective and motorcycle plates.
_MIN_ASPECT = 1.3
_MAX_ASPECT = 5.5
_MIN_AREA_FRAC = 0.002
_MAX_AREA_FRAC = 0.5
_MIN_RECTANGULARITY = 0.6
_PAD_FRAC = 0.08


def locate_plate_regions(image: Image.Image, max_candidates: int = 3) -> list[Box]:
    """Return up to ``max_candidates`` padded plate-like boxes, best first."""

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    image_area = float(height * width)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # smooth texture, keep plate edges
    edges = cv2.Canny(gray, 30, 200)
    # Close small gaps so the plate border and characters merge into one blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scored: list[tuple[float, Box]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        aspect = w / h
        area_frac = (w * h) / image_area
        if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
            continue
        if not (_MIN_AREA_FRAC <= area_frac <= _MAX_AREA_FRAC):
            continue
        rectangularity = cv2.contourArea(contour) / float(w * h)
        if rectangularity < _MIN_RECTANGULARITY:
            continue
        # Prefer boxes whose aspect is near a car plate, then larger, more rectangular ones.
        aspect_fit = 1.0 / (1.0 + abs(aspect - 2.1))
        pad_x, pad_y = int(w * _PAD_FRAC), int(h * _PAD_FRAC)
        box = (
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(width, x + w + pad_x),
            min(height, y + h + pad_y),
        )
        scored.append((aspect_fit * rectangularity * area_frac**0.5, box))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in scored[:max_candidates]]
