"""OpenCV-based license-plate localisation.

Finds plate-shaped rectangles (edge map -> contours -> aspect-ratio / area filters) so the OCR
engine only has to read a small, tight crop instead of the whole photo. Each candidate also
carries its four corners so the crop can be straightened with a perspective transform.
"""

from __future__ import annotations

from dataclasses import dataclass

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
_QUAD_PAD_FRAC = 0.04  # grow the quad slightly so the plate border is not clipped
_MIN_WARP_WIDTH = 320  # upscale small plates to at least this width


@dataclass(frozen=True)
class PlateCandidate:
    box: Box  # padded axis-aligned box
    quad: np.ndarray  # 4x2 float32 corners (tl, tr, br, bl) in the located image's pixels


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return four points ordered top-left, top-right, bottom-right, bottom-left."""

    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype="float32",
    )


def four_point_transform(
    rgb: np.ndarray, quad: np.ndarray, min_width: int = _MIN_WARP_WIDTH
) -> tuple[np.ndarray, np.ndarray]:
    """Straighten the quadrilateral ``quad`` of ``rgb`` into an upright rectangle.

    Returns ``(warped, matrix)`` where ``matrix`` maps source pixels to warped pixels (invert it
    to map OCR boxes back onto the original image).
    """

    tl, tr, br, bl = order_points(quad)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if width < 1 or height < 1:
        raise ValueError("degenerate plate quadrilateral")
    if width < min_width:  # upscale small plates so glyph strokes are thick enough to read
        height *= min_width / width
        width = float(min_width)
    w, h = round(width), round(height)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    warped = cv2.warpPerspective(rgb, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, matrix


def _contour_quad(contour: np.ndarray) -> np.ndarray:
    """Four corners of ``contour``: a polygon approximation, else its min-area rectangle."""

    hull = cv2.convexHull(contour)
    approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
    if len(approx) == 4:
        return order_points(approx.reshape(4, 2))
    return order_points(cv2.boxPoints(cv2.minAreaRect(contour)))


def _expand_quad(quad: np.ndarray, frac: float) -> np.ndarray:
    centre = quad.mean(axis=0)
    return (centre + (quad - centre) * (1.0 + 2 * frac)).astype("float32")


def _plate_colour_fraction(rgb: np.ndarray, box: Box) -> float:
    """Share of white / yellow pixels in ``box`` (plates are light; the frame is not)."""

    x1, y1, x2, y2 = box
    hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    if hsv.size == 0:
        return 0.0
    white = cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))
    yellow = cv2.inRange(hsv, (15, 80, 120), (35, 255, 255))
    return float(np.count_nonzero(white | yellow)) / (hsv.shape[0] * hsv.shape[1])


def locate_plate_candidates(image: Image.Image, max_candidates: int = 3) -> list[PlateCandidate]:
    """Return up to ``max_candidates`` plate-like candidates (box + corners), best first."""

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

    scored: list[tuple[float, PlateCandidate]] = []
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
        # Colour only nudges the ranking: rental plates vary, so it is never a hard filter.
        colour = 0.5 + _plate_colour_fraction(rgb, (x, y, x + w, y + h))
        quad = _expand_quad(_contour_quad(contour), _QUAD_PAD_FRAC)
        score = aspect_fit * rectangularity * area_frac**0.5 * colour
        scored.append((score, PlateCandidate(box=box, quad=quad)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [cand for _, cand in scored[:max_candidates]]


def locate_plate_regions(image: Image.Image, max_candidates: int = 3) -> list[Box]:
    """Return up to ``max_candidates`` padded plate-like boxes, best first."""

    return [c.box for c in locate_plate_candidates(image, max_candidates)]
