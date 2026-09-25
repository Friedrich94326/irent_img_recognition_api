"""Step-by-step images of classic image processing for web/demo.html (used by build_demo_page.py).

Two stories on one iRent photo:

* **Locating the licence plate** - the intermediates of the real OpenCV pipeline in
  app/services/plate_locator.py (bilateral filter -> Canny -> morphological close -> shape-filtered
  contours -> perspective warp), recomputed with the same parameters.
* **Filtering light reflections** - a concept, not in the API yet: find specular glare (very
  bright, nearly colourless pixels), even out contrast with CLAHE, inpaint the highlights, and
  report honestly whether the damage model's output changes.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.services import plate_locator as pl

STEP_WIDTH = 900
WORK_SIDE = 1600
"""Longest side the plate API locates on (plate_recognizer._WORK_SIDE)."""
GREY, BLUE, ORANGE, RED = (160, 160, 160), (42, 120, 214), (235, 104, 52), (239, 68, 68)
GLARE_V, GLARE_S = 235, 40
"""Specular highlights are near-white: HSV value at least GLARE_V and saturation at most GLARE_S."""


def _save(rgb: np.ndarray, out_dir: Path, name: str) -> str:
    image = Image.fromarray(rgb)
    if image.width > STEP_WIDTH:
        image = image.resize((STEP_WIDTH, round(image.height * STEP_WIDTH / image.width)))
    image.save(out_dir / name, quality=88)
    return f"{out_dir.name}/{name}"


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _thickness(rgb: np.ndarray) -> int:
    return max(2, rgb.shape[1] // 400)


# --- plate -----------------------------------------------------------------------------------


def plate_steps(image: Image.Image, out_dir: Path, plate_detector=None) -> list[dict]:
    # Like PlateRecognizer.read: locate on a copy downscaled to 1600 px, warp from the original.
    full = np.asarray(image.convert("RGB"))
    scale = min(1.0, WORK_SIDE / max(image.size))
    image = image.resize((round(image.width * scale), round(image.height * scale)),
                         Image.Resampling.LANCZOS)
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    area = float(height * width)
    t = _thickness(rgb)

    # Same calls and parameters as plate_locator.locate_plate_candidates.
    gray = cv2.bilateralFilter(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = rgb.copy()
    cv2.drawContours(shapes, contours, -1, GREY, max(1, t // 2))
    passing = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        if (pl._MIN_ASPECT <= w / h <= pl._MAX_ASPECT
                and pl._MIN_AREA_FRAC <= w * h / area <= pl._MAX_AREA_FRAC
                and cv2.contourArea(contour) / float(w * h) >= pl._MIN_RECTANGULARITY):
            cv2.rectangle(shapes, (x, y), (x + w, y + h), BLUE, t)
            passing += 1
    candidates = pl.locate_plate_candidates(image)
    detected = ([tuple(v / scale for v in b) for b in plate_detector(image)]
                if plate_detector else [])
    classic_hit = bool(candidates and detected and _iou(
        tuple(v / scale for v in candidates[0].box), detected[0]) > 0.3)
    if candidates:
        cv2.polylines(shapes, [candidates[0].quad.astype(np.int32)], True, ORANGE, t * 2)
    if not candidates:
        verdict = "none survives, so the classic pipeline finds no plate here."
    elif classic_hit or not detected:
        verdict = ("the best-scoring one, also weighted by white/yellow colour, is the plate "
                   "(orange).")
    else:
        verdict = ("its best pick (orange) is not the plate: the white plate merged into the white "
                   "bumper, so the classic pipeline alone fails on this photo.")

    steps = [
        ("Original photo", "The plate is a small part of a busy scene.", rgb),
        ("Bilateral filter", "Grayscale, then smooth paint texture and noise while keeping sharp "
         "edges such as the plate border.", _gray_to_rgb(gray)),
        ("Canny edges", "Keep only strong intensity changes: outlines, characters, panel gaps.",
         _gray_to_rgb(edges)),
        ("Morphological close", "A wide 17×3 kernel bridges the gaps between characters, so a "
         "plate becomes one solid blob.", _gray_to_rgb(closed)),
        ("Shape filter", f"{len(contours)} contours (grey); {passing} have a plate-like aspect, "
         f"size and rectangularity (blue); {verdict}", shapes),
    ]

    quad = None
    if classic_hit or (candidates and not detected):
        quad = candidates[0].quad / scale
    elif detected:
        box = detected[0]
        found = full.copy()
        cv2.rectangle(found, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]),
                      ORANGE, _thickness(full) * 2)
        steps.append(("Learned plate detector", "So the API asks the YOLO tyre + plate model "
                      "first: it returns a box around the plate (orange).", found))
        quad = pl.refine_plate_quad(full, box)
        window = _refine_view(full, box, quad)
        steps.append(("Local edge refinement", "Plain Canny edges in a small window around that "
                      "box (no wide close) recover the plate's own border, so the corners follow "
                      "its real tilt." if quad is not None else "No clean border in the window, "
                      "so the detector box itself is used as the corners.", window))
        if quad is None:
            x1, y1, x2, y2 = box
            quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
    if quad is not None:
        try:
            warped, _ = pl.four_point_transform(full, quad)
            steps.append(("Perspective warp", "The four corners are mapped to an upright "
                          "rectangle, so OCR reads a straight, enlarged plate.", warped))
        except ValueError:
            pass
    return [{"src": _save(img, out_dir, f"ip_plate_{i}.jpg"), "title": title, "caption": caption}
            for i, (title, caption, img) in enumerate(steps, 1)]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _refine_view(rgb: np.ndarray, box, quad) -> np.ndarray:
    """Window refine_plate_quad searches, with its edge map in blue and the found quad in orange."""
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * pl._REFINE_WINDOW_PAD_FRAC, (y2 - y1) * pl._REFINE_WINDOW_PAD_FRAC
    wx1, wy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    wx2, wy2 = min(rgb.shape[1], int(x2 + pad_x)), min(rgb.shape[0], int(y2 + pad_y))
    crop = rgb[wy1:wy2, wx1:wx2].copy()
    gray = cv2.bilateralFilter(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), 9, 30, 30)
    edges = cv2.dilate(cv2.Canny(gray, 40, 160), np.ones((3, 3), np.uint8))
    crop[edges > 0] = BLUE
    if quad is not None:
        local = quad - np.array([wx1, wy1], dtype="float32")
        cv2.polylines(crop, [local.astype(np.int32)], True, ORANGE, max(2, crop.shape[1] // 120))
    return cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


# --- glare -----------------------------------------------------------------------------------


def glare_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, GLARE_V), (180, GLARE_S, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop specks
    return cv2.dilate(mask, np.ones((7, 7), np.uint8))  # cover the highlight's soft halo


def clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _draw_detections(rgb: np.ndarray, detections) -> np.ndarray:
    image = Image.fromarray(rgb.copy())
    pen = ImageDraw.Draw(image)
    t = _thickness(rgb) * 2
    for d in detections:
        pen.rectangle(d.xyxy, outline=ORANGE, width=t)
        text = f"{d.damage_class.value.replace('_', ' ')} {d.confidence:.2f}"
        left, top, right, bottom = pen.textbbox((0, 0), text, font_size=max(18, rgb.shape[1] // 45))
        y = max(0, d.xyxy[1] - (bottom - top) - 10)
        pen.rectangle([d.xyxy[0], y, d.xyxy[0] + right - left + 12, y + bottom - top + 10],
                      fill=ORANGE)
        pen.text((d.xyxy[0] + 6, y + 5 - top), text, fill=(20, 22, 26),
                 font_size=max(18, rgb.shape[1] // 45))
    return np.asarray(image)


def _describe(detections) -> str:
    if not detections:
        return "nothing detected"
    return ", ".join(f"{d.damage_class.value.replace('_', ' ')} {d.confidence:.2f}"
                     for d in detections)


def glare_steps(image: Image.Image, evaluator, out_dir: Path) -> tuple[list[dict], dict]:
    rgb = np.asarray(image.convert("RGB"))
    mask = glare_mask(rgb)
    share = float((mask > 0).mean())
    overlay = rgb.copy()
    overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array(RED)).astype(np.uint8)
    equalised = clahe(rgb)
    cleaned = cv2.inpaint(equalised, mask, 5, cv2.INPAINT_TELEA)

    before = evaluator.predict(Image.fromarray(rgb))
    after = evaluator.predict(Image.fromarray(cleaned))
    effect = {"before": _describe(before), "after": _describe(after)}
    side = np.concatenate([_draw_detections(rgb, before),
                           np.full((rgb.shape[0], max(8, rgb.shape[1] // 100), 3), 255, np.uint8),
                           _draw_detections(cleaned, after)], axis=1)
    steps = [
        ("Original photo", "Sunlight on the windscreen and bonnet creates bright white patches "
         "that the damage model can mistake for damage.", rgb),
        ("Glare mask", f"Pixels that are very bright but nearly colourless (HSV V ≥ {GLARE_V}, "
         f"S ≤ {GLARE_S}) are specular reflections: {share:.1%} of this photo (red).", overlay),
        ("CLAHE contrast", "Contrast-limited adaptive histogram equalisation on the lightness "
         "channel brings back detail in washed-out and shadowed areas without shifting colours.",
         equalised),
        ("Inpainting", "The masked highlights are filled in from the surrounding texture "
         "(Telea inpainting), removing the reflection before detection.", cleaned),
        ("Effect on the damage model", f"Before: {effect['before']}. After: {effect['after']}. "
         "A looser mask (V ≥ 200) removes this false detection but also erases real white "
         "scratches on other photos, so the threshold is a trade-off to tune on labelled data. "
         "Candidate pre-processing step, not in the API yet.", side),
    ]
    return ([{"src": _save(img, out_dir, f"ip_glare_{i}.jpg"), "title": title,
              "caption": caption} for i, (title, caption, img) in enumerate(steps, 1)], effect)


def build_steps(image: Image.Image, evaluator, out_dir: Path, plate_detector=None) -> dict:
    """``plate_detector``: e.g. plate_recognizer.YOLOPlateDetector, returns xyxy boxes."""
    glare, effect = glare_steps(image, evaluator, out_dir)
    return {"plate": plate_steps(image, out_dir, plate_detector), "glare": glare,
            "glare_effect": effect}
