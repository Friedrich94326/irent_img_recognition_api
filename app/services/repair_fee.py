"""Size damage against a tyre in the same photo and estimate the repair fee.

A tyre is a ruler of known size (~63 cm for a 195/65 R15 rental car tyre), so each damage box is
measured as a share of its nearest tyre's area and as approximate cm², and its severity is nudged
by that size. Body-damage boxes are merged into one damaged area (overlaps counted once), touching
damage that looks like a collision is flagged as an impact, and those features feed the XGBoost
fee model trained by ``scripts/repair_fee_model.py``.

Shared by ``POST /api/v1/fee/estimate`` and ``scripts/build_demo_page.py``. Boxes are
``(x1, y1, x2, y2)`` pixel tuples; tyres are given as their boxes.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.schemas.common import Severity
from app.schemas.damage import Detection
from app.services.evaluator import RawDetection
from app.services.severity import _shift, derive_overall_severity, detection_severity

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]

TYRE_DIAMETER_CM = 63.0
SMALL_SHARE, LARGE_SHARE = 0.05, 0.25
"""Damage smaller than 5 % of the tyre's area is 'small', larger than 25 % is 'large'."""
_BAND_SHIFT = {"small": -1, "medium": 0, "large": 1}

BODY_EXCLUDED = {"tire_flat"}
"""Its box is the wheel itself, not damaged bodywork, so it stays out of the area union."""
DEFORMATION = {"dent", "crack"}
BROKEN_PARTS = {"crack", "lamp_broken", "glass_shatter", "missing_part", "parts_broken"}
IMPACT_AREA_SHARE = 1.0
"""A single cluster of damage bigger than a whole wheel reads as a collision."""
_IMPACT_PAD_FRAC = 0.02  # of the shorter image side: boxes this close count as touching

# Fee-model feature layout; must match the columns weights/repair_fee_xgb.json was trained on.
CLASSES = ("dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat", "missing_part")
FEATURES = (
    [f"n_{c}" for c in CLASSES]
    + [f"max_share_{c}" for c in CLASSES]
    + ["union_share", "n_detections", "impact", "n_impacts"]
)


@dataclass
class Measured:
    damage_class: str
    confidence: float
    xyxy: Box
    base_severity: str
    severity: str
    tyre_share: float | None
    area_cm2: float | None
    size_band: str | None
    tyre_index: int | None


@dataclass
class DamagedArea:
    union_px: float
    summed_px: float
    union_share: float | None
    summed_share: float | None
    union_cm2: float | None


@dataclass
class Impact:
    box: Box
    classes: list[str]
    reason: str


def boxes_overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def size_band(share: float) -> str:
    if share < SMALL_SHARE:
        return "small"
    return "large" if share > LARGE_SHARE else "medium"


def nearest_tyre(box: Box, tyres: list[Box]) -> int:
    """Index of the tyre whose centre is closest to the box centre.

    The nearest wheel is the best guess for a ruler at the same distance from the camera; the
    most confident one is often the far wheel, shrunk by perspective.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(
        range(len(tyres)),
        key=lambda i: math.dist((cx, cy), ((tyres[i][0] + tyres[i][2]) / 2,
                                           (tyres[i][1] + tyres[i][3]) / 2)),
    )


def tyre_scale(tyre: Box, tyre_cm: float) -> tuple[float, float]:
    """(tyre area in px², pixels per cm) for a detected tyre."""
    tx1, ty1, tx2, ty2 = tyre
    rx, ry = (tx2 - tx1) / 2, (ty2 - ty1) / 2
    # Seen at an angle a wheel shrinks along one axis only: the longer axis is its diameter.
    return math.pi * rx * ry, 2 * max(rx, ry) / tyre_cm


def measure(raw_damage: list[RawDetection], tyres: list[Box], tyre_cm: float) -> list[Measured]:
    """Size each detection against its nearest tyre; unscaled (``None``) without a tyre."""
    out = []
    for det in raw_damage:
        x1, y1, x2, y2 = det.xyxy
        box_area = (x2 - x1) * (y2 - y1)
        base = detection_severity(det.damage_class, det.confidence)
        share = cm2 = band = ref = None
        severity = base
        if tyres:
            ref = nearest_tyre(det.xyxy, tyres)
            tyre_area, px_per_cm = tyre_scale(tyres[ref], tyre_cm)
            share = box_area / tyre_area
            cm2 = box_area / px_per_cm**2
            band = size_band(share)
            severity = _shift(base, _BAND_SHIFT[band])
        out.append(
            Measured(det.damage_class.value, det.confidence, det.xyxy, base.value,
                     severity.value, share, cm2, band, ref)
        )
    return out


def union_area(boxes: list[Box]) -> float:
    """Exact area covered by axis-aligned boxes, overlaps counted once (coordinate compression)."""
    xs = sorted({v for b in boxes for v in (b[0], b[2])})
    ys = sorted({v for b in boxes for v in (b[1], b[3])})
    area = 0.0
    for x0, x1 in itertools.pairwise(xs):
        for y0, y1 in itertools.pairwise(ys):
            if any(b[0] <= x0 and x1 <= b[2] and b[1] <= y0 and y1 <= b[3] for b in boxes):
                area += (x1 - x0) * (y1 - y0)
    return area


def damaged_area(measured: list[Measured], tyres: list[Box], tyre_cm: float) -> DamagedArea:
    """Union of body-damage boxes per reference tyre, next to the plain sum of the boxes."""
    body = [m for m in measured if m.damage_class not in BODY_EXCLUDED]
    groups: dict[int | None, list[Measured]] = {}
    for m in body:
        groups.setdefault(m.tyre_index, []).append(m)
    union_px = summed_px = 0.0
    union_share = summed_share = union_cm2 = 0.0
    for ref, items in groups.items():
        u = union_area([m.xyxy for m in items])
        s = sum((m.xyxy[2] - m.xyxy[0]) * (m.xyxy[3] - m.xyxy[1]) for m in items)
        union_px, summed_px = union_px + u, summed_px + s
        if ref is not None:
            tyre_area, px_per_cm = tyre_scale(tyres[ref], tyre_cm)
            union_share += u / tyre_area
            summed_share += s / tyre_area
            union_cm2 += u / px_per_cm**2
    scaled = bool(body) and all(ref is not None for ref in groups)
    return DamagedArea(
        union_px=union_px,
        summed_px=summed_px,
        union_share=union_share if scaled else None,
        summed_share=summed_share if scaled else None,
        union_cm2=union_cm2 if scaled else None,
    )


def impact_clusters(
    measured: list[Measured], tyres: list[Box], tyre_cm: float, image_size: tuple[int, int]
) -> list[Impact]:
    """Groups of touching body-damage boxes that look like a collision rather than wear.

    The damage models have no 'crash' or 'broken bumper' class, so an impact is inferred from how
    detections combine: deformation next to a broken part, a pile-up of several damage types in
    one spot, or damage covering more than a wheel's worth of area.
    """
    body = [m for m in measured if m.damage_class not in BODY_EXCLUDED]
    pad = _IMPACT_PAD_FRAC * min(image_size)
    grown = [(m.xyxy[0] - pad, m.xyxy[1] - pad, m.xyxy[2] + pad, m.xyxy[3] + pad) for m in body]
    parent = list(range(len(body)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(body)):
        for j in range(i + 1, len(body)):
            if boxes_overlap(grown[i], grown[j]):
                parent[find(i)] = find(j)
    clusters: dict[int, list[Measured]] = {}
    for i, m in enumerate(body):
        clusters.setdefault(find(i), []).append(m)

    impacts = []
    for items in clusters.values():
        kinds = {m.damage_class for m in items}
        reasons = []
        deformed = [m for m in items if m.damage_class in DEFORMATION]
        broken = [m for m in items if m.damage_class in BROKEN_PARTS]
        if any(d is not b and boxes_overlap(d.xyxy, b.xyxy) for d in deformed for b in broken):
            reasons.append("deformation next to a broken part")
        if len(items) >= 3 and len(kinds) >= 2:
            reasons.append(f"{len(items)} damages of {len(kinds)} kinds in one spot")
        ref = items[0].tyre_index
        if ref is not None and all(m.tyre_index == ref for m in items):
            tyre_area, _ = tyre_scale(tyres[ref], tyre_cm)
            share = union_area([m.xyxy for m in items]) / tyre_area
            if share >= IMPACT_AREA_SHARE:
                reasons.append(f"damaged area {share:.0%} of a tyre")
        if reasons:
            boxes = [m.xyxy for m in items]
            impacts.append(Impact(
                box=(min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes)),
                classes=sorted(kinds),
                reason="; ".join(reasons),
            ))
    return impacts


def overall(measured: list[Measured], impacts: list[Impact]) -> tuple[Severity, Severity]:
    """(overall severity, base severity before any impact escalation)."""
    detections = [
        Detection(
            damage_class=m.damage_class,
            confidence=round(m.confidence, 4),
            severity=Severity(m.severity),
            bounding_box=dict(zip(("x1", "y1", "x2", "y2"), m.xyxy, strict=True)),
        )
        for m in measured
    ]
    base = derive_overall_severity(detections)
    return (Severity.SEVERE if impacts else base), base


def features(measured: list[Measured], area: DamagedArea, impacts: list[Impact]) -> list[float]:
    """Fee-model feature row (:data:`FEATURES` order) for one photo.

    Shares are NaN when no tyre gave a scale; XGBoost routes missing values natively.
    """
    counts = dict.fromkeys(CLASSES, 0)
    shares: dict[str, float] = dict.fromkeys(CLASSES, math.nan)
    for m in measured:
        if m.damage_class not in counts:
            continue
        counts[m.damage_class] += 1
        if m.tyre_share is not None:
            prev = shares[m.damage_class]
            shares[m.damage_class] = m.tyre_share if math.isnan(prev) else max(prev, m.tyre_share)
    union = area.union_share
    return (
        [counts[c] for c in CLASSES]
        + [shares[c] for c in CLASSES]
        + [math.nan if union is None else union, len(measured), int(bool(impacts)), len(impacts)]
    )


class FeeModel:
    """The XGBoost regressor from ``scripts/repair_fee_model.py`` (trained on log1p(NT$))."""

    def __init__(self, path: Path) -> None:
        import xgboost as xgb  # noqa: PLC0415 - optional dependency (requirements-fee.txt)

        self.name = f"xgboost:{path.name}"
        self._model = xgb.XGBRegressor()
        self._model.load_model(path)

    def predict(self, row: list[float]) -> float:
        import numpy as np  # noqa: PLC0415

        return float(np.expm1(self._model.predict(np.array([row], dtype=float))[0]))


def build_fee_model(settings: Settings) -> FeeModel | None:
    """Load the fee model, or ``None`` (fees are not estimated) when it cannot be."""

    path = settings.fee_model_path
    if path is None:
        logger.info("No repair-fee model configured - fees are not estimated.")
        return None
    if not path.exists():
        logger.warning("Repair-fee model not found at %s - fees are not estimated.", path)
        return None
    try:
        model = FeeModel(path)
    except Exception:  # noqa: BLE001 - a missing xgboost must not take the API down
        logger.exception("Failed to load the repair-fee model - fees are not estimated.")
        return None
    logger.info("Loaded repair-fee model from %s.", path)
    return model
