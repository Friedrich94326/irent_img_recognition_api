"""Derive per-detection and overall severity from raw detections."""

from __future__ import annotations

from app.schemas.common import SEVERITY_ORDER, DamageClass, Severity
from app.schemas.damage import DamageSummary, Detection
from app.services.evaluator import RawDetection

CLASS_BASE_SEVERITY: dict[DamageClass, Severity] = {
    DamageClass.PAINT_CHIP: Severity.MINOR,
    DamageClass.SCRATCH: Severity.MINOR,
    DamageClass.PARTS_BROKEN: Severity.MINOR,
    DamageClass.DENT: Severity.MODERATE,
    DamageClass.CRACK: Severity.MODERATE,
    DamageClass.LAMP_BROKEN: Severity.MODERATE,
    DamageClass.MISSING_PART: Severity.SEVERE,
    DamageClass.GLASS_SHATTER: Severity.SEVERE,
    DamageClass.TIRE_FLAT: Severity.SEVERE,
}
"""Baseline severity assumed for a confident detection of each damage class."""

_LOW_CONFIDENCE = 0.4
"""Below this confidence, a detection's severity is knocked down one level."""

_ORDER_TO_SEVERITY = {rank: sev for sev, rank in SEVERITY_ORDER.items()}


def _shift(severity: Severity, delta: int) -> Severity:
    rank = SEVERITY_ORDER[severity] + delta
    rank = max(SEVERITY_ORDER[Severity.NONE], min(SEVERITY_ORDER[Severity.SEVERE], rank))
    return _ORDER_TO_SEVERITY[rank]


def detection_severity(damage_class: DamageClass, confidence: float) -> Severity:
    """Severity for one detection, tempered by low confidence."""

    base = CLASS_BASE_SEVERITY.get(damage_class, Severity.MODERATE)
    if confidence < _LOW_CONFIDENCE:
        return _shift(base, -1)
    return base


def enrich(raw: list[RawDetection]) -> list[Detection]:
    """Turn raw detections into response :class:`Detection` objects with severity."""

    return [
        Detection(
            damage_class=item.damage_class,
            confidence=round(item.confidence, 4),
            severity=detection_severity(item.damage_class, item.confidence),
            bounding_box={
                "x1": item.xyxy[0],
                "y1": item.xyxy[1],
                "x2": item.xyxy[2],
                "y2": item.xyxy[3],
            },
        )
        for item in raw
    ]


def summarize(detections: list[Detection]) -> DamageSummary:
    counts: dict[DamageClass, int] = {}
    for det in detections:
        counts[det.damage_class] = counts.get(det.damage_class, 0) + 1
    max_conf = max((d.confidence for d in detections), default=None)
    return DamageSummary(
        total_detections=len(detections),
        counts_by_class=counts,
        max_confidence=max_conf,
    )


def derive_overall_severity(detections: list[Detection]) -> Severity:
    """Overall severity: the worst single detection, bumped when damage is widespread."""

    if not detections:
        return Severity.NONE

    worst = max(detections, key=lambda d: SEVERITY_ORDER[d.severity]).severity
    moderate_plus = sum(
        1 for d in detections if SEVERITY_ORDER[d.severity] >= SEVERITY_ORDER[Severity.MODERATE]
    )
    if moderate_plus >= 3:
        return _shift(worst, 1)
    return worst
