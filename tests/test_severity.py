from __future__ import annotations

from app.schemas.common import DamageClass, Severity
from app.services.evaluator import RawDetection
from app.services.severity import (
    derive_overall_severity,
    detection_severity,
    enrich,
    summarize,
)


def _raw(cls: DamageClass, conf: float, box=(0.0, 0.0, 10.0, 10.0)) -> RawDetection:
    return RawDetection(damage_class=cls, confidence=conf, xyxy=box)


def test_detection_severity_low_confidence_is_downgraded() -> None:
    assert detection_severity(DamageClass.GLASS_SHATTER, 0.9) is Severity.SEVERE
    assert detection_severity(DamageClass.GLASS_SHATTER, 0.2) is Severity.MODERATE


def test_overall_severity_empty_is_none() -> None:
    assert derive_overall_severity([]) is Severity.NONE


def test_overall_severity_takes_worst_detection() -> None:
    dets = enrich([_raw(DamageClass.SCRATCH, 0.9), _raw(DamageClass.GLASS_SHATTER, 0.9)])
    assert derive_overall_severity(dets) is Severity.SEVERE


def test_overall_severity_bumps_when_widespread() -> None:
    dets = enrich([_raw(DamageClass.DENT, 0.9) for _ in range(3)])
    # three moderate detections -> bumped to severe
    assert derive_overall_severity(dets) is Severity.SEVERE


def test_summarize_counts_by_class() -> None:
    dets = enrich(
        [
            _raw(DamageClass.DENT, 0.8),
            _raw(DamageClass.DENT, 0.6),
            _raw(DamageClass.SCRATCH, 0.7),
        ]
    )
    summary = summarize(dets)
    assert summary.total_detections == 3
    assert summary.counts_by_class[DamageClass.DENT] == 2
    assert summary.counts_by_class[DamageClass.SCRATCH] == 1
    assert summary.max_confidence == 0.8


def test_parts_broken_is_low_severity() -> None:
    # A cracked bumper (Hotai 'parts_boken') is minor, unlike a missing part.
    assert detection_severity(DamageClass.PARTS_BROKEN, 0.9) == Severity.MINOR
    assert detection_severity(DamageClass.PARTS_BROKEN, 0.3) == Severity.NONE
    assert detection_severity(DamageClass.MISSING_PART, 0.9) == Severity.SEVERE
