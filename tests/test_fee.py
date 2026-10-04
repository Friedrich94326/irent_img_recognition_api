from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_evaluator, get_fee_model, get_tire_detector
from app.schemas.common import DamageClass
from app.schemas.fee import FeeEstimateResponse
from app.services.evaluator import DamageEvaluator, RawDetection
from app.services.repair_fee import (
    FEATURES,
    damaged_area,
    features,
    impact_clusters,
    measure,
    union_area,
)
from app.services.tire_detector import RawTire, TireDetector

ENDPOINT = "/api/v1/fee/estimate"
# A 100 px tyre: area pi * 50² ≈ 7854 px², 100 px / 63 cm.
TYRE = (500.0, 350.0, 600.0, 450.0)
TYRE_AREA = math.pi * 50 * 50


class FixedEvaluator(DamageEvaluator):
    name = "fixed"
    version = "test"

    def __init__(self, detections: list[RawDetection]) -> None:
        self._detections = detections

    def predict(self, image, min_confidence=None):
        return self._detections


class FixedTireDetector(TireDetector):
    name = "fixed-tires"

    def __init__(self, tires: list[RawTire]) -> None:
        self._tires = tires

    def detect(self, image):
        return self._tires


class FixedFeeModel:
    name = "fixed-fee"

    def __init__(self, fee: float) -> None:
        self.fee = fee
        self.rows: list[list[float]] = []

    def predict(self, row: list[float]) -> float:
        self.rows.append(row)
        return self.fee


def setup(
    client: TestClient,
    detections: list[RawDetection],
    tires: list[RawTire],
    fee_model: FixedFeeModel | None,
) -> None:
    overrides = client.app.dependency_overrides
    overrides[get_evaluator] = lambda: FixedEvaluator(detections)
    overrides[get_tire_detector] = lambda: FixedTireDetector(tires)
    overrides[get_fee_model] = lambda: fee_model


def post(client: TestClient, jpeg: bytes) -> dict:
    resp = client.post(ENDPOINT, files={"file": ("car.jpg", jpeg, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    FeeEstimateResponse.model_validate(resp.json())
    return resp.json()


def test_union_area_counts_overlap_once() -> None:
    assert union_area([(0, 0, 10, 10), (5, 5, 15, 15)]) == 175
    assert union_area([(0, 0, 10, 10), (20, 0, 30, 10)]) == 200
    assert union_area([(0, 0, 10, 10), (2, 2, 4, 4)]) == 100


def test_measure_sizes_against_nearest_tyre_and_shifts_severity() -> None:
    far_tyre = (0.0, 0.0, 300.0, 300.0)  # bigger, but far from the damage
    raw = [
        RawDetection(DamageClass.SCRATCH, 0.9, (450.0, 300.0, 460.0, 310.0)),  # 100 px²: small
        RawDetection(DamageClass.DENT, 0.9, (400.0, 250.0, 500.0, 350.0)),  # 10000 px²: large
    ]
    small, large = measure(raw, [far_tyre, TYRE], 63.0)
    assert small.tyre_index == large.tyre_index == 1
    assert small.tyre_share == pytest.approx(100 / TYRE_AREA)
    assert small.area_cm2 == pytest.approx(100 / (100 / 63) ** 2)
    assert (small.size_band, small.base_severity, small.severity) == ("small", "minor", "none")
    assert (large.size_band, large.base_severity, large.severity) == ("large", "moderate", "severe")


def test_measure_without_tyre_is_unscaled() -> None:
    (m,) = measure([RawDetection(DamageClass.DENT, 0.9, (0, 0, 10, 10))], [], 63.0)
    assert (m.tyre_share, m.area_cm2, m.size_band, m.tyre_index) == (None, None, None, None)
    assert m.severity == m.base_severity == "moderate"


def test_damaged_area_excludes_flat_tyre_and_merges_overlaps() -> None:
    raw = [
        RawDetection(DamageClass.SCRATCH, 0.9, (400.0, 300.0, 440.0, 340.0)),
        RawDetection(DamageClass.DENT, 0.9, (420.0, 320.0, 460.0, 360.0)),
        RawDetection(DamageClass.TIRE_FLAT, 0.9, TYRE),
    ]
    measured = measure(raw, [TYRE], 63.0)
    area = damaged_area(measured, [TYRE], 63.0)
    assert area.summed_px == 3200
    assert area.union_px == 2800
    assert area.union_share == pytest.approx(2800 / TYRE_AREA)


def test_impact_when_dent_touches_broken_lamp() -> None:
    raw = [
        RawDetection(DamageClass.DENT, 0.9, (100.0, 100.0, 200.0, 200.0)),
        RawDetection(DamageClass.LAMP_BROKEN, 0.9, (180.0, 120.0, 260.0, 180.0)),
        RawDetection(DamageClass.SCRATCH, 0.9, (500.0, 20.0, 520.0, 40.0)),  # elsewhere
    ]
    measured = measure(raw, [TYRE], 63.0)
    (impact,) = impact_clusters(measured, [TYRE], 63.0, (640, 480))
    assert impact.classes == ["dent", "lamp_broken"]
    assert impact.box == (100.0, 100.0, 260.0, 200.0)
    assert "deformation next to a broken part" in impact.reason


def test_features_follow_model_layout() -> None:
    raw = [
        RawDetection(DamageClass.SCRATCH, 0.9, (400.0, 300.0, 440.0, 340.0)),
        RawDetection(DamageClass.SCRATCH, 0.8, (100.0, 100.0, 110.0, 110.0)),
        RawDetection(DamageClass.PAINT_CHIP, 0.8, (0.0, 0.0, 5.0, 5.0)),  # not a model class
    ]
    measured = measure(raw, [TYRE], 63.0)
    row = dict(zip(FEATURES, features(measured, damaged_area(measured, [TYRE], 63.0), []),
                   strict=True))
    assert row["n_scratch"] == 2 and row["n_dent"] == 0
    assert row["max_share_scratch"] == pytest.approx(1600 / TYRE_AREA)
    assert math.isnan(row["max_share_dent"])
    assert row["n_detections"] == 3 and row["impact"] == 0


def test_estimate_returns_scaled_damage_and_fee(client: TestClient, sample_jpeg: bytes) -> None:
    fee_model = FixedFeeModel(5321.0)
    setup(
        client,
        [RawDetection(DamageClass.DENT, 0.9, (400.0, 300.0, 440.0, 340.0))],
        [RawTire(0.4, (0.0, 0.0, 60.0, 60.0)), RawTire(0.9, TYRE)],
        fee_model,
    )
    body = post(client, sample_jpeg)

    (det,) = body["detections"]
    assert det["tire_index"] == 1  # the nearer tyre, not the more confident one
    assert det["tire_share"] == pytest.approx(1600 / TYRE_AREA)
    assert det["size_band"] == "medium"
    assert body["damaged_area"]["union_px"] == 1600
    assert body["estimated_fee"] == 5300  # rounded to NT$100
    assert body["currency"] == "TWD"
    assert body["fee_model_name"] == "fixed-fee"
    assert len(fee_model.rows) == 1 and len(fee_model.rows[0]) == len(FEATURES)
    assert body["vehicle"] is None


def test_estimate_without_damage_costs_nothing(client: TestClient, sample_jpeg: bytes) -> None:
    fee_model = FixedFeeModel(9999.0)
    setup(client, [], [RawTire(0.9, TYRE)], fee_model)
    body = post(client, sample_jpeg)
    assert body["estimated_fee"] == 0
    assert body["overall_severity"] == "none"
    assert fee_model.rows == []  # the model is not asked about an undamaged photo


def test_estimate_without_fee_model_still_measures(
    client: TestClient, sample_jpeg: bytes
) -> None:
    setup(client, [RawDetection(DamageClass.SCRATCH, 0.9, (10, 10, 50, 50))], [], None)
    body = post(client, sample_jpeg)
    assert body["estimated_fee"] is None and body["fee_model_name"] is None
    assert body["detections"][0]["tire_share"] is None  # no tyre: unscaled
    assert body["damaged_area"]["union_share"] is None


def test_estimate_impact_escalates_to_severe(client: TestClient, sample_jpeg: bytes) -> None:
    setup(
        client,
        [
            RawDetection(DamageClass.DENT, 0.9, (100.0, 100.0, 200.0, 200.0)),
            RawDetection(DamageClass.LAMP_BROKEN, 0.9, (180.0, 120.0, 260.0, 180.0)),
        ],
        [RawTire(0.9, TYRE)],
        FixedFeeModel(12000.0),
    )
    body = post(client, sample_jpeg)
    assert len(body["impacts"]) == 1
    assert body["overall_severity"] == "severe"
    assert body["impacts"][0]["bounding_box"] == {"x1": 100, "y1": 100, "x2": 260, "y2": 200}
