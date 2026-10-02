from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_evaluator
from app.api.deps import get_settings as get_settings_dep
from app.config import Settings, get_settings
from app.schemas.common import DamageClass
from app.schemas.damage import DamageEvaluationResponse
from app.services.evaluator import MockEvaluator, RawDetection

ENDPOINT = "/api/v1/damage/evaluate"


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_evaluate_happy_path(client: TestClient, sample_jpeg: bytes) -> None:
    resp = client.post(
        ENDPOINT,
        files={"file": ("car.jpg", sample_jpeg, "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text

    body = DamageEvaluationResponse.model_validate(resp.json())
    assert body.is_mock is True
    assert body.image.width == 640
    assert body.image.height == 480
    assert body.summary.total_detections == len(body.detections)
    for det in body.detections:
        assert 0.0 <= det.confidence <= 1.0
        assert det.bounding_box.x2 <= body.image.width
        assert det.bounding_box.y2 <= body.image.height


def test_evaluate_is_deterministic(client: TestClient, sample_jpeg: bytes) -> None:
    first = client.post(ENDPOINT, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")}).json()
    second = client.post(ENDPOINT, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")}).json()
    assert first["detections"] == second["detections"]


def test_evaluate_rejects_unsupported_type(client: TestClient) -> None:
    resp = client.post(ENDPOINT, files={"file": ("note.txt", b"hello", "text/plain")})
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"


def test_evaluate_rejects_corrupt_image(client: TestClient) -> None:
    resp = client.post(ENDPOINT, files={"file": ("car.jpg", b"not-an-image", "image/jpeg")})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_image"


def test_evaluate_requires_file(client: TestClient) -> None:
    resp = client.post(ENDPOINT)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_evaluate_rejects_oversized_image(client: TestClient, sample_jpeg: bytes) -> None:
    client.app.dependency_overrides[get_settings_dep] = lambda: Settings(max_image_bytes=10)
    try:
        resp = client.post(ENDPOINT, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")})
    finally:
        client.app.dependency_overrides.pop(get_settings_dep, None)

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


class _FixedConfidences(MockEvaluator):
    """Mock detector that reports one box per given confidence."""

    def __init__(self, confidences: list[float]) -> None:
        super().__init__(version="test")
        self.confidences = confidences

    def predict(self, image, min_confidence=None):
        raws = [RawDetection(DamageClass.GLASS_SHATTER, c, (10.0, 10.0, 60.0, 60.0))
                for c in self.confidences]
        return [r for r in raws if min_confidence is None or r.confidence >= min_confidence]


@pytest.mark.parametrize(
    ("form", "kept", "used"),
    [({}, [0.9], 0.70),  # server default
     ({"confidence_threshold": "0.2"}, [0.9, 0.26], 0.2),
     ({"confidence_threshold": "0.95"}, [], 0.95)],
    ids=["default", "lower", "higher"],
)
def test_threshold_is_controllable_per_request(
    client: TestClient, sample_jpeg: bytes, form: dict, kept: list[float], used: float
) -> None:
    client.app.dependency_overrides[get_evaluator] = lambda: _FixedConfidences([0.9, 0.26])
    resp = client.post(ENDPOINT, data=form, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [d["confidence"] for d in body["detections"]] == kept
    assert body["confidence_threshold"] == used


@pytest.mark.parametrize("value", ["-0.1", "1.5", "high"])
def test_bad_threshold_is_422(client: TestClient, sample_jpeg: bytes, value: str) -> None:
    resp = client.post(
        ENDPOINT,
        data={"confidence_threshold": value},
        files={"file": ("car.jpg", sample_jpeg, "image/jpeg")},
    )
    assert resp.status_code == 422


def test_health_reports_default_threshold(client: TestClient) -> None:
    assert client.get("/api/v1/health").json()["confidence_threshold"] == 0.70
