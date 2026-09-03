from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_settings as get_settings_dep
from app.config import Settings, get_settings
from app.schemas.damage import DamageEvaluationResponse

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
