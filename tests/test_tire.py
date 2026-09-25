from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import get_tire_detector
from app.config import Settings
from app.schemas.tire import TireDetectionResponse
from app.services.tire_detector import (
    MockTireDetector,
    RawTire,
    TireDetector,
    build_tire_detector,
    tire_class_ids,
)

ENDPOINT = "/api/v1/tire/detect"


class FakeTireDetector(TireDetector):
    name = "fake"

    def __init__(self, tires: list[RawTire]) -> None:
        self._tires = tires

    def detect(self, image):
        return self._tires


def test_detect_mock(client: TestClient, sample_jpeg: bytes) -> None:
    resp = client.post(ENDPOINT, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = TireDetectionResponse.model_validate(resp.json())
    assert body.is_mock and body.model_name == "mock-tire-detector"
    assert body.tire_count == len(body.tires)
    for tire in body.tires:
        b, e = tire.bounding_box, tire.ellipse
        assert (e.cx, e.cy) == ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)
        assert (e.rx, e.ry) == (b.width / 2, b.height / 2)


def test_detect_returns_boxes_clipped_to_image(client: TestClient, sample_jpeg: bytes) -> None:
    fake = FakeTireDetector(
        [
            RawTire(0.9, (100, 300, 220, 420)),
            RawTire(0.7, (600, 400, 700, 520)),  # runs off the 640x480 image: clipped
            RawTire(0.5, (700, 10, 800, 60)),  # entirely outside: dropped
        ]
    )
    client.app.dependency_overrides[get_tire_detector] = lambda: fake
    body = client.post(ENDPOINT, files={"file": ("c.jpg", sample_jpeg, "image/jpeg")}).json()
    assert body["tire_count"] == 2
    assert body["tires"][0]["bounding_box"] == {"x1": 100, "y1": 300, "x2": 220, "y2": 420}
    assert body["tires"][1]["bounding_box"]["x2"] == 640
    assert body["tires"][1]["bounding_box"]["y2"] == 480
    # Each tire's outline is the ellipse inscribed in its (clipped) box.
    assert body["tires"][0]["ellipse"] == {"cx": 160, "cy": 360, "rx": 60, "ry": 60}
    clipped = body["tires"][1]["ellipse"]
    assert clipped["cx"] + clipped["rx"] <= 640 and clipped["cy"] + clipped["ry"] <= 480


def test_detect_no_tires(client: TestClient, sample_jpeg: bytes) -> None:
    client.app.dependency_overrides[get_tire_detector] = lambda: FakeTireDetector([])
    body = client.post(ENDPOINT, files={"file": ("c.jpg", sample_jpeg, "image/jpeg")}).json()
    assert body["tires"] == [] and body["tire_count"] == 0


def test_detect_rejects_bad_type(client: TestClient) -> None:
    resp = client.post(ENDPOINT, files={"file": ("a.txt", b"hi", "text/plain")})
    assert resp.status_code == 415


def test_mock_is_deterministic_and_inside_image() -> None:
    img = Image.new("RGB", (640, 480), (120, 130, 140))
    first, second = MockTireDetector().detect(img), MockTireDetector().detect(img)
    assert first == second
    for tire in first:
        x1, y1, x2, y2 = tire.xyxy
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480


def test_build_tire_detector_raises_instead_of_mocking(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="IRENT_TIRE_USE_MOCK"):
        build_tire_detector(Settings(_env_file=None, tire_use_mock=False))
    missing = Settings(
        _env_file=None, tire_use_mock=False, tire_weights_path=tmp_path / "missing.pt"
    )
    with pytest.raises(RuntimeError, match="not found"):
        build_tire_detector(missing)


def test_build_tire_detector_mock_only_when_requested() -> None:
    assert build_tire_detector(Settings(_env_file=None, tire_use_mock=True)).is_mock


def test_tire_class_ids_keeps_only_tyres_of_a_multi_class_model() -> None:
    assert tire_class_ids({0: "tire"}) is None  # one-class model: keep everything
    assert tire_class_ids({0: "tyre", 1: "license_plate"}) == [0]
    with pytest.raises(ValueError, match="no tyre/tire class"):
        tire_class_ids({0: "dent", 1: "scratch"})


def test_health_reports_tire_detector(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["tire_model_name"] == "mock-tire-detector"
    assert body["tire_is_mock"] is True
