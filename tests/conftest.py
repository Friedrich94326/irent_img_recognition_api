"""Shared test fixtures."""

from __future__ import annotations

import io
import os

# Keep tests fast and hermetic: never load EasyOCR during app startup. This must run before
# ``app.main`` is imported, because that module builds the app (and caches Settings) at import.
os.environ.setdefault("IRENT_PLATE_USE_MOCK", "true")
os.environ.setdefault("IRENT_TIRE_USE_MOCK", "true")
# Keep the lifespan away from the real ops database (IRENT_DB_PATH in a local .env): a missing
# file disables both the vehicle link and the API call log.
os.environ.setdefault("IRENT_DB_PATH", "tests/__no_ops_db__.sqlite")
# Likewise never load a corner classifier configured for dev use; missing weights = no check.
os.environ.setdefault("IRENT_CORNER_WEIGHTS_PATH", "tests/__no_corner_model__.pt")
# Nor the XGBoost fee model: tests inject their own through get_fee_model.
os.environ.setdefault("IRENT_FEE_MODEL_PATH", "tests/__no_fee_model__.json")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.api.deps import (  # noqa: E402
    get_corner_classifier,
    get_evaluator,
    get_precheck_repository,
    get_settings,
    get_tire_detector,
    get_vehicle_photo_repository,
    get_vehicle_repository,
)
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.evaluator import MockEvaluator  # noqa: E402
from app.services.tire_detector import MockTireDetector  # noqa: E402


@pytest.fixture
def evaluator()-> MockEvaluator:
    return MockEvaluator(version="test")


@pytest.fixture
def client(evaluator: MockEvaluator) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_evaluator] = lambda: evaluator
    app.dependency_overrides[get_tire_detector] = MockTireDetector
    app.dependency_overrides[get_corner_classifier] = lambda: None
    # Isolate from any local .env (e.g. real YOLO weights configured for dev use)
    # so tests reflect the mocked detector regardless of host machine config.
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    # Never touch the real ops database (IRENT_DB_PATH in a local .env) from tests.
    app.dependency_overrides[get_vehicle_repository] = lambda: None
    app.dependency_overrides[get_precheck_repository] = lambda: None
    app.dependency_overrides[get_vehicle_photo_repository] = lambda: None
    # The lifespan still runs and sets app.state.evaluator; the override wins for requests.
    with TestClient(app) as test_client:
        # Same for the API call log the lifespan may have attached.
        app.state.api_call_log = None
        yield test_client
    app.dependency_overrides.clear()


def _encode(image: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def sample_jpeg() -> bytes:
    img = Image.new("RGB", (640, 480), color=(200, 120, 60))
    for x in range(0, 640, 8):
        for y in range(0, 480, 8):
            img.putpixel((x, y), ((x * y) % 255, (x + y) % 255, (x * 3) % 255))
    return _encode(img, "JPEG")


@pytest.fixture
def sample_png() -> bytes:
    return _encode(Image.new("RGB", (320, 240), color=(10, 90, 200)), "PNG")
