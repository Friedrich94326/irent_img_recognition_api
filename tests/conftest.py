"""Shared test fixtures."""

from __future__ import annotations

import io
import os

# Keep tests fast and hermetic: never load EasyOCR during app startup. This must run before
# ``app.main`` is imported, because that module builds the app (and caches Settings) at import.
os.environ.setdefault("IRENT_PLATE_USE_MOCK", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.api.deps import get_evaluator, get_settings  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.evaluator import MockEvaluator  # noqa: E402


@pytest.fixture
def evaluator()-> MockEvaluator:
    return MockEvaluator(version="test")


@pytest.fixture
def client(evaluator: MockEvaluator) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_evaluator] = lambda: evaluator
    # Isolate from any local .env (e.g. real YOLO weights configured for dev use)
    # so tests reflect the mocked detector regardless of host machine config.
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    # The lifespan still runs and sets app.state.evaluator; the override wins for requests.
    with TestClient(app) as test_client:
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
