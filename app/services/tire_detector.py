"""Tire detectors.

Mirrors :mod:`app.services.evaluator`: :class:`YOLOTireDetector` wraps a one-class Ultralytics
model trained on auto-labelled iRent photos (see ``scripts/ontology_tire.yaml``), and
:class:`MockTireDetector` returns deterministic boxes so the endpoint works before any weights
exist. :func:`build_tire_detector` falls back to the mock if the real model cannot be loaded.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawTire:
    confidence: float
    xyxy: tuple[float, float, float, float]


class TireDetector(ABC):
    """Common interface for tire detectors."""

    name: str = "unknown"
    is_mock: bool = False

    @abstractmethod
    def detect(self, image: Image.Image) -> list[RawTire]:
        """Return every tire found in an RGB image, most confident first."""


class MockTireDetector(TireDetector):
    """Deterministic stand-in: 0-4 tire-sized boxes along the lower half, from the image bytes."""

    name = "mock-tire-detector"
    is_mock = True

    def detect(self, image: Image.Image) -> list[RawTire]:
        width, height = image.size
        seed = int.from_bytes(hashlib.sha256(image.tobytes()).digest()[:8], "big")
        count = seed % 5
        side = max(16.0, min(width, height) * 0.18)
        slot = width / max(count, 1)  # one tire centred in each equal-width slot
        tires: list[RawTire] = []
        for i in range(count):
            x1 = max(0.0, slot * i + (slot - side) / 2)
            y1 = height * 0.6
            conf = 0.6 + ((seed >> (8 * i)) % 35) / 100.0  # 0.60..0.94
            tires.append(
                RawTire(round(conf, 3), (x1, y1, min(width, x1 + side), min(height, y1 + side)))
            )
        return sorted(tires, key=lambda t: t.confidence, reverse=True)


class YOLOTireDetector(TireDetector):
    """Ultralytics wrapper; ``ultralytics`` is imported lazily so the mock never needs it."""

    name = "yolo-tire"
    is_mock = False

    def __init__(self, settings: Settings) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        if settings.tire_weights_path is None:
            raise ValueError("tire_weights_path is not configured")
        self._model = YOLO(str(settings.tire_weights_path))
        self._conf = settings.tire_confidence_threshold
        self._iou = settings.yolo_iou_threshold
        self._device = settings.tire_device or settings.yolo_device

    def detect(self, image: Image.Image) -> list[RawTire]:
        results = self._model.predict(
            source=image, conf=self._conf, iou=self._iou, device=self._device, verbose=False
        )
        tires: list[RawTire] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                tires.append(RawTire(float(box.conf[0]), (x1, y1, x2, y2)))
        return sorted(tires, key=lambda t: t.confidence, reverse=True)


def build_tire_detector(settings: Settings) -> TireDetector:
    """Use the YOLO tire model when its weights are configured and load; otherwise the mock."""

    weights = settings.tire_weights_path
    if weights is None:
        logger.info("No tire weights configured - using mock tire detector.")
        return MockTireDetector()
    if not weights.exists():
        logger.warning("Tire weights not found at %s - falling back to mock detector.", weights)
        return MockTireDetector()
    try:
        detector = YOLOTireDetector(settings)
        logger.info("Loaded tire detector from %s.", weights)
        return detector
    except Exception:  # noqa: BLE001 - any import/load failure should degrade gracefully
        logger.exception("Failed to load tire detector - falling back to mock detector.")
        return MockTireDetector()
