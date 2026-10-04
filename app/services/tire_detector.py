"""Tire detectors.

Mirrors :mod:`app.services.evaluator`: :class:`YOLOTireDetector` wraps an Ultralytics model
trained on iRent photos - either the one-class tire model (``scripts/train_tire_detector.py``) or
the tyre + license-plate model (``scripts/train_tyre_plate_detector.py``), whose other classes
are ignored - and
:class:`MockTireDetector` returns deterministic boxes for tests and demos.
:func:`build_tire_detector` uses the mock only when explicitly requested; missing or broken
weights raise instead of silently serving fake tires.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from app.config import Settings
from app.schemas.common import BoundingBox
from app.schemas.tire import TireDetection, TireEllipse

logger = logging.getLogger(__name__)

_TIRE_CLASS_NAMES = {"tyre", "tire"}


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


def tire_class_ids(names: dict[int, str]) -> list[int] | None:
    """Class ids to keep: ``None`` (all) for a one-class model, else those named tyre/tire."""

    if len(names) == 1:
        return None
    ids = [i for i, name in names.items() if name.lower() in _TIRE_CLASS_NAMES]
    if not ids:
        raise ValueError(f"tire model has no tyre/tire class: {names}")
    return ids


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
        self._classes = tire_class_ids(self._model.names)

    def detect(self, image: Image.Image) -> list[RawTire]:
        results = self._model.predict(
            source=image,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            classes=self._classes,
            verbose=False,
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


def to_tire_detections(raws: list[RawTire], image_size: tuple[int, int]) -> list[TireDetection]:
    """Response tires: boxes clipped to the image (dropped if nothing is left), each with the
    ellipse inscribed in its clipped box."""

    width, height = image_size
    tires: list[TireDetection] = []
    for raw in raws:
        x1, y1, x2, y2 = raw.xyxy
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue  # clipped away entirely: nothing left to show
        tires.append(
            TireDetection(
                confidence=max(0.0, min(raw.confidence, 1.0)),
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                # From the clipped box, so the outline never leaves the image.
                ellipse=TireEllipse(
                    cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, rx=(x2 - x1) / 2, ry=(y2 - y1) / 2
                ),
            )
        )
    return tires


def build_tire_detector(settings: Settings) -> TireDetector:
    """Build the YOLO tire model, or the mock only if ``IRENT_TIRE_USE_MOCK=true``.

    Unset, missing or unloadable weights raise rather than silently serving fake tires.
    """

    if settings.tire_use_mock:
        logger.warning("IRENT_TIRE_USE_MOCK is set: tire results are FAKE")
        return MockTireDetector()
    hint = "Set IRENT_TIRE_WEIGHTS_PATH, or IRENT_TIRE_USE_MOCK=true to run with fake tire output."
    weights = settings.tire_weights_path
    if weights is None:
        raise RuntimeError(f"No tire weights configured. {hint}")
    if not weights.exists():
        raise RuntimeError(f"Tire weights not found at {weights}. {hint}")
    try:
        detector = YOLOTireDetector(settings)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the tire detector from {weights} ({type(exc).__name__}: {exc}). {hint}"
        ) from exc
    logger.info("Loaded tire detector from %s.", weights)
    return detector
