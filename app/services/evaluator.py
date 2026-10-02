"""Damage detectors.

Two implementations share one interface:

* :class:`YOLOv8Evaluator` wraps an Ultralytics YOLOv8 model loaded from a weights file.
* :class:`MockEvaluator` produces deterministic pseudo-detections so the API is usable
  before any model has been trained.

:func:`build_evaluator` picks one based on :class:`app.config.Settings` and falls back to
the mock detector if the real model cannot be loaded.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from app.config import Settings
from app.schemas.common import DamageClass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawDetection:
    """Detector output before schema/severity enrichment."""

    damage_class: DamageClass
    confidence: float
    xyxy: tuple[float, float, float, float]


class DamageEvaluator(ABC):
    """Common interface for damage detectors."""

    name: str = "unknown"
    version: str = "0"
    is_mock: bool = False

    @abstractmethod
    def predict(
        self, image: Image.Image, min_confidence: float | None = None
    ) -> list[RawDetection]:
        """Return detections for a single RGB image.

        ``min_confidence`` overrides the configured threshold for this call (a client-chosen
        threshold); ``None`` keeps the detector's default.
        """


class MockEvaluator(DamageEvaluator):
    """Deterministic stand-in detector.

    The output depends only on the image bytes, so repeated calls on the same image
    return the same detections - useful for tests and demos.
    """

    name = "mock-detector"
    is_mock = True

    _CLASSES = [
        DamageClass.SCRATCH,
        DamageClass.DENT,
        DamageClass.PAINT_CHIP,
        DamageClass.LAMP_BROKEN,
        DamageClass.GLASS_SHATTER,
    ]

    def __init__(self, version: str = "0.1.0") -> None:
        self.version = version

    def predict(
        self, image: Image.Image, min_confidence: float | None = None
    ) -> list[RawDetection]:
        width, height = image.size
        digest = hashlib.sha256(image.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")

        n_detections = seed % 4  # 0..3
        detections: list[RawDetection] = []
        cursor = seed
        for _ in range(n_detections):
            cursor = (cursor * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            cls = self._CLASSES[cursor % len(self._CLASSES)]
            conf = 0.55 + ((cursor >> 8) % 40) / 100.0  # 0.55..0.94
            bw = max(16.0, width * (0.12 + ((cursor >> 16) % 20) / 100.0))
            bh = max(16.0, height * (0.12 + ((cursor >> 24) % 20) / 100.0))
            x1 = ((cursor >> 32) % max(1, int(width - bw)))
            y1 = ((cursor >> 40) % max(1, int(height - bh)))
            detections.append(
                RawDetection(
                    damage_class=cls,
                    confidence=round(conf, 3),
                    xyxy=(float(x1), float(y1), float(x1 + bw), float(y1 + bh)),
                )
            )
        if min_confidence is not None:
            detections = [d for d in detections if d.confidence >= min_confidence]
        return detections


class YOLOv8Evaluator(DamageEvaluator):
    """Ultralytics YOLOv8 wrapper.

    ``ultralytics`` (and its heavy ``torch`` dependency) is imported lazily inside
    ``__init__`` so that the mock path never needs it installed.
    """

    name = "yolov8"
    is_mock = False

    def __init__(self, settings: Settings) -> None:
        from ultralytics import YOLO  # noqa: PLC0415
        from ultralytics import __version__ as ultralytics_version  # noqa: PLC0415

        if settings.yolo_weights_path is None:
            raise ValueError("yolo_weights_path is not configured")

        self.version = ultralytics_version
        self._device = settings.yolo_device
        self._conf = settings.yolo_confidence_threshold
        self._iou = settings.yolo_iou_threshold
        self._model = YOLO(str(settings.yolo_weights_path))

        # Map the model's own label ids -> our DamageClass enum. Labels that do not
        # correspond to a known damage class are dropped (with a one-time warning).
        self._label_map: dict[int, DamageClass] = {}
        unknown: list[str] = []
        for idx, label in self._model.names.items():
            key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
            try:
                self._label_map[int(idx)] = DamageClass(key)
            except ValueError:
                unknown.append(str(label))
        if unknown:
            logger.warning(
                "YOLOv8 model exposes labels with no matching DamageClass (dropped): %s",
                ", ".join(sorted(set(unknown))),
            )

    def predict(
        self, image: Image.Image, min_confidence: float | None = None
    ) -> list[RawDetection]:
        results = self._model.predict(
            source=image,
            conf=self._conf if min_confidence is None else min_confidence,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )
        detections: list[RawDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                damage_class = self._label_map.get(cls_id)
                if damage_class is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    RawDetection(
                        damage_class=damage_class,
                        confidence=float(box.conf[0]),
                        xyxy=(x1, y1, x2, y2),
                    )
                )
        return detections


def build_evaluator(settings: Settings) -> DamageEvaluator:
    """Return the best available detector for the current settings.

    Uses :class:`YOLOv8Evaluator` when a weights file is configured and present;
    otherwise (or on any load failure) returns :class:`MockEvaluator`.
    """

    weights = settings.yolo_weights_path
    if weights is None:
        logger.info("No YOLOv8 weights configured - using mock detector.")
        return MockEvaluator(version=settings.version)

    if not weights.exists():
        logger.warning("YOLOv8 weights not found at %s - falling back to mock detector.", weights)
        return MockEvaluator(version=settings.version)

    try:
        evaluator = YOLOv8Evaluator(settings)
        logger.info("Loaded YOLOv8 detector from %s (ultralytics %s).", weights, evaluator.version)
        return evaluator
    except Exception:  # noqa: BLE001 - any import/load failure should degrade gracefully
        logger.exception("Failed to load YOLOv8 detector - falling back to mock detector.")
        return MockEvaluator(version=settings.version)
