"""Corner classifier: which corner of the car (左前 / 右前 / 左後 / 右後) a condition photo shows.

The corner a photo is filed under comes from its label (the Hotai file name or the upload's
``corner`` field); this model only checks it. A confident disagreement flags the photo - it is
still stored - so a wrong label is visible in the pre-rental check instead of silently putting
the wrong view in the grid. Trained by ``scripts/train_corner_classifier.py``.

Unlike the damage and tire models this one is optional: with no weights configured the check is
skipped.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from app.config import Settings
from app.schemas.vehicle import Corner
from app.services.image_io import orient_landscape

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CornerPrediction:
    corner: Corner
    confidence: float


class CornerClassifier(ABC):
    name: str = "unknown"

    @abstractmethod
    def predict(self, image: Image.Image) -> CornerPrediction:
        """Return the most likely corner of an upright RGB photo."""


class YOLOCornerClassifier(CornerClassifier):
    """Ultralytics classify model whose class names are the ``Corner`` values."""

    name = "yolo-corner"

    def __init__(self, settings: Settings) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        if settings.corner_weights_path is None:
            raise ValueError("corner_weights_path is not configured")
        self._model = YOLO(str(settings.corner_weights_path))
        self._device = settings.yolo_device
        self._corners = {i: Corner(n) for i, n in self._model.names.items()}

    def predict(self, image: Image.Image) -> CornerPrediction:
        # Trained on landscape copies, like the damage detector sees them.
        [result] = self._model.predict(
            source=orient_landscape(image), device=self._device, verbose=False
        )
        probs = result.probs
        return CornerPrediction(self._corners[int(probs.top1)], round(float(probs.top1conf), 4))


def is_mismatch(corner: Corner, prediction: CornerPrediction | None, min_confidence: float) -> bool:
    """True when the classifier confidently says the photo shows a different corner."""

    return (
        prediction is not None
        and prediction.corner != corner
        and prediction.confidence >= min_confidence
    )


def build_corner_classifier(settings: Settings) -> CornerClassifier | None:
    """Load the corner model, or ``None`` (no check) when no weights are configured or found.

    Weights that exist but can't be loaded raise.
    """

    weights = settings.corner_weights_path
    if weights is None or not weights.exists():
        logger.info("No corner classifier weights - photo corners are not checked.")
        return None
    try:
        classifier = YOLOCornerClassifier(settings)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the corner classifier from {weights} ({type(exc).__name__}: {exc})."
        ) from exc
    logger.info("Loaded corner classifier from %s.", weights)
    return classifier
