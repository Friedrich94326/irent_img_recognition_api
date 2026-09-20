"""License-plate recognisers.

Mirrors :mod:`app.services.evaluator`: a real OCR-backed implementation
(:class:`EasyOCRPlateRecognizer`) and a deterministic :class:`MockPlateRecognizer`, chosen by
:func:`build_plate_recognizer` with a fallback to the mock if EasyOCR is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.config import Settings
from app.services.plate_locator import locate_plate_regions

logger = logging.getLogger(__name__)

# Taiwan layouts once separators are removed: ABC1234 / AB1234 (cars), 1234AB / 123ABC (older),
# ABC123 (motorcycles). The hyphen is re-inserted by ``format_plate``.
_PLATE_RE = re.compile(
    r"(?P<letters>[A-Z]{2,3})(?P<digits>\d{3,4})|(?P<digits2>\d{3,4})(?P<letters2>[A-Z]{2,3})"
)
# Common OCR confusions, applied only inside the segment that should be letters / digits.
_TO_DIGIT = str.maketrans("OQDIZSB", "0001258")
_TO_LETTER = str.maketrans("0158", "OISB")


@dataclass(frozen=True)
class RawPlate:
    text: str
    confidence: float
    xyxy: tuple[float, float, float, float] | None


def _clean(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def format_plate(text: str) -> str | None:
    """Return ``text`` as a hyphenated Taiwan plate (e.g. ``ABC-1234``) or ``None``."""

    cleaned = _clean(text)
    m = _PLATE_RE.fullmatch(cleaned)
    if m:
        if m.group("letters"):
            return f"{m.group('letters')}-{m.group('digits')}"
        return f"{m.group('digits2')}-{m.group('letters2')}"
    # Retry as letters-then-digits with confused characters (e.g. 'A8C-I234').
    for n_letters in (3, 2):
        for n_digits in (4, 3):
            if len(cleaned) == n_letters + n_digits:
                letters = cleaned[:n_letters].translate(_TO_LETTER)
                digits = cleaned[n_letters:].translate(_TO_DIGIT)
                if letters.isalpha() and digits.isdigit():
                    return f"{letters}-{digits}"
    return None


class PlateRecognizer(ABC):
    name: str = "unknown"
    is_mock: bool = False

    @abstractmethod
    def read(self, image: Image.Image) -> list[RawPlate]:
        """Return text regions found in an RGB image, in reading order."""


class MockPlateRecognizer(PlateRecognizer):
    """Deterministic stand-in: the same image bytes always yield the same plate."""

    name = "mock-plate-recognizer"
    is_mock = True

    def read(self, image: Image.Image) -> list[RawPlate]:
        digest = hashlib.sha256(image.tobytes()).digest()
        letters = "".join(chr(ord("A") + digest[i] % 26) for i in range(3))
        digits = f"{int.from_bytes(digest[3:5], 'big') % 10000:04d}"
        w, h = image.size
        box = (w * 0.3, h * 0.55, w * 0.7, h * 0.7)
        return [RawPlate(f"{letters}-{digits}", 0.5, box)]


class EasyOCRPlateRecognizer(PlateRecognizer):
    """OpenCV finds plate-shaped regions; EasyOCR reads only those crops.

    If no region yields a valid plate, the whole image is read as a fallback.
    """

    name = "opencv+easyocr"

    def __init__(self, gpu: bool = False, use_locator: bool = True) -> None:
        import easyocr  # noqa: PLC0415 - heavy; deferred so the mock works without it

        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self._use_locator = use_locator

    def _ocr(self, image: Image.Image, origin: tuple[int, int] = (0, 0)) -> list[RawPlate]:
        results = self._reader.readtext(
            np.asarray(image),
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            detail=1,
        )
        ox, oy = origin
        plates: list[RawPlate] = []
        for points, text, conf in results:
            xs = [p[0] + ox for p in points]
            ys = [p[1] + oy for p in points]
            plates.append(RawPlate(text, float(conf), (min(xs), min(ys), max(xs), max(ys))))
        return plates

    def read(self, image: Image.Image) -> list[RawPlate]:
        if self._use_locator:
            found: list[RawPlate] = []
            for x1, y1, x2, y2 in locate_plate_regions(image):
                found.extend(self._ocr(image.crop((x1, y1, x2, y2)), origin=(x1, y1)))
            if any(format_plate(p.text) for p in found):
                return found
            # No crop produced a valid plate; keep any crop text but also scan the full frame.
            return found + self._ocr(image)
        return self._ocr(image)


def build_plate_recognizer(settings: Settings) -> PlateRecognizer:
    if settings.plate_use_mock:
        return MockPlateRecognizer()
    try:
        return EasyOCRPlateRecognizer(
            gpu=settings.yolo_device != "cpu", use_locator=settings.plate_use_opencv_locator
        )
    except Exception:  # noqa: BLE001 - any load failure should degrade to the mock
        logger.warning("EasyOCR unavailable; falling back to mock plate recognizer", exc_info=True)
        return MockPlateRecognizer()
