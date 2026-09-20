"""License-plate recognisers.

Mirrors :mod:`app.services.evaluator`: a real OCR-backed implementation
(:class:`EasyOCRPlateRecognizer`) and a deterministic :class:`MockPlateRecognizer`, chosen by
:func:`build_plate_recognizer`. The mock is used only when explicitly requested; a broken EasyOCR
install raises instead of silently serving fake plates.
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
from app.schemas.common import BoundingBox
from app.schemas.plate import PlateReading
from app.services.plate_locator import locate_plate_regions

logger = logging.getLogger(__name__)

# Taiwan layouts once separators are removed: ABC1234 / AB1234 (cars), 1234AB / 123ABC (older),
# ABC123 (motorcycles). The hyphen is re-inserted by ``format_plate``.
_PLATE_RE = re.compile(
    r"(?P<letters>[A-Z]{2,3})(?P<digits>\d{3,4})|(?P<digits2>\d{3,4})(?P<letters2>[A-Z]{2,3})"
)
# Common OCR confusions, applied only inside the segment that should be letters / digits.
_TO_DIGIT = str.maketrans("OQDIZSB", "0001258")
_TO_LETTER = str.maketrans("0158", "QISB")
# Taiwan plates never use the letter 'O' (too close to '0'), so a letter-position 'O' is a 'Q'.
_FIX_LETTERS = str.maketrans("O", "Q")
_WORK_SIDE = 1600  # longest side, px, for locating plates / full-frame OCR
_MIN_CROP_WIDTH = 320  # upscale smaller plate crops to at least this width before OCR


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
            return f"{m.group('letters').translate(_FIX_LETTERS)}-{m.group('digits')}"
        return f"{m.group('digits2')}-{m.group('letters2').translate(_FIX_LETTERS)}"
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

    def _ocr(
        self,
        image: Image.Image,
        origin: tuple[int, int] = (0, 0),
        scale: float = 1.0,
    ) -> list[RawPlate]:
        """OCR ``image``; ``scale`` maps pixels to the original image, ``origin`` then shifts."""

        results = self._reader.readtext(
            np.asarray(image),
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            detail=1,
        )
        ox, oy = origin
        plates: list[RawPlate] = []
        for points, text, conf in results:
            xs = [p[0] * scale + ox for p in points]
            ys = [p[1] * scale + oy for p in points]
            plates.append(RawPlate(text, float(conf), (min(xs), min(ys), max(xs), max(ys))))
        return plates

    def read(self, image: Image.Image) -> list[RawPlate]:
        # Locate and run the full-frame fallback on a downscaled copy (phone photos are huge and
        # OCR on them is slow); crops are cut from the original so small plates keep their detail.
        scale = min(1.0, _WORK_SIDE / max(image.size))
        work = image if scale == 1.0 else image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )

        found: list[RawPlate] = []
        if self._use_locator:
            for bx1, by1, bx2, by2 in locate_plate_regions(work):
                x1, y1, x2, y2 = (round(v / scale) for v in (bx1, by1, bx2, by2))
                crop = image.crop((x1, y1, x2, y2))
                if crop.width < _MIN_CROP_WIDTH:
                    up = _MIN_CROP_WIDTH / crop.width
                    crop = crop.resize(
                        (_MIN_CROP_WIDTH, round(crop.height * up)), Image.Resampling.LANCZOS
                    )
                    found.extend(self._ocr(crop, origin=(x1, y1), scale=1 / up))
                else:
                    found.extend(self._ocr(crop, origin=(x1, y1)))
            if any(format_plate(p.text) for p in found):
                return found

        # No crop produced a valid plate (or the locator is off): scan the whole frame.
        return found + self._ocr(work, scale=1.0 / scale)


def _to_reading(raw: RawPlate, size: tuple[int, int]) -> PlateReading:
    formatted = format_plate(raw.text)
    box = None
    if raw.xyxy:
        w, h = size
        x1, y1, x2, y2 = (
            max(0.0, min(v, limit)) for v, limit in zip(raw.xyxy, (w, h, w, h), strict=True)
        )
        if x2 > x1 and y2 > y1:
            box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return PlateReading(
        plate_number=formatted or raw.text.upper(),
        raw_text=raw.text,
        confidence=max(0.0, min(raw.confidence, 1.0)),
        valid_format=formatted is not None,
        bounding_box=box,
    )


def _merge_split_plate(raws: list[RawPlate]) -> RawPlate | None:
    """Plates are often OCR'd as two boxes ('ABC' + '1234'); try joining neighbours."""

    for a, b in zip(raws, raws[1:], strict=False):
        if format_plate(a.text) is None and format_plate(b.text) is None:
            joined = a.text + b.text
            if format_plate(joined):
                boxes = [x for x in (a.xyxy, b.xyxy) if x]
                xyxy = None
                if boxes:
                    xyxy = (
                        min(x[0] for x in boxes),
                        min(x[1] for x in boxes),
                        max(x[2] for x in boxes),
                        max(x[3] for x in boxes),
                    )
                return RawPlate(joined, min(a.confidence, b.confidence), xyxy)
    return None


def resolve_plates(
    raws: list[RawPlate], size: tuple[int, int], min_confidence: float
) -> tuple[list[PlateReading], PlateReading | None]:
    """Turn raw OCR regions into ranked readings plus the best valid plate (or ``None``)."""

    merged = _merge_split_plate(raws)
    candidates = ([merged] if merged else []) + raws
    plates = [_to_reading(r, size) for r in candidates]
    plates = [p for p in plates if p.valid_format or p.confidence >= min_confidence]
    plates.sort(key=lambda p: (p.valid_format, p.confidence), reverse=True)
    best = next((p for p in plates if p.valid_format), None)
    return plates, best


def build_plate_recognizer(settings: Settings) -> PlateRecognizer:
    """Build the real recogniser, or the mock only if ``IRENT_PLATE_USE_MOCK=true``.

    A missing/broken EasyOCR install raises rather than silently serving fabricated plates.
    """

    if settings.plate_use_mock:
        logger.warning("IRENT_PLATE_USE_MOCK is set: plate results are FAKE")
        return MockPlateRecognizer()
    try:
        return EasyOCRPlateRecognizer(
            gpu=settings.yolo_device != "cpu", use_locator=settings.plate_use_opencv_locator
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the EasyOCR plate recogniser "
            f"({type(exc).__name__}: {exc}). Install it with `pip install easyocr`, or set "
            "IRENT_PLATE_USE_MOCK=true to run with fake plate output."
        ) from exc
