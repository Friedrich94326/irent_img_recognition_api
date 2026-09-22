"""License-plate recognisers.

Mirrors :mod:`app.services.evaluator`: a real OCR-backed implementation
(:class:`EasyOCRPlateRecognizer`) and a deterministic :class:`MockPlateRecognizer`, chosen by
:func:`build_plate_recognizer`. The mock is used only when explicitly requested; a broken EasyOCR
install raises instead of silently serving fake plates.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.config import Settings
from app.schemas.common import BoundingBox
from app.schemas.plate import PlateReading
from app.services.plate_locator import (
    four_point_transform,
    locate_plate_candidates,
    order_points,
)

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
_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Glyphs that frame edges, screws and stickers get read as; only ever stripped from the *front* of
# an over-long letter run (e.g. 'IRCK9206' -> 'RCK9206').
_LEADING_NOISE = "ILJ"
_LONG_LETTER_RUN_RE = re.compile(r"(?P<letters>[A-Z]{4,})(?P<digits>\d{3,4})")
_CLAHE_CLIP = 2.0
_BORDER_TRIM_FRAC = 0.04  # share of each side blanked so the plate frame is not read as glyphs
# Visually close pairs; used to offer an alternative reading, never to overwrite the original.
_LETTER_SWAPS = str.maketrans("KW", "WK")
_ALT_CONFIDENCE_FACTOR = 0.5


@dataclass(frozen=True)
class RawPlate:
    text: str
    confidence: float
    xyxy: tuple[float, float, float, float] | None
    # Corners (tl, tr, br, bl) of the plate the text was read from, in original-image pixels; only
    # known when the OpenCV locator found the plate, so it can be straightened for display.
    corners: tuple[tuple[float, float], ...] | None = None


def _clean(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _strip_leading_noise(cleaned: str) -> str:
    """Drop leading I/L/J artefacts while the letter run is longer than the 3 a plate allows."""

    m = _LONG_LETTER_RUN_RE.fullmatch(cleaned)
    if not m:
        return cleaned
    letters = m.group("letters")
    while len(letters) > 3 and letters[0] in _LEADING_NOISE:
        letters = letters[1:]
    return letters + m.group("digits")


def format_plate(text: str) -> str | None:
    """Return ``text`` as a hyphenated Taiwan plate (e.g. ``ABC-1234``) or ``None``."""

    cleaned = _strip_leading_noise(_clean(text))
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


def preprocess_plate(warped_rgb: np.ndarray) -> list[np.ndarray]:
    """Return same-size grayscale variants of a straightened plate crop, best guess first.

    1. CLAHE-equalised gray: local contrast makes glyph edges crisp under glare / shadow.
    2. Otsu-binarised with the outer frame blanked: removes the plate border that OCR otherwise
       reads as a leading 'I' / 'L'.

    Variants keep the crop's size so OCR boxes map back through one perspective matrix.
    """

    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)  # denoise, keep glyph edges
    equalised = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=(4, 4)).apply(gray)
    _, binary = cv2.threshold(equalised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    trimmed = binary.copy()
    background = 255 if binary.mean() > 127 else 0  # plates are dark glyphs on a light plate
    h, w = trimmed.shape
    dy, dx = max(1, round(h * _BORDER_TRIM_FRAC)), max(1, round(w * _BORDER_TRIM_FRAC))
    trimmed[:dy, :] = trimmed[-dy:, :] = background
    trimmed[:, :dx] = trimmed[:, -dx:] = background
    return [equalised, trimmed]


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


def _disable_cuda_pin_memory() -> None:
    """Work around an EasyOCR bug that breaks CPU-only OCR on a machine with a GPU.

    EasyOCR's recognizer hardcodes ``pin_memory=True`` on its internal ``DataLoader``
    regardless of the ``gpu`` flag it was constructed with. Pinning tries to talk to CUDA even
    when we asked for the CPU path, so on a machine where a CUDA device exists but is busy or
    otherwise unavailable (e.g. held by another process), every OCR call raises instead of
    running on the CPU as configured. Pinning is a transfer-speed optimisation only, never
    required for correctness, so making it a no-op is safe: nothing breaks for a fully-CPU
    pipeline, and it only forgoes that optimisation for any other CUDA work in the process.
    """

    import torch  # noqa: PLC0415 - heavy; only needed for this one-time patch

    if getattr(torch.Tensor.pin_memory, "_irent_patched", False):
        return
    patched = lambda self, *args, **kwargs: self  # noqa: E731
    patched._irent_patched = True
    torch.Tensor.pin_memory = patched


class EasyOCRPlateRecognizer(PlateRecognizer):
    """OpenCV finds plate-shaped regions; EasyOCR reads only those crops.

    If no region yields a valid plate, the whole image is read as a fallback.
    """

    name = "opencv+easyocr"

    def __init__(self, gpu: bool = False, use_locator: bool = True) -> None:
        import easyocr  # noqa: PLC0415 - heavy; deferred so the mock works without it

        if not gpu:
            _disable_cuda_pin_memory()
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self._use_locator = use_locator

    def _ocr(
        self,
        image: Image.Image | np.ndarray,
        scale: float = 1.0,
        to_original: np.ndarray | None = None,
        corners: tuple[tuple[float, float], ...] | None = None,
    ) -> list[RawPlate]:
        """OCR ``image``.

        Boxes are mapped back to the original photo either by ``scale`` (plain resize) or, for a
        perspective-corrected crop, by ``to_original`` (the inverse of the warp matrix).
        """

        results = self._reader.readtext(np.asarray(image), allowlist=_ALLOWLIST, detail=1)
        plates: list[RawPlate] = []
        for points, text, conf in results:
            pts = np.asarray(points, dtype="float32").reshape(-1, 1, 2)
            if to_original is not None:
                pts = cv2.perspectiveTransform(pts, to_original)
            else:
                pts = pts * scale
            xs, ys = pts[:, 0, 0], pts[:, 0, 1]
            plates.append(
                RawPlate(
                    text,
                    float(conf),
                    (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                    corners,
                )
            )
        return plates

    def read(self, image: Image.Image) -> list[RawPlate]:
        # Locate and run the full-frame fallback on a downscaled copy (phone photos are huge and
        # OCR on them is slow); plates are warped from the original so small ones keep detail.
        scale = min(1.0, _WORK_SIDE / max(image.size))
        work = image if scale == 1.0 else image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )

        found: list[RawPlate] = []
        if self._use_locator:
            original = np.asarray(image.convert("RGB"))
            for cand in locate_plate_candidates(work):
                try:
                    warped, matrix = four_point_transform(original, cand.quad / scale)
                    to_original = np.linalg.inv(matrix)
                except (ValueError, np.linalg.LinAlgError, cv2.error):
                    continue  # degenerate quad; try the next candidate
                corners = tuple(
                    (float(x), float(y)) for x, y in order_points(cand.quad / scale)
                )
                for variant in preprocess_plate(warped):
                    found.extend(self._ocr(variant, to_original=to_original, corners=corners))
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
        corners=[list(c) for c in raw.corners] if raw.corners else None,
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
                corners = a.corners or b.corners
                return RawPlate(joined, min(a.confidence, b.confidence), xyxy, corners)
    return None


def resolve_plates(
    raws: list[RawPlate], size: tuple[int, int], min_confidence: float
) -> tuple[list[PlateReading], PlateReading | None]:
    """Turn raw OCR regions into ranked readings plus the best valid plate (or ``None``)."""

    merged = _merge_split_plate(raws)
    candidates = ([merged] if merged else []) + raws
    plates = [_to_reading(r, size) for r in candidates]
    plates = [p for p in plates if p.valid_format or p.confidence >= min_confidence]

    # The same plate read from several preprocessing variants is more trustworthy than one read.
    votes = Counter(p.plate_number for p in plates if p.valid_format)
    # Offer the K<->W lookalike as a lower-ranked alternative; without a plate registry the text
    # alone cannot say which is right, so the original reading always stays on top.
    alternatives = [_lookalike_alternative(p) for p in plates if p.valid_format]
    plates += [a for a in alternatives if a is not None and a.plate_number not in votes]

    plates.sort(
        key=lambda p: (p.valid_format, votes.get(p.plate_number, 0), p.confidence), reverse=True
    )
    best = next((p for p in plates if p.valid_format), None)
    return plates, best


def rectify_plate(image: Image.Image, corners: list[list[float]]) -> str | None:
    """Return the plate at ``corners`` as a front-on PNG data URL, or ``None`` on failure."""

    try:
        warped, _ = four_point_transform(np.asarray(image.convert("RGB")), np.array(corners))
    except (ValueError, cv2.error):
        return None
    ok, png = cv2.imencode(".png", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def _lookalike_alternative(reading: PlateReading) -> PlateReading | None:
    """A copy of ``reading`` with K/W swapped in its letters, at reduced confidence."""

    swapped = reading.plate_number.translate(_LETTER_SWAPS)  # digits and '-' are unaffected
    if swapped == reading.plate_number:
        return None
    return reading.model_copy(
        update={
            "plate_number": swapped,
            "confidence": reading.confidence * _ALT_CONFIDENCE_FACTOR,
        }
    )


def build_plate_recognizer(settings: Settings) -> PlateRecognizer:
    """Build the real recogniser, or the mock only if ``IRENT_PLATE_USE_MOCK=true``.

    A missing/broken EasyOCR install raises rather than silently serving fabricated plates.
    """

    if settings.plate_use_mock:
        logger.warning("IRENT_PLATE_USE_MOCK is set: plate results are FAKE")
        return MockPlateRecognizer()
    device = settings.plate_device or settings.yolo_device
    try:
        return EasyOCRPlateRecognizer(
            gpu=device != "cpu", use_locator=settings.plate_use_opencv_locator
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the EasyOCR plate recogniser "
            f"({type(exc).__name__}: {exc}). Install it with `pip install easyocr`, or set "
            "IRENT_PLATE_USE_MOCK=true to run with fake plate output."
        ) from exc
