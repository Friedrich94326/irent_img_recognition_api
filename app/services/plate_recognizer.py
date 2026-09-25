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
from collections.abc import Callable
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
    refine_plate_quad,
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
_REREAD_PAD_FRAC = 0.15  # margin around a whole-frame hit before upscaling it for a second look
_DETECTOR_PAD_FRAC = 0.05  # detector boxes hug the plate; keep its frame inside the crop
_PLATE_CLASS_NAMES = {"license_plate", "plate"}
# Minimum widths plate crops are upscaled to, tried in order until one yields a valid plate. On
# the iRent benchmark photos 640 px read W/K/H and dropped glyphs best; 320 px rescues the rest.
_CROP_WIDTHS = (640, 320)
# The current car layout (ABC-1234). Older layouts stay valid; this only breaks ties in ranking.
_STANDARD_PLATE_RE = re.compile(r"[A-Z]{3}-\d{4}")


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
    """Return same-size variants of a straightened plate crop for OCR, best guess first.

    1. Colour, lightly denoised: EasyOCR's recognizer can lean on colour cues that flattening to
       greyscale throws away (observed on a real plate: a clean colour crop read a 'W' correctly,
       but every greyscale-derived variant below misread the same glyph as 'K' or 'Y'), so the
       colour crop gets its own OCR pass rather than only ever feeding it derived greyscale.
    2. CLAHE-equalised gray: local contrast makes glyph edges crisp under glare / shadow.
    3. Otsu-binarised with the outer frame blanked: removes the plate border that OCR otherwise
       reads as a leading 'I' / 'L'.

    Variants keep the crop's size so OCR boxes map back through one perspective matrix.
    """

    denoised_colour = cv2.bilateralFilter(warped_rgb, 7, 40, 40)
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
    return [denoised_colour, equalised, trimmed]


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


def _padded_quad(xyxy: tuple[float, float, float, float], pad_frac: float) -> np.ndarray:
    """Corners (tl, tr, br, bl) of ``xyxy`` grown by ``pad_frac`` of its size on every side."""

    x1, y1, x2, y2 = xyxy
    pad_x, pad_y = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    return np.array(
        [
            [x1 - pad_x, y1 - pad_y],
            [x2 + pad_x, y1 - pad_y],
            [x2 + pad_x, y2 + pad_y],
            [x1 - pad_x, y2 + pad_y],
        ],
        dtype="float32",
    )


class YOLOPlateDetector:
    """Finds plate boxes with an Ultralytics model that has a ``license_plate`` class.

    Called with an RGB image, it returns ``xyxy`` boxes, most confident first. ``ultralytics`` is
    imported lazily so the recogniser works without it when no plate weights are configured.
    """

    def __init__(self, settings: Settings) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        if settings.plate_detector_weights_path is None:
            raise ValueError("plate_detector_weights_path is not configured")
        self._model = YOLO(str(settings.plate_detector_weights_path))
        self._classes = [
            i for i, name in self._model.names.items() if name.lower() in _PLATE_CLASS_NAMES
        ]
        if not self._classes:
            raise ValueError(f"plate model has no license_plate class: {self._model.names}")
        self._conf = settings.plate_detector_confidence_threshold
        self._iou = settings.yolo_iou_threshold
        self._device = settings.plate_device or settings.yolo_device

    def __call__(self, image: Image.Image) -> list[tuple[float, float, float, float]]:
        results = self._model.predict(
            source=image,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            classes=self._classes,
            verbose=False,
        )
        scored: list[tuple[float, tuple[float, float, float, float]]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                scored.append((float(box.conf[0]), (x1, y1, x2, y2)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [xyxy for _, xyxy in scored]


def build_plate_detector(settings: Settings) -> YOLOPlateDetector | None:
    """Load the learned plate detector if configured; ``None`` (OpenCV locator only) otherwise."""

    weights = settings.plate_detector_weights_path
    if weights is None:
        return None
    if not weights.exists():
        logger.warning("Plate detector weights not found at %s - OpenCV locator only.", weights)
        return None
    try:
        detector = YOLOPlateDetector(settings)
        logger.info("Loaded plate detector from %s.", weights)
        return detector
    except Exception:  # noqa: BLE001 - a broken detector must not take plate reading down
        logger.exception("Failed to load plate detector - using the OpenCV locator only.")
        return None


class EasyOCRPlateRecognizer(PlateRecognizer):
    """OpenCV (and, if configured, a learned plate detector) finds plate regions; EasyOCR reads
    only those crops.

    If no region yields a valid plate, the whole image is read as a fallback.
    """

    name = "opencv+easyocr"
    _plate_detector: Callable[[Image.Image], list[tuple[float, float, float, float]]] | None = None

    def __init__(
        self,
        gpu: bool = False,
        use_locator: bool = True,
        plate_detector: Callable[[Image.Image], list[tuple[float, float, float, float]]]
        | None = None,
    ) -> None:
        import easyocr  # noqa: PLC0415 - heavy; deferred so the mock works without it

        if not gpu:
            _disable_cuda_pin_memory()
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self._use_locator = use_locator
        self._plate_detector = plate_detector
        if plate_detector is not None:
            self.name = "yolo+opencv+easyocr"

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

    def _read_quad(self, rgb: np.ndarray, quad: np.ndarray, min_width: int) -> list[RawPlate]:
        """Straighten ``quad`` of ``rgb`` to at least ``min_width`` px and OCR every variant."""

        try:
            warped, matrix = four_point_transform(rgb, quad, min_width=min_width)
            to_original = np.linalg.inv(matrix)
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            return []  # implausible or degenerate quad: never going to be a plate
        corners = tuple((float(x), float(y)) for x, y in order_points(quad))
        results: list[RawPlate] = []
        for variant in preprocess_plate(warped):
            results.extend(self._ocr(variant, to_original=to_original, corners=corners))
        return results

    def _reread_region(
        self, rgb: np.ndarray, xyxy: tuple[float, float, float, float], min_width: int
    ) -> list[RawPlate]:
        """Re-OCR a whole-frame hit at higher resolution.

        The whole-frame fallback reads a plate at its native size within the full photo, which
        is far more error-prone (e.g. misreading 'W' as 'K') than the locator path's tightly
        cropped, upscaled reads. Pad and upscale this hit's own box the same way a located
        candidate would be, and read it again; an implausible box (wrong aspect ratio) is simply
        skipped, since it was never going to be a plate anyway.

        The plate's own border is searched for around the hit first, so a tilted plate is truly
        perspective-corrected; the padded axis-aligned box is only the fallback.
        """

        quad = refine_plate_quad(rgb, xyxy)
        if quad is None:
            quad = _padded_quad(xyxy, _REREAD_PAD_FRAC)
        return self._read_quad(rgb, quad, min_width)

    def _read_at_width(
        self,
        rgb: np.ndarray,
        quads: list[np.ndarray],
        frame_hits: Callable[[], list[RawPlate]],
        min_width: int,
    ) -> list[RawPlate]:
        """Run the locate-then-whole-frame pipeline, straightening crops to ``min_width`` px."""

        found: list[RawPlate] = []
        for quad in quads:
            found.extend(self._read_quad(rgb, quad, min_width))
        if _has_valid_plate(found):
            return found

        # No crop produced a valid plate (or the locator is off): scan the whole frame, then
        # re-read each hit at locator-candidate quality instead of trusting the native-size text.
        refined: list[RawPlate] = []
        for raw in frame_hits():
            reread = self._reread_region(rgb, raw.xyxy, min_width) if raw.xyxy else []
            refined.extend(reread or [raw])
        return found + refined

    def read(self, image: Image.Image) -> list[RawPlate]:
        # Locate and run the full-frame fallback on a downscaled copy (phone photos are huge and
        # OCR on them is slow); plates are warped from the original so small ones keep detail.
        scale = min(1.0, _WORK_SIDE / max(image.size))
        work = image if scale == 1.0 else image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
        rgb = np.asarray(image.convert("RGB"))
        # Learned plate boxes first: _read_at_width stops at the first crop that reads a valid
        # plate, so the OpenCV candidates and the whole-frame scan only run when these fail.
        quads: list[np.ndarray] = []
        if self._plate_detector is not None:
            for xyxy in self._plate_detector(work):
                box = tuple(v / scale for v in xyxy)
                quad = refine_plate_quad(rgb, box)
                quads.append(quad if quad is not None else _padded_quad(box, _DETECTOR_PAD_FRAC))
        if self._use_locator:
            quads.extend(c.quad / scale for c in locate_plate_candidates(work))

        cached_hits: list[list[RawPlate]] = []

        def frame_hits() -> list[RawPlate]:
            # The native-size whole-frame scan does not depend on the crop width: run it once.
            if not cached_hits:
                cached_hits.append(self._ocr(work, scale=1.0 / scale))
            return cached_hits[0]

        # Glyph reads flip with crop size (a 'W' read as 'H'/'K'/'N'/'V' at one width, correctly
        # at another). Wider crops read best on the benchmark photos, so try them first and only
        # fall back to a narrower crop when the wider one yields no valid plate at all.
        raws: list[RawPlate] = []
        for min_width in _CROP_WIDTHS:
            raws = self._read_at_width(rgb, quads, frame_hits, min_width)
            if _has_valid_plate(raws):
                break
        return raws


def _has_valid_plate(raws: list[RawPlate]) -> bool:
    return any(format_plate(r.text) for r in raws) or _merge_split_plate(raws) is not None


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

    # Between equally-voted readings, prefer the current ABC-1234 layout: a dropped glyph
    # ('RDJ-077' for 'RDJ-0772') still matches an older layout, so it must not win a tie.
    plates.sort(
        key=lambda p: (
            p.valid_format,
            votes.get(p.plate_number, 0),
            bool(_STANDARD_PLATE_RE.fullmatch(p.plate_number)),
            p.confidence,
        ),
        reverse=True,
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
            gpu=device != "cpu",
            use_locator=settings.plate_use_opencv_locator,
            plate_detector=build_plate_detector(settings),
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the EasyOCR plate recogniser "
            f"({type(exc).__name__}: {exc}). Install it with `pip install easyocr`, or set "
            "IRENT_PLATE_USE_MOCK=true to run with fake plate output."
        ) from exc
