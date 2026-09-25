from __future__ import annotations

import io
import math

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.api.deps import get_plate_recognizer
from app.config import Settings
from app.schemas.plate import PlateRecognitionResponse
from app.services.plate_locator import (
    four_point_transform,
    locate_plate_candidates,
    locate_plate_regions,
    refine_plate_quad,
)
from app.services.plate_recognizer import (
    EasyOCRPlateRecognizer,
    PlateRecognizer,
    RawPlate,
    build_plate_recognizer,
    format_plate,
    preprocess_plate,
    resolve_plates,
)

ENDPOINT = "/api/v1/plate/recognize"


class FakeRecognizer(PlateRecognizer):
    name = "fake"

    def __init__(self, raws: list[RawPlate]) -> None:
        self._raws = raws

    def read(self, image):
        return self._raws


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ABC-1234", "ABC-1234"),
        ("abc 1234", "ABC-1234"),
        ("AB1234", "AB-1234"),
        ("1234-AB", "1234-AB"),
        ("A8C-I234", "ABC-1234"),
        ("HELLO", None),
    ],
)
def test_format_plate(text: str, expected: str | None) -> None:
    assert format_plate(text) == expected


def test_recognize_mock(client: TestClient, sample_jpeg: bytes) -> None:
    resp = client.post(ENDPOINT, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = PlateRecognitionResponse.model_validate(resp.json())
    assert body.best_plate is not None
    assert body.best_plate.valid_format


def test_recognize_merges_split_boxes(client: TestClient, sample_jpeg: bytes) -> None:
    fake = FakeRecognizer(
        [RawPlate("ABC", 0.9, (10, 10, 50, 40)), RawPlate("1234", 0.8, (55, 10, 120, 40))]
    )
    client.app.dependency_overrides[get_plate_recognizer] = lambda: fake
    resp = client.post(ENDPOINT, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")})
    assert resp.json()["best_plate"]["plate_number"] == "ABC-1234"


def test_recognize_no_plate(client: TestClient, sample_jpeg: bytes) -> None:
    client.app.dependency_overrides[get_plate_recognizer] = lambda: FakeRecognizer([])
    body = client.post(ENDPOINT, files={"file": ("c.jpg", sample_jpeg, "image/jpeg")}).json()
    assert body["plates"] == []
    assert body["best_plate"] is None


def test_recognize_rejects_bad_type(client: TestClient) -> None:
    resp = client.post(ENDPOINT, files={"file": ("a.txt", b"hi", "text/plain")})
    assert resp.status_code == 415


def _scene_with_plate() -> Image.Image:
    img = Image.new("RGB", (800, 600), (90, 90, 95))
    draw = ImageDraw.Draw(img)
    draw.rectangle((300, 380, 520, 452), fill=(245, 245, 245), outline=(0, 0, 0), width=4)
    for i in range(7):
        x = 315 + i * 28
        draw.rectangle((x, 395, x + 16, 437), fill=(20, 20, 20))
    return img


def test_locator_finds_plate_rectangle() -> None:
    boxes = locate_plate_regions(_scene_with_plate())
    assert boxes, "expected at least one candidate"
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 300 and y1 <= 380 and x2 >= 520 and y2 >= 452
    assert (x2 - x1) < 400  # a tight crop, not the whole frame


class _FakeReader:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def readtext(self, arr, **_):
        self.shapes.append(arr.shape)
        h, w = arr.shape[:2]
        return [([[0, 0], [w, 0], [w, h], [0, h]], "ABC-1234", 0.9)]


def test_recognizer_reads_crop_and_offsets_boxes() -> None:
    rec = EasyOCRPlateRecognizer.__new__(EasyOCRPlateRecognizer)
    rec._reader, rec._use_locator = _FakeReader(), True
    plates = rec.read(_scene_with_plate())
    assert plates[0].text == "ABC-1234"
    assert rec._reader.shapes[0][1] < 800  # OCR ran on a crop, not the full frame
    assert plates[0].xyxy[0] > 0  # crop origin added back to coordinates
    # every located candidate is OCR'd once per preprocessing variant
    variants = len(preprocess_plate(np.zeros((40, 80, 3), dtype=np.uint8)))
    assert len(rec._reader.shapes) == variants * len(locate_plate_regions(_scene_with_plate()))


class _NarrowOnlyReader(_FakeReader):
    """Reads the plate only from 320 px crops, like a glyph that flips with crop size."""

    def readtext(self, arr, **_):
        self.shapes.append(arr.shape)
        h, w = arr.shape[:2]
        return [([[0, 0], [w, 0], [w, h], [0, h]], "ABC-1234" if w == 320 else "XY", 0.9)]


def test_recognizer_tries_wide_crop_first_then_falls_back_to_narrow() -> None:
    rec = EasyOCRPlateRecognizer.__new__(EasyOCRPlateRecognizer)
    rec._reader, rec._use_locator = _NarrowOnlyReader(), True
    plates = rec.read(_scene_with_plate())
    assert any(p.text == "ABC-1234" for p in plates)
    widths = [shape[1] for shape in rec._reader.shapes]
    assert widths[0] == 640  # the wide crop is read first
    assert 320 in widths  # and the narrow crop only after it found no valid plate


def test_recognizer_stops_at_wide_crop_when_it_reads_a_plate() -> None:
    rec = EasyOCRPlateRecognizer.__new__(EasyOCRPlateRecognizer)
    rec._reader, rec._use_locator = _FakeReader(), True
    rec.read(_scene_with_plate())
    assert all(shape[1] == 640 for shape in rec._reader.shapes)


class _FirstCropReader(_FakeReader):
    """Reads a plate from the first crop only, so the test sees which quad was tried first."""

    def readtext(self, arr, **_):
        self.shapes.append(arr.shape)
        h, w = arr.shape[:2]
        text = "ABC-1234" if len(self.shapes) == 1 else "XY"
        return [([[0, 0], [w, 0], [w, h], [0, h]], text, 0.9)]


def test_recognizer_reads_plate_detector_boxes_before_locator() -> None:
    rec = EasyOCRPlateRecognizer.__new__(EasyOCRPlateRecognizer)
    rec._reader, rec._use_locator = _FirstCropReader(), True
    calls = []

    def detector(image):
        calls.append(image.size)
        return [(300.0, 380.0, 520.0, 452.0)]

    rec._plate_detector = detector
    plates = rec.read(_scene_with_plate())
    assert calls == [(800, 600)]
    first = next(p for p in plates if p.text == "ABC-1234")
    x1, y1, x2, y2 = first.xyxy
    assert abs(x1 - 300) < 25 and abs(y1 - 380) < 25 and abs(x2 - 520) < 25 and abs(y2 - 452) < 25


def test_build_plate_recognizer_without_detector_weights(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_init(self, gpu=False, use_locator=True, plate_detector=None):
        captured["detector"] = plate_detector

    monkeypatch.setattr(EasyOCRPlateRecognizer, "__init__", fake_init)
    build_plate_recognizer(Settings(_env_file=None, plate_use_mock=False))
    assert captured["detector"] is None
    missing = tmp_path / "missing.pt"
    build_plate_recognizer(
        Settings(_env_file=None, plate_use_mock=False, plate_detector_weights_path=missing)
    )
    assert captured["detector"] is None  # missing weights: OpenCV locator only, no crash


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IRCK9206", "RCK-9206"),  # leading 'I' artefact removed (K vs W is left to alternatives)
        ("LRCW9206", "RCW-9206"),
        ("RCW-9206", "RCW-9206"),
        ("RCW 9206", "RCW-9206"),
        ("IABCD1234", None),  # still too many letters after stripping noise
    ],
)
def test_format_plate_strips_leading_noise(text: str, expected: str | None) -> None:
    assert format_plate(text) == expected


def test_resolve_plates_offers_k_w_alternative_below_original() -> None:
    raws = [RawPlate("IRCK9206", 0.8, (0, 0, 100, 40))]
    plates, best = resolve_plates(raws, (200, 100), 0.5)
    assert best is not None and best.plate_number == "RCK-9206"
    assert [p.plate_number for p in plates] == ["RCK-9206", "RCW-9206"]


def test_resolve_plates_prefers_plate_seen_in_more_variants() -> None:
    raws = [
        RawPlate("RCW9206", 0.6, (0, 0, 100, 40)),
        RawPlate("RCW9206", 0.6, (0, 0, 100, 40)),
        RawPlate("RCK9206", 0.9, (0, 0, 100, 40)),
    ]
    _, best = resolve_plates(raws, (200, 100), 0.5)
    assert best is not None and best.plate_number == "RCW-9206"


def test_resolve_plates_prefers_standard_layout_on_a_vote_tie() -> None:
    # A dropped last digit still matches the older ABC-123 layout; it must not win a tie.
    raws = [RawPlate("RDJ077", 0.9, (0, 0, 100, 40)), RawPlate("RDJ0772", 0.6, (0, 0, 100, 40))]
    _, best = resolve_plates(raws, (200, 100), 0.5)
    assert best is not None and best.plate_number == "RDJ-0772"


def test_resolve_plates_votes_outrank_standard_layout() -> None:
    raws = [
        RawPlate("AB1234", 0.6, (0, 0, 100, 40)),
        RawPlate("AB1234", 0.6, (0, 0, 100, 40)),
        RawPlate("ABC1234", 0.9, (0, 0, 100, 40)),
    ]
    _, best = resolve_plates(raws, (200, 100), 0.5)
    assert best is not None and best.plate_number == "AB-1234"


def test_four_point_transform_straightens_skewed_plate() -> None:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    quad = np.array([[60, 80], [300, 110], [290, 200], [50, 160]], dtype="float32")
    cv2.fillPoly(img, [quad.astype(np.int32)], (255, 255, 255))
    warped, matrix = four_point_transform(img, quad, min_width=200)
    assert warped.shape[1] >= 200
    assert warped[5:-5, 5:-5].mean() > 240  # interior is all plate: the skew was removed
    top_left = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), matrix)[0, 0]
    assert np.allclose(top_left, [0, 0], atol=1)


def test_four_point_transform_rejects_implausible_aspect() -> None:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # A near-square quad: no real plate has this aspect ratio, valid or oblique.
    quad = np.array([[50, 50], [200, 55], [195, 200], [45, 195]], dtype="float32")
    with pytest.raises(ValueError, match="aspect"):
        four_point_transform(img, quad, min_width=50)


def test_four_point_transform_rejects_sliver_quad() -> None:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # A plausible-aspect quad with one corner pinched to ~161 degrees: a bad contour-approximation
    # artifact (near-collinear edge), not a real plate corner.
    quad = np.array([[10, 140], [300, 0], [300, 150], [0, 150]], dtype="float32")
    with pytest.raises(ValueError, match="quadrilateral"):
        four_point_transform(img, quad, min_width=50)


def _rotated_plate_corners(
    angle_deg: float, cx=400, cy=410, w=220, h=72
) -> list[tuple[float, float]]:
    a = math.radians(angle_deg)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [
        (cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a))
        for x, y in corners
    ]


def test_locator_finds_rotated_plate() -> None:
    # A plate-shaped quad rotated 10 degrees, not axis-aligned.
    quad = _rotated_plate_corners(10)
    img = Image.new("RGB", (800, 600), (90, 90, 95))
    draw = ImageDraw.Draw(img)
    draw.polygon(quad, fill=(245, 245, 245), outline=(0, 0, 0), width=4)

    candidates = locate_plate_candidates(img)
    assert candidates, "expected at least one candidate"
    found = candidates[0].quad
    # A rotated quad's corners must not collapse onto an axis-aligned box: both x- and y-span
    # should reflect the rotated rectangle's extent, not just its width or height alone.
    xs, ys = found[:, 0], found[:, 1]
    assert xs.max() - xs.min() > 150
    assert ys.max() - ys.min() > 50
    # The found top-left corner should stay close to the drawn one (within the locator's padding).
    drawn_top_left = min(quad, key=lambda p: p[0] + p[1])
    top_left_found = found[np.argmin(found.sum(axis=1))]
    assert abs(top_left_found[0] - drawn_top_left[0]) < 40
    assert abs(top_left_found[1] - drawn_top_left[1]) < 40


def test_refine_plate_quad_finds_tilted_plate_around_text_box() -> None:
    # A white plate on a white car body, framed by a dark grille: the global locator's wide
    # close merges the two, so the plate must be found from its OCR text box instead.
    quad = _rotated_plate_corners(-12, cx=300, cy=260, w=200, h=90)
    img = Image.new("RGB", (600, 500), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle((150, 190, 450, 340), fill=(30, 30, 30))
    draw.polygon(quad, fill=(245, 245, 245))
    draw.text((255, 250), "ABC-1234", fill=(0, 0, 0))

    # The OCR box covers most of the plate, as the whole-frame fallback's text hits do.
    found = refine_plate_quad(np.asarray(img), (215, 215, 385, 305))
    assert found is not None
    # Each corner lands near the drawn one (allowing for the small outward quad padding), so
    # the result is the tilted plate, not the axis-aligned text box.
    for drawn, got in zip(quad, found, strict=True):
        assert abs(got[0] - drawn[0]) < 15 and abs(got[1] - drawn[1]) < 15
    assert abs(found[0][1] - found[1][1]) > 20  # top edge is tilted


def test_refine_plate_quad_returns_none_without_a_plate() -> None:
    img = np.full((300, 400, 3), 128, dtype=np.uint8)
    assert refine_plate_quad(img, (150, 120, 250, 170)) is None


def test_preprocess_plate_variants_keep_crop_size() -> None:
    crop = np.full((60, 130, 3), 240, dtype=np.uint8)
    variants = preprocess_plate(crop)
    # The colour variant keeps its 3 channels; the rest are flattened to grayscale.
    assert variants and all(v.shape[:2] == (60, 130) for v in variants)
    assert sum(v.ndim == 3 for v in variants) == 1


def test_format_plate_maps_letter_o_to_q() -> None:
    # Taiwan plates never use the letter O, so OCR's 'O' in a letter slot is a 'Q'.
    assert format_plate("RCO-6760") == "RCQ-6760"


def test_build_plate_recognizer_raises_instead_of_mocking(monkeypatch) -> None:
    def boom(self, *args, **kwargs):
        raise ImportError("No module named 'easyocr'")

    monkeypatch.setattr(EasyOCRPlateRecognizer, "__init__", boom)
    with pytest.raises(RuntimeError, match="IRENT_PLATE_USE_MOCK"):
        build_plate_recognizer(Settings(_env_file=None, plate_use_mock=False))


def test_build_plate_recognizer_mock_only_when_requested() -> None:
    rec = build_plate_recognizer(Settings(_env_file=None, plate_use_mock=True))
    assert rec.is_mock


def test_upload_applies_exif_rotation(client: TestClient) -> None:
    img = Image.new("RGB", (40, 20), (200, 30, 30))
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 270 to display -> portrait
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    resp = client.post(ENDPOINT, files={"file": ("p.jpg", buf.getvalue(), "image/jpeg")})
    assert resp.status_code == 200, resp.text
    assert (resp.json()["image"]["width"], resp.json()["image"]["height"]) == (20, 40)


def test_health_reports_plate_recognizer(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["plate_model_name"]
    assert body["plate_is_mock"] is True  # tests force the mock via IRENT_PLATE_USE_MOCK
