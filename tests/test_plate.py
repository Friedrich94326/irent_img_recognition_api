from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.api.deps import get_plate_recognizer
from app.schemas.plate import PlateRecognitionResponse
from app.services.plate_locator import locate_plate_regions
from app.services.plate_recognizer import (
    EasyOCRPlateRecognizer,
    PlateRecognizer,
    RawPlate,
    format_plate,
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
    assert len(rec._reader.shapes) == len(locate_plate_regions(_scene_with_plate()))
