"""Corner condition photos: upload, image serving, vehicle list, and the Hotai import's parsing."""

from __future__ import annotations

import io
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import (
    get_corner_classifier,
    get_evaluator,
    get_settings,
    get_vehicle_photo_repository,
)
from app.config import Settings
from app.main import create_app
from app.schemas.common import DamageClass
from app.schemas.vehicle import Corner, PhotoSource
from app.services.corner_classifier import CornerClassifier, CornerPrediction
from app.services.evaluator import DamageEvaluator, RawDetection
from app.services.vehicle_photos import VehiclePhotoRepository, inspect_photo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_hotai_photos import parse_name, pick_latest  # noqa: E402

SOURCE_DB = Path("data/irent_op_backend.sqlite")
PLATE = "RMU-5268"

needs_db = pytest.mark.skipif(not SOURCE_DB.exists(), reason="ops database not present")


class RecordingEvaluator(DamageEvaluator):
    name = "fixed"
    version = "test"
    is_mock = True

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def predict(self, image: Image.Image) -> list[RawDetection]:
        self.sizes.append(image.size)
        return [RawDetection(DamageClass.DENT, 0.8, (10.0, 20.0, 110.0, 90.0))]


@pytest.fixture
def repo(tmp_path: Path) -> VehiclePhotoRepository:
    db = tmp_path / "ops.sqlite"
    shutil.copy(SOURCE_DB, db)
    repo = VehiclePhotoRepository(db, tmp_path / "photos")
    repo.ensure_schema()
    return repo


class FixedCorner(CornerClassifier):
    name = "fixed-corner"

    def __init__(self, corner: Corner, confidence: float) -> None:
        self.prediction = CornerPrediction(corner, confidence)

    def predict(self, image: Image.Image) -> CornerPrediction:
        return self.prediction


def make_client(
    repo: VehiclePhotoRepository | None,
    evaluator: DamageEvaluator | None = None,
    classifier: CornerClassifier | None = None,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    app.dependency_overrides[get_evaluator] = lambda: evaluator or RecordingEvaluator()
    app.dependency_overrides[get_corner_classifier] = lambda: classifier
    app.dependency_overrides[get_vehicle_photo_repository] = lambda: repo
    return TestClient(app)


def jpeg(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 140, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def upload(client: TestClient, plate: str, data: dict, size=(640, 480)):
    return client.post(
        f"/api/v1/vehicles/{plate}/photos",
        data=data,
        files={"file": ("corner.jpg", jpeg(size), "image/jpeg")},
    )


@needs_db
def test_upload_stores_upright_photo_with_detections(repo: VehiclePhotoRepository) -> None:
    evaluator = RecordingEvaluator()
    with make_client(repo, evaluator) as client:
        resp = upload(client, "rmu5268", {"corner": "rear_left", "source": "return",
                                          "captured_at": "2026-09-30 08:00:00"}, size=(480, 640))
        assert resp.status_code == 201, resp.text
        body = resp.json()
        image = client.get(body["image_url"])

    # The detector sees a landscape copy; the stored photo and its boxes stay upright.
    assert evaluator.sizes == [(640, 480)]
    assert (body["width"], body["height"]) == (480, 640)
    box = body["detections"][0]["bounding_box"]
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (390.0, 10.0, 460.0, 110.0)
    assert body["corner"] == "rear_left"
    assert body["corner_label"] == "左後"
    assert body["source"] == "return"
    assert body["captured_at"] == "2026-09-30 08:00:00"
    assert body["summary"]["total_detections"] == 1
    assert body["detections"][0]["damage_class"] == "dent"

    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(image.content)).size == (480, 640)

    [photo] = repo.latest_by_corner(repo.find_vehicle_id(PLATE)).values()
    assert photo.id == body["photo_id"]
    assert repo.file_for(photo).is_file()


@needs_db
@pytest.mark.parametrize(
    ("classifier", "expected"),
    [
        (None, (None, None, False)),
        (FixedCorner(Corner.FRONT_LEFT, 0.95), ("front_left", 0.95, False)),  # agrees
        (FixedCorner(Corner.REAR_RIGHT, 0.95), ("rear_right", 0.95, True)),  # confident mismatch
        (FixedCorner(Corner.REAR_RIGHT, 0.40), ("rear_right", 0.40, False)),  # unsure: no flag
    ],
    ids=["no-classifier", "match", "mismatch", "unsure"],
)
def test_upload_checks_corner_but_keeps_label(
    repo: VehiclePhotoRepository, classifier: CornerClassifier | None, expected: tuple
) -> None:
    with make_client(repo, classifier=classifier) as client:
        resp = upload(client, PLATE, {"corner": "front_left"})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["corner"] == "front_left"  # the label stays the corner of record
    check = (body["predicted_corner"], body["corner_confidence"], body["corner_mismatch"])
    assert check == expected
    stored = repo.get(body["photo_id"])
    assert stored.corner_mismatch is expected[2]
    assert (stored.predicted_corner.value if stored.predicted_corner else None) == expected[0]


@needs_db
def test_set_corner_check_updates_stored_photo(repo: VehiclePhotoRepository) -> None:
    with make_client(repo) as client:
        photo_id = upload(client, PLATE, {"corner": "rear_left"}).json()["photo_id"]
    repo.set_corner_check(photo_id, CornerPrediction(Corner.FRONT_RIGHT, 0.88), True)
    photo = repo.get(photo_id)
    assert (photo.predicted_corner, photo.corner_confidence, photo.corner_mismatch) == (
        Corner.FRONT_RIGHT, 0.88, True
    )
    assert repo.all_photos()[-1] == photo


@needs_db
def test_set_inspection_replaces_stored_damage(repo: VehiclePhotoRepository) -> None:
    with make_client(repo) as client:  # RecordingEvaluator: one dent
        photo_id = upload(client, PLATE, {"corner": "front_left"}).json()["photo_id"]
    clean = inspect_photo(EmptyEvaluator(), Image.new("RGB", (64, 48)))
    repo.set_inspection(photo_id, clean, "rechecked")
    photo = repo.get(photo_id)
    assert (photo.detections, photo.overall_severity.value, photo.model_name) == (
        [], "none", "rechecked"
    )


class EmptyEvaluator(DamageEvaluator):
    name = "empty"

    def predict(self, image: Image.Image) -> list[RawDetection]:
        return []


def test_damage_threshold_defaults_to_70_percent() -> None:
    assert Settings(_env_file=None).yolo_confidence_threshold == 0.70


def test_old_table_gets_corner_check_columns(tmp_path: Path) -> None:
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        'CREATE TABLE "vehicle_photos" ("id" INTEGER PRIMARY KEY, "vehicle_id" INTEGER NOT NULL, '
        '"corner" TEXT NOT NULL, "file_path" TEXT NOT NULL, "captured_at" TEXT NOT NULL)'
    )
    conn.close()
    repo = VehiclePhotoRepository(db, tmp_path / "photos")
    repo.ensure_schema()
    repo.ensure_schema()
    conn = sqlite3.connect(db)
    try:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(vehicle_photos)")}
    finally:
        conn.close()
    assert {"predicted_corner", "corner_confidence", "corner_mismatch"} <= columns


@needs_db
@pytest.mark.parametrize(
    ("plate", "data", "status"),
    [("ZZZ-9999", {"corner": "front_left"}, 404),
     (PLATE, {"corner": "roof"}, 422),
     (PLATE, {"corner": "front_left", "source": "drone"}, 422)],
)
def test_upload_rejects_bad_input(
    repo: VehiclePhotoRepository, plate: str, data: dict, status: int
) -> None:
    def count() -> tuple:
        conn = sqlite3.connect(repo.db_path)
        try:
            return conn.execute("SELECT COUNT(*) FROM vehicle_photos").fetchone()
        finally:
            conn.close()

    before = count()
    with make_client(repo) as client:
        assert upload(client, plate, data).status_code == status
    assert count() == before


@needs_db
def test_unknown_photo_is_404(repo: VehiclePhotoRepository) -> None:
    with make_client(repo) as client:
        assert client.get("/api/v1/vehicle-photos/999/image").status_code == 404


@needs_db
def test_vehicle_list_counts_corners(repo: VehiclePhotoRepository) -> None:
    with make_client(repo) as client:
        before = {v["license_plate"]: v for v in client.get("/api/v1/vehicles").json()}
        upload(client, PLATE, {"corner": "front_left"})
        upload(client, PLATE, {"corner": "front_left"})  # same corner again
        upload(client, PLATE, {"corner": "rear_right"})
        vehicles = client.get("/api/v1/vehicles").json()
    assert before[PLATE]["photo_corners"] == 0
    after = {v["license_plate"]: v["photo_corners"] for v in vehicles}
    assert after[PLATE] == 2
    counts = [v["photo_corners"] for v in vehicles]
    assert counts == sorted(counts, reverse=True)  # cars with photos first


class MarkerFinder(DamageEvaluator):
    """Reports the bounding box of the red marker in whatever image it is given."""

    name = "marker"
    version = "test"
    is_mock = True

    def predict(self, image: Image.Image) -> list[RawDetection]:
        red = image.convert("RGB").point(lambda v: 255 if v > 200 else 0).split()[0]
        mask = Image.eval(image.convert("RGB").split()[1], lambda v: 255 if v < 50 else 0)
        x1, y1, x2, y2 = Image.composite(red, Image.new("L", image.size), mask).getbbox()
        return [RawDetection(DamageClass.SCRATCH, 0.9, (x1, y1, x2, y2))]


@pytest.mark.parametrize("size", [(300, 500), (500, 300)], ids=["portrait", "landscape"])
def test_boxes_land_on_the_upright_photo(size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (40, 40, 40))
    image.paste((255, 0, 0), (60, 30, 110, 70))  # marker at x 60-110, y 30-70
    [det] = inspect_photo(MarkerFinder(), image).detections
    b = det.bounding_box
    assert (b.x1, b.y1, b.x2, b.y2) == (60, 30, 110, 70)


def test_endpoints_without_database_are_503() -> None:
    with make_client(None) as client:
        assert client.get("/api/v1/vehicles").status_code == 503
        assert client.get("/api/v1/vehicle-photos/1/image").status_code == 503
        assert upload(client, PLATE, {"corner": "front_left"}).status_code == 503


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RCG2235右前 借.jfif", ("RCG-2235", Corner.FRONT_RIGHT, PhotoSource.PICKUP, 1)),
        ("RCR-7661 左後 還2.jpg", ("RCR-7661", Corner.REAR_LEFT, PhotoSource.RETURN, 2)),
        ("RDL7827左前還.jpeg", ("RDL-7827", Corner.FRONT_LEFT, PhotoSource.RETURN, 1)),
    ],
)
def test_parse_name(name: str, expected: tuple) -> None:
    p = parse_name(Path(name))
    assert (p.plate, p.corner, p.source, p.sequence) == expected


@pytest.mark.parametrize("name", ["RCW-8160 車況回報.jfif", "RCG2235右前 借.json", "notes.txt"])
def test_parse_name_skips_other_files(name: str) -> None:
    assert parse_name(Path(name)) is None


def test_pick_latest_prefers_last_return_and_needs_all_corners() -> None:
    names = [f"AAA1111{c} 借.jpg" for c in ("左前", "右前", "左後", "右後")]
    names += ["AAA1111左前 還.jpg", "AAA1111左前 還2.jpg", "BBB2222左前 借.jpg"]
    picked = pick_latest([parse_name(Path(n)) for n in names])
    assert list(picked) == ["AAA-1111"]  # BBB-2222 has one corner only
    corners = picked["AAA-1111"]
    assert corners[Corner.FRONT_LEFT].path.name == "AAA1111左前 還2.jpg"
    assert corners[Corner.REAR_RIGHT].source == PhotoSource.PICKUP
