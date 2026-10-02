"""Linking damage / tire results to the iRent ops database (vehicles, damage_annotations,
ai_anomaly_alerts). Every test works on a throwaway copy of ``data/irent_op_backend.sqlite``."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import (
    get_evaluator,
    get_plate_recognizer,
    get_settings,
    get_tire_detector,
    get_vehicle_repository,
)
from app.config import Settings
from app.main import create_app
from app.schemas.common import DamageClass
from app.services.evaluator import DamageEvaluator, RawDetection
from app.services.plate_recognizer import PlateRecognizer, RawPlate
from app.services.tire_detector import MockTireDetector
from app.services.vehicle_repository import VehicleRepository

SOURCE_DB = Path("data/irent_op_backend.sqlite")
KNOWN_PLATE = "RAC-4582"  # vehicles.id = 1 in the seed data
DAMAGE = "/api/v1/damage/evaluate"
TIRE = "/api/v1/tire/detect"

pytestmark = pytest.mark.skipif(not SOURCE_DB.exists(), reason="ops database not present")


class FixedEvaluator(DamageEvaluator):
    name = "fixed"
    version = "test"
    is_mock = True

    def __init__(self, detections: list[RawDetection]) -> None:
        self.detections = detections

    def predict(
        self, image: Image.Image, min_confidence: float | None = None
    ) -> list[RawDetection]:
        return self.detections


class FixedPlateRecognizer(PlateRecognizer):
    name = "fixed-plate"
    is_mock = True

    def __init__(self, text: str | None) -> None:
        self.text = text
        self.calls = 0

    def read(self, image: Image.Image) -> list[RawPlate]:
        self.calls += 1
        return [RawPlate(self.text, 0.9, (10.0, 10.0, 110.0, 40.0))] if self.text else []


TWO_DAMAGES = [
    RawDetection(DamageClass.SCRATCH, 0.91, (100.0, 50.0, 300.0, 150.0)),
    RawDetection(DamageClass.DENT, 0.72, (320.0, 200.0, 480.0, 360.0)),
]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "ops.sqlite"
    shutil.copy(SOURCE_DB, path)
    VehicleRepository(path).ensure_schema()  # as build_vehicle_repository does at startup
    return path


def make_client(
    db: Path | None,
    detections: list[RawDetection] = TWO_DAMAGES,
    ocr_text: str | None = None,
) -> tuple[TestClient, FixedPlateRecognizer]:
    app = create_app()
    recognizer = FixedPlateRecognizer(ocr_text)
    app.dependency_overrides[get_evaluator] = lambda: FixedEvaluator(detections)
    app.dependency_overrides[get_plate_recognizer] = lambda: recognizer
    app.dependency_overrides[get_tire_detector] = MockTireDetector
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    repo = VehicleRepository(db) if db else None
    app.dependency_overrides[get_vehicle_repository] = lambda: repo
    return TestClient(app), recognizer


def rows(db: Path, sql: str, *args: object) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def test_known_plate_writes_annotations_alert_and_latest_anomaly(
    db: Path, sample_jpeg: bytes
) -> None:
    client, recognizer = make_client(db)
    with client:
        resp = client.post(
            DAMAGE,
            files={"file": ("car.jpg", sample_jpeg, "image/jpeg")},
            data={"plate_number": "rac4582", "image_side": "right"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert recognizer.calls == 0  # the client's plate is used; no OCR
    assert body["vehicle"]["id"] == 1
    assert body["vehicle"]["license_plate"] == KNOWN_PLATE
    assert body["vehicle"]["plate_source"] == "request"

    record = body["db_record"]
    assert record["case_id"] == body["request_id"]
    assert record["anomaly_text"] == "右側 刮傷、凹陷"

    annotations = rows(
        db,
        "SELECT id, plate_number, image_side, category, x, y, width, height, yolo_coordinates "
        "FROM damage_annotations WHERE case_id = ? ORDER BY id",
        body["request_id"],
    )
    assert [a[0] for a in annotations] == record["annotation_ids"]
    first = annotations[0]
    assert first[1:8] == (KNOWN_PLATE, "right", "scratch", 100.0, 50.0, 200.0, 100.0)
    # 640x480 image: centre (200, 100) -> (0.3125, 0.2083), size (0.3125, 0.2083); scratch = id 0
    assert first[8] == "0 0.312500 0.208333 0.312500 0.208333"

    assert (record["vehicle_id"], record["plate_number"]) == (1, KNOWN_PLATE)
    alert = rows(
        db,
        "SELECT vehicle_id, plate_number, anomaly_type, confidence, status "
        "FROM ai_anomaly_alerts WHERE id = ?",
        record["alert_id"],
    )
    assert alert == [(1, KNOWN_PLATE, "右側 刮傷、凹陷", 91, "pending")]
    assert rows(db, "SELECT latest_anomaly FROM vehicles WHERE id = 1") == [("右側 刮傷、凹陷",)]


def test_plate_read_by_ocr_when_not_sent(db: Path, sample_jpeg: bytes) -> None:
    client, recognizer = make_client(db, ocr_text="RAC-4582")
    with client:
        body = client.post(DAMAGE, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")}).json()
    assert recognizer.calls == 1
    assert body["vehicle"]["plate_source"] == "ocr"
    assert body["db_record"]["anomaly_text"] == "刮傷、凹陷"  # side unknown: no side prefix
    assert rows(db, "SELECT image_side FROM damage_annotations WHERE case_id = ?",
                body["request_id"])[0] == ("unknown",)


@pytest.mark.parametrize(
    ("plate", "ocr_text", "stored_plate"),
    [("ZZZ-9999", None, "ZZZ-9999"), (None, "ZZZ-9999", "ZZZ-9999"), (None, None, None)],
    ids=["unregistered-plate", "unregistered-ocr", "no-plate-found"],
)
def test_unmatched_vehicle_still_writes_alert(
    db: Path, sample_jpeg: bytes, plate: str | None, ocr_text: str | None, stored_plate: str | None
) -> None:
    vehicles_before = rows(db, "SELECT * FROM vehicles ORDER BY id")
    client, _ = make_client(db, ocr_text=ocr_text)
    with client:
        data = {"plate_number": plate} if plate else {}
        files = {"file": ("car.jpg", sample_jpeg, "image/jpeg")}
        resp = client.post(DAMAGE, files=files, data=data)
    body = resp.json()
    assert resp.status_code == 200, resp.text
    assert body["summary"]["total_detections"] == 2
    assert body["vehicle"] is None

    record = body["db_record"]
    assert (record["vehicle_id"], record["plate_number"]) == (None, stored_plate)
    alert = rows(
        db,
        "SELECT vehicle_id, plate_number, anomaly_type, confidence, status "
        "FROM ai_anomaly_alerts WHERE id = ?",
        record["alert_id"],
    )
    assert alert == [(None, stored_plate, "刮傷、凹陷", 91, "pending")]
    annotations = rows(
        db, "SELECT id, plate_number FROM damage_annotations WHERE case_id = ? ORDER BY id",
        body["request_id"],
    )
    assert [a[0] for a in annotations] == record["annotation_ids"]
    assert {a[1] for a in annotations} == {stored_plate or ""}
    assert rows(db, "SELECT * FROM vehicles ORDER BY id") == vehicles_before


LEGACY_ALERTS_DDL = """
CREATE TABLE "vehicles" ("id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "license_plate" TEXT NOT NULL);
CREATE TABLE "ai_anomaly_alerts" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "vehicle_id" INTEGER NOT NULL,
  "anomaly_type" TEXT NOT NULL,
  "confidence" INTEGER NOT NULL CHECK ("confidence" BETWEEN 0 AND 100),
  "status" TEXT NOT NULL DEFAULT 'pending' CHECK ("status" IN ('pending', 'review', 'resolved')),
  "detected_at" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "created_at" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "ai_anomaly_alerts_vehicle_id_fkey"
    FOREIGN KEY ("vehicle_id") REFERENCES "vehicles" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO vehicles (id, license_plate) VALUES (1, 'RAC-4582'), (2, 'RAA-0001');
INSERT INTO ai_anomaly_alerts (id, vehicle_id, anomaly_type, confidence, status)
  VALUES (1, 1, 'a', 96, 'pending'), (2, 2, 'b', 92, 'review');
"""


def test_alerts_migration_keeps_rows_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_ALERTS_DDL)
    conn.close()

    repo = VehicleRepository(path)
    repo.ensure_schema()
    repo.ensure_schema()  # second run is a no-op

    assert rows(
        path, "SELECT id, vehicle_id, plate_number, anomaly_type, confidence, status "
        "FROM ai_anomaly_alerts ORDER BY id"
    ) == [(1, 1, "RAC-4582", "a", 96, "pending"), (2, 2, "RAA-0001", "b", 92, "review")]
    columns = {r[1]: r[3] for r in rows(path, "PRAGMA table_info(ai_anomaly_alerts)")}
    assert columns["vehicle_id"] == 0  # notnull flag cleared
    assert rows(path, "PRAGMA foreign_key_list(ai_anomaly_alerts)")[0][2] == "vehicles"
    assert {r[0] for r in rows(path, "SELECT name FROM sqlite_master WHERE type = 'index'")} >= {
        "ai_anomaly_alerts_vehicle_id_status_detected_at_idx",
        "ai_anomaly_alerts_plate_number_detected_at_idx",
    }


def test_no_damage_links_vehicle_but_writes_nothing(db: Path, sample_jpeg: bytes) -> None:
    latest_before = rows(db, "SELECT latest_anomaly FROM vehicles WHERE id = 1")
    annotations_before = rows(db, "SELECT COUNT(*) FROM damage_annotations")
    client, _ = make_client(db, detections=[])
    with client:
        body = client.post(
            DAMAGE,
            files={"file": ("car.jpg", sample_jpeg, "image/jpeg")},
            data={"plate_number": KNOWN_PLATE},
        ).json()
    assert body["vehicle"]["id"] == 1
    assert body["db_record"] is None
    assert rows(db, "SELECT COUNT(*) FROM damage_annotations") == annotations_before
    assert rows(db, "SELECT latest_anomaly FROM vehicles WHERE id = 1") == latest_before


def test_without_database_nothing_is_looked_up(sample_jpeg: bytes) -> None:
    client, recognizer = make_client(None, ocr_text=KNOWN_PLATE)
    with client:
        body = client.post(DAMAGE, files={"file": ("car.jpg", sample_jpeg, "image/jpeg")}).json()
    assert recognizer.calls == 0  # no DB, so no extra OCR pass
    assert body["vehicle"] is None
    assert body["db_record"] is None


def test_tire_links_vehicle_read_only(db: Path, sample_jpeg: bytes) -> None:
    annotations_before = rows(db, "SELECT COUNT(*) FROM damage_annotations")
    client, _ = make_client(db)
    with client:
        body = client.post(
            TIRE,
            files={"file": ("car.jpg", sample_jpeg, "image/jpeg")},
            data={"plate_number": KNOWN_PLATE},
        ).json()
    assert body["vehicle"]["license_plate"] == KNOWN_PLATE
    assert rows(db, "SELECT COUNT(*) FROM damage_annotations") == annotations_before
