"""Online pre-rental check (``POST /api/v1/rental/precheck``) on a throwaway copy of the ops DB,
with corner photos seeded straight into the photo store."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import (
    get_precheck_repository,
    get_settings,
    get_vehicle_photo_repository,
)
from app.config import Settings
from app.main import create_app
from app.schemas.common import BoundingBox, DamageClass, Severity
from app.schemas.damage import Detection
from app.schemas.precheck import Verdict
from app.schemas.vehicle import Corner, PhotoSource
from app.services.corner_classifier import CornerPrediction
from app.services.evaluator import RawDetection
from app.services.precheck import PrecheckRepository, VehicleSnapshot, decide
from app.services.severity import derive_overall_severity, enrich, summarize
from app.services.vehicle_photos import PhotoInspectionResult, VehiclePhotoRepository

SOURCE_DB = Path("data/irent_op_backend.sqlite")
PRECHECK = "/api/v1/rental/precheck"
# Seed data: available, clean, no active rental / open repair / unresolved alert.
CLEAN_CAR = "RMU-5268"
IN_MAINTENANCE = "RAC-4582"
ACTIVE_RENTAL = "RPW-7095"
WITH_ALERT = "RBG-1357"  # available, one unresolved ai_anomaly_alerts row
WITH_REPAIR = "RBA-6935"  # available, one open repair order

pytestmark = pytest.mark.skipif(not SOURCE_DB.exists(), reason="ops database not present")


def detection(cls: DamageClass, conf: float) -> Detection:
    [det] = enrich([RawDetection(cls, conf, (100.0, 50.0, 300.0, 150.0))])
    return det


SCRATCH = detection(DamageClass.SCRATCH, 0.6)
SHATTER = detection(DamageClass.GLASS_SHATTER, 0.95)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "ops.sqlite"
    shutil.copy(SOURCE_DB, path)
    PrecheckRepository(path).ensure_table()
    return path


@pytest.fixture
def photos(db: Path, tmp_path: Path) -> VehiclePhotoRepository:
    repo = VehiclePhotoRepository(db, tmp_path / "photos")
    repo.ensure_schema()
    return repo


def add(
    repo: VehiclePhotoRepository,
    plate: str,
    corner: Corner,
    detections: list[Detection] | None = None,
    captured_at: str = "2026-09-01 10:00:00",
) -> int:
    dets = detections or []
    inspection = PhotoInspectionResult(dets, summarize(dets), derive_overall_severity(dets))
    vehicle_id = repo.find_vehicle_id(plate)
    image = Image.new("RGB", (640, 480), (120, 120, 120))
    return repo.add_photo(
        vehicle_id, corner, PhotoSource.RETURN, image, inspection, "fixed", captured_at
    ).id


def add_all(repo: VehiclePhotoRepository, plate: str, **dets: list[Detection]) -> list[int]:
    return [add(repo, plate, c, dets.get(c.value)) for c in Corner]


def make_client(db: Path | None, photos: VehiclePhotoRepository | None = None) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    repo = PrecheckRepository(db) if db else None
    app.dependency_overrides[get_precheck_repository] = lambda: repo
    app.dependency_overrides[get_vehicle_photo_repository] = lambda: photos
    return TestClient(app)


def rows(db: Path, sql: str, *args: object) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def check(db: Path, photos: VehiclePhotoRepository, plate: str, **extra: str) -> dict:
    with make_client(db, photos) as client:
        resp = client.post(PRECHECK, json={"plate_number": plate, **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


def codes(body: dict) -> set[str]:
    return {r["code"] for r in body["reasons"]}


def test_clean_car_with_all_corners_is_ok_and_saved(
    db: Path, photos: VehiclePhotoRepository
) -> None:
    ids = add_all(photos, CLEAN_CAR)
    body = check(db, photos, CLEAN_CAR.lower().replace("-", ""), member_no="MEM0001")

    assert body["verdict"] == "ok"
    assert body["can_rent"] is True
    assert body["reasons"] == []
    assert body["missing_corners"] == []
    assert [p["corner"] for p in body["photos"]] == [c.value for c in Corner]
    first = body["photos"][0]
    assert first["corner_label"] == "左前"
    assert first["image_url"] == f"/api/v1/vehicle-photos/{first['photo_id']}/image"
    saved = rows(
        db,
        "SELECT p.verdict, p.photo_count, p.damage_count, p.photo_ids, c.member_no "
        "FROM rental_prechecks p JOIN customers c ON c.id = p.customer_id WHERE p.id = ?",
        body["precheck_id"],
    )
    assert saved == [("ok", 4, 0, json.dumps(sorted(ids)), "MEM0001")]


def test_latest_photo_per_corner_is_used(db: Path, photos: VehiclePhotoRepository) -> None:
    add_all(photos, CLEAN_CAR)
    newer = add(photos, CLEAN_CAR, Corner.REAR_LEFT, [SCRATCH], captured_at="2026-09-20 10:00:00")
    add(photos, CLEAN_CAR, Corner.REAR_LEFT, [SHATTER], captured_at="2026-08-01 10:00:00")

    body = check(db, photos, CLEAN_CAR)
    rear_left = next(p for p in body["photos"] if p["corner"] == "rear_left")
    assert rear_left["photo_id"] == newer
    assert rear_left["summary"]["total_detections"] == 1
    assert body["verdict"] == "warning"
    assert codes(body) == {"damage_detected"}


def test_precheck_reads_stored_damage_and_writes_no_annotations(
    db: Path, photos: VehiclePhotoRepository
) -> None:
    annotations = rows(db, "SELECT COUNT(*) FROM damage_annotations")
    alerts = rows(db, "SELECT COUNT(*) FROM ai_anomaly_alerts")
    add_all(photos, CLEAN_CAR, front_right=[SHATTER])

    body = check(db, photos, CLEAN_CAR)
    assert body["verdict"] == "blocked"
    assert "severe_damage_detected" in codes(body)
    assert body["overall_severity"] == "severe"
    assert rows(db, "SELECT COUNT(*) FROM damage_annotations") == annotations
    assert rows(db, "SELECT COUNT(*) FROM ai_anomaly_alerts") == alerts


def test_missing_corners_warn(db: Path, photos: VehiclePhotoRepository) -> None:
    add(photos, CLEAN_CAR, Corner.FRONT_LEFT)
    add(photos, CLEAN_CAR, Corner.FRONT_RIGHT)

    body = check(db, photos, CLEAN_CAR)
    assert body["verdict"] == "warning"
    assert body["missing_corners"] == ["rear_left", "rear_right"]
    [reason] = body["reasons"]
    assert reason["code"] == "photos_missing"
    assert "左後" in reason["message"] and "右後" in reason["message"]


def test_no_photos_at_all(db: Path, photos: VehiclePhotoRepository) -> None:
    body = check(db, photos, CLEAN_CAR)
    assert body["photos"] == []
    assert len(body["missing_corners"]) == 4
    assert codes(body) == {"photos_missing"}


@pytest.mark.parametrize(
    ("plate", "expected"),
    [(IN_MAINTENANCE, "vehicle_in_maintenance"), (ACTIVE_RENTAL, "vehicle_already_rented")],
)
def test_blocked_by_status(
    db: Path, photos: VehiclePhotoRepository, plate: str, expected: str
) -> None:
    add_all(photos, plate)
    body = check(db, photos, plate)
    assert body["verdict"] == "blocked"
    assert body["can_rent"] is False
    assert expected in codes(body)


def test_known_issues_warn(db: Path, photos: VehiclePhotoRepository) -> None:
    add_all(photos, WITH_ALERT)
    add_all(photos, WITH_REPAIR)
    assert codes(check(db, photos, WITH_ALERT)) == {"unresolved_anomaly"}
    repair = check(db, photos, WITH_REPAIR)
    assert codes(repair) == {"open_repair_order"}
    assert len(repair["known_issues"]["open_repair_orders"]) == 1


@pytest.mark.parametrize(
    ("body", "field"),
    [({"plate_number": "ZZZ-9999"}, "plate_number"),
     ({"plate_number": CLEAN_CAR, "member_no": "NOPE"}, "member_no")],
)
def test_unknown_vehicle_or_customer_is_404(
    db: Path, photos: VehiclePhotoRepository, body: dict, field: str
) -> None:
    before = rows(db, "SELECT COUNT(*) FROM rental_prechecks")
    with make_client(db, photos) as client:
        resp = client.post(PRECHECK, json=body)
    assert resp.status_code == 404
    assert resp.json()["error"]["field"] == field
    assert rows(db, "SELECT COUNT(*) FROM rental_prechecks") == before


def test_without_database_is_503() -> None:
    with make_client(None) as client:
        assert client.post(PRECHECK, json={"plate_number": CLEAN_CAR}).status_code == 503


def test_missing_plate_is_422(db: Path, photos: VehiclePhotoRepository) -> None:
    with make_client(db, photos) as client:
        assert client.post(PRECHECK, json={}).status_code == 422
        assert client.post(PRECHECK, json={"plate_number": ""}).status_code == 422


def test_old_table_gets_photo_ids_column(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        'CREATE TABLE "rental_prechecks" ("id" INTEGER PRIMARY KEY, "case_id" TEXT NOT NULL)'
    )
    conn.close()
    PrecheckRepository(path).ensure_table()
    PrecheckRepository(path).ensure_table()
    assert "photo_ids" in {r[1] for r in rows(path, "PRAGMA table_info(rental_prechecks)")}


def _snapshot(**overrides: object) -> VehicleSnapshot:
    base = dict(
        id=1, license_plate="AAA-0001", model="m", color="c", status="available",
        cabin_condition="clean", latest_anomaly=None, station_name="s", active_rentals=0,
        unresolved_alerts=[], open_repairs=[],
    )
    return VehicleSnapshot(**{**base, **overrides})


def test_decide_dirty_cabin_warns_and_blocked_wins() -> None:
    verdict, reasons = decide(_snapshot(cabin_condition="dirty"), {}, Severity.NONE)
    assert verdict == Verdict.WARNING
    assert {r.code for r in reasons} == {"cabin_dirty", "photos_missing"}

    verdict, reasons = decide(
        _snapshot(status="cleaning", cabin_condition="dirty"), {}, Severity.NONE
    )
    assert verdict == Verdict.BLOCKED
    assert {r.code for r in reasons} == {"vehicle_in_cleaning", "cabin_dirty", "photos_missing"}


def test_corner_mismatch_warns(db: Path, photos: VehiclePhotoRepository) -> None:
    ids = add_all(photos, CLEAN_CAR)
    photos.set_corner_check(ids[0], CornerPrediction(Corner.REAR_RIGHT, 0.91), True)  # front_left
    photos.set_corner_check(ids[1], CornerPrediction(Corner.FRONT_RIGHT, 0.99), False)

    body = check(db, photos, CLEAN_CAR)
    assert body["verdict"] == "warning"
    [reason] = body["reasons"]
    assert reason["code"] == "corner_mismatch"
    assert "左前" in reason["message"] and "右後" in reason["message"]
    front_left, front_right = body["photos"][:2]
    assert (front_left["predicted_corner"], front_left["corner_mismatch"]) == ("rear_right", True)
    assert (front_right["corner_confidence"], front_right["corner_mismatch"]) == (0.99, False)


def test_bounding_box_roundtrip_through_store(db: Path, photos: VehiclePhotoRepository) -> None:
    add(photos, CLEAN_CAR, Corner.FRONT_LEFT, [SCRATCH])
    body = check(db, photos, CLEAN_CAR)
    box = body["photos"][0]["detections"][0]["bounding_box"]
    assert BoundingBox(**box) == SCRATCH.bounding_box
