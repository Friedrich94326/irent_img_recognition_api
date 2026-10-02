"""Corner condition photos of each car (``vehicle_photos``), pushed by station cameras or the
return inspection and shown to customers in the online pre-rental check.

Damage detection runs once, when a photo is added; its result is stored with the photo so a
pre-check only reads it back. Image files live under ``Settings.vehicle_photo_dir`` as upright
(EXIF-corrected) JPEGs - the way the customer should see them - and box coordinates are in that
upright image's pixels.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.schemas.common import Severity
from app.schemas.damage import DamageSummary, Detection
from app.schemas.vehicle import Corner, PhotoSource
from app.services.corner_classifier import CornerPrediction
from app.services.evaluator import DamageEvaluator, RawDetection
from app.services.image_io import orient_landscape
from app.services.severity import derive_overall_severity, enrich, summarize

logger = logging.getLogger(__name__)

_JPEG_QUALITY = 90

_DDL = """
CREATE TABLE IF NOT EXISTS "vehicle_photos" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "vehicle_id" INTEGER NOT NULL,
  "corner" TEXT NOT NULL
    CHECK ("corner" IN ('front_left', 'front_right', 'rear_left', 'rear_right')),
  "source" TEXT NOT NULL DEFAULT 'station' CHECK ("source" IN ('return', 'pickup', 'station')),
  "file_path" TEXT NOT NULL,
  "width" INTEGER NOT NULL,
  "height" INTEGER NOT NULL,
  "detections" TEXT NOT NULL DEFAULT '[]',
  "damage_count" INTEGER NOT NULL DEFAULT 0 CHECK ("damage_count" >= 0),
  "overall_severity" TEXT NOT NULL DEFAULT 'none',
  "model_name" TEXT NOT NULL,
  "predicted_corner" TEXT,
  "corner_confidence" REAL,
  "corner_mismatch" INTEGER NOT NULL DEFAULT 0,
  "captured_at" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "created_at" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "vehicle_photos_vehicle_id_fkey"
    FOREIGN KEY ("vehicle_id") REFERENCES "vehicles" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "vehicle_photos_file_path_key" ON "vehicle_photos"("file_path");
CREATE INDEX IF NOT EXISTS "vehicle_photos_vehicle_id_corner_captured_at_idx"
  ON "vehicle_photos"("vehicle_id", "corner", "captured_at");
"""

# Corner check columns, added to tables created before the corner classifier existed.
_CORNER_COLUMNS = {
    "predicted_corner": "TEXT",
    "corner_confidence": "REAL",
    "corner_mismatch": "INTEGER NOT NULL DEFAULT 0",
}


@dataclass(frozen=True)
class PhotoInspectionResult:
    detections: list[Detection]
    summary: DamageSummary
    overall_severity: Severity


@dataclass(frozen=True)
class VehiclePhoto:
    id: int
    vehicle_id: int
    corner: Corner
    source: PhotoSource
    file_path: str
    width: int
    height: int
    detections: list[Detection]
    overall_severity: Severity
    model_name: str
    captured_at: str
    predicted_corner: Corner | None = None
    corner_confidence: float | None = None
    corner_mismatch: bool = False

    @property
    def summary(self) -> DamageSummary:
        return summarize(self.detections)


@dataclass(frozen=True)
class VehicleListing:
    id: int
    license_plate: str
    model: str
    status: str
    station_name: str
    photo_corners: int


def _unrotate(raw: RawDetection, upright_width: int) -> RawDetection:
    """Map a box found in the 90° counter-clockwise rotated copy back onto the upright photo."""

    x1, y1, x2, y2 = raw.xyxy
    return RawDetection(
        raw.damage_class, raw.confidence, (upright_width - y2, x1, upright_width - y1, x2)
    )


def inspect_photo(evaluator: DamageEvaluator, image: Image.Image) -> PhotoInspectionResult:
    """Run damage detection on one upright photo; boxes are returned in its coordinates.

    The detector expects the car lying horizontal, so a portrait photo is inspected as a landscape
    copy and the boxes are rotated back.
    """

    landscape = orient_landscape(image)
    raws = evaluator.predict(landscape)
    if landscape is not image:
        raws = [_unrotate(r, image.width) for r in raws]
    detections = enrich(raws)
    return PhotoInspectionResult(
        detections, summarize(detections), derive_overall_severity(detections)
    )


def _row_to_photo(row: sqlite3.Row) -> VehiclePhoto:
    return VehiclePhoto(
        id=row["id"],
        vehicle_id=row["vehicle_id"],
        corner=Corner(row["corner"]),
        source=PhotoSource(row["source"]),
        file_path=row["file_path"],
        width=row["width"],
        height=row["height"],
        detections=[Detection.model_validate(d) for d in json.loads(row["detections"])],
        overall_severity=Severity(row["overall_severity"]),
        model_name=row["model_name"],
        captured_at=row["captured_at"],
        predicted_corner=Corner(row["predicted_corner"]) if row["predicted_corner"] else None,
        corner_confidence=row["corner_confidence"],
        corner_mismatch=bool(row["corner_mismatch"]),
    )


class VehiclePhotoRepository:
    def __init__(self, db_path: Path, photo_dir: Path) -> None:
        self.db_path = db_path
        self.photo_dir = photo_dir

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_DDL)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(vehicle_photos)")}
            for name, kind in _CORNER_COLUMNS.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE vehicle_photos ADD COLUMN {name} {kind}")
            conn.commit()
        finally:
            conn.close()

    def find_vehicle_id(self, license_plate: str) -> int | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM vehicles WHERE license_plate = ?", (license_plate,)
            ).fetchone()
        finally:
            conn.close()
        return row["id"] if row else None

    def has_file(self, file_path: str) -> bool:
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT 1 FROM vehicle_photos WHERE file_path = ?", (file_path,)
            ).fetchone() is not None
        finally:
            conn.close()

    def add_photo(
        self,
        vehicle_id: int,
        corner: Corner,
        source: PhotoSource,
        image: Image.Image,
        inspection: PhotoInspectionResult,
        model_name: str,
        captured_at: str | None = None,
        file_name: str | None = None,
        corner_check: CornerPrediction | None = None,
        corner_mismatch: bool = False,
    ) -> VehiclePhoto:
        """Save ``image`` as a JPEG and record it with its inspection result and, when the corner
        classifier ran, its prediction (``corner_mismatch`` flags a confident disagreement).

        ``file_name`` gives a stable name (the Hotai import uses it to skip photos it already
        added); by default every photo gets a new random name.
        """

        rel_path = f"{vehicle_id}/{file_name or uuid.uuid4().hex}.jpg"
        target = self.photo_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(target, format="JPEG", quality=_JPEG_QUALITY)

        detections_json = json.dumps(
            [d.model_dump(mode="json") for d in inspection.detections], ensure_ascii=False
        )
        conn = self._connect()
        try:
            with conn:
                photo_id = conn.execute(
                    "INSERT INTO vehicle_photos (vehicle_id, corner, source, file_path, width, "
                    "height, detections, damage_count, overall_severity, model_name, captured_at, "
                    "predicted_corner, corner_confidence, corner_mismatch) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), "
                    "?, ?, ?)",
                    (
                        vehicle_id,
                        corner.value,
                        source.value,
                        rel_path,
                        image.width,
                        image.height,
                        detections_json,
                        len(inspection.detections),
                        inspection.overall_severity.value,
                        model_name,
                        captured_at,
                        corner_check.corner.value if corner_check else None,
                        corner_check.confidence if corner_check else None,
                        int(corner_mismatch),
                    ),
                ).lastrowid
        except Exception:
            target.unlink(missing_ok=True)  # keep files and rows in step
            raise
        finally:
            conn.close()
        photo = self.get(photo_id)
        assert photo is not None
        return photo

    def get(self, photo_id: int) -> VehiclePhoto | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM vehicle_photos WHERE id = ?", (photo_id,)).fetchone()
        finally:
            conn.close()
        return _row_to_photo(row) if row else None

    def set_inspection(
        self, photo_id: int, inspection: PhotoInspectionResult, model_name: str
    ) -> None:
        """Replace a stored photo's damage result, e.g. after a model or threshold change."""

        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "UPDATE vehicle_photos SET detections = ?, damage_count = ?, "
                    "overall_severity = ?, model_name = ? WHERE id = ?",
                    (
                        json.dumps([d.model_dump(mode="json") for d in inspection.detections],
                                   ensure_ascii=False),
                        len(inspection.detections),
                        inspection.overall_severity.value,
                        model_name,
                        photo_id,
                    ),
                )
        finally:
            conn.close()

    def set_corner_check(
        self, photo_id: int, corner_check: CornerPrediction, corner_mismatch: bool
    ) -> None:
        """Record the corner classifier's verdict on a photo stored before it ran."""

        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "UPDATE vehicle_photos SET predicted_corner = ?, corner_confidence = ?, "
                    "corner_mismatch = ? WHERE id = ?",
                    (corner_check.corner.value, corner_check.confidence, int(corner_mismatch),
                     photo_id),
                )
        finally:
            conn.close()

    def all_photos(self) -> list[VehiclePhoto]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM vehicle_photos ORDER BY id").fetchall()
        finally:
            conn.close()
        return [_row_to_photo(r) for r in rows]

    def file_for(self, photo: VehiclePhoto) -> Path:
        return self.photo_dir / photo.file_path

    def latest_by_corner(self, vehicle_id: int) -> dict[Corner, VehiclePhoto]:
        """The current photo of each corner: the latest captured, newest id on ties."""

        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM vehicle_photos p WHERE vehicle_id = ? AND id = ("
                "  SELECT id FROM vehicle_photos q WHERE q.vehicle_id = p.vehicle_id "
                "  AND q.corner = p.corner ORDER BY captured_at DESC, id DESC LIMIT 1)",
                (vehicle_id,),
            ).fetchall()
        finally:
            conn.close()
        return {Corner(r["corner"]): _row_to_photo(r) for r in rows}

    def list_vehicles(self) -> list[VehicleListing]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT v.id, v.license_plate, v.model, v.status, s.name AS station_name, "
                "(SELECT COUNT(DISTINCT corner) FROM vehicle_photos p WHERE p.vehicle_id = v.id) "
                "AS photo_corners FROM vehicles v JOIN stations s ON s.id = v.station_id "
                "ORDER BY photo_corners DESC, v.license_plate"
            ).fetchall()
        finally:
            conn.close()
        return [VehicleListing(**dict(r)) for r in rows]


def build_vehicle_photo_repository(settings: Settings) -> VehiclePhotoRepository | None:
    path = settings.db_path
    if path is None or not path.exists():
        logger.info("No iRent ops database available - vehicle photos are unavailable.")
        return None
    repo = VehiclePhotoRepository(path, settings.vehicle_photo_dir)
    repo.ensure_schema()
    return repo
