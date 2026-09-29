"""Link API results to the iRent ops database (``vehicles``, ``damage_annotations``,
``ai_anomaly_alerts``).

The database is the back office's SQLite file. Every call opens its own short-lived connection,
so the repository is safe to use from the worker threads FastAPI routes hand blocking work to.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.schemas.common import DamageClass
from app.schemas.damage import Detection
from app.schemas.vehicle import ImageSide

logger = logging.getLogger(__name__)

# Chinese wording used by the back office's anomaly texts (e.g. '右後保桿刮傷').
DAMAGE_LABELS_ZH: dict[DamageClass, str] = {
    DamageClass.SCRATCH: "刮傷",
    DamageClass.DENT: "凹陷",
    DamageClass.CRACK: "裂痕",
    DamageClass.GLASS_SHATTER: "玻璃破裂",
    DamageClass.LAMP_BROKEN: "燈具破損",
    DamageClass.TIRE_FLAT: "輪胎失壓",
    DamageClass.MISSING_PART: "零件缺失",
    DamageClass.PAINT_CHIP: "掉漆",
}

SIDE_LABELS_ZH: dict[ImageSide, str] = {
    ImageSide.FRONT: "車頭",
    ImageSide.REAR: "車尾",
    ImageSide.LEFT: "左側",
    ImageSide.RIGHT: "右側",
    ImageSide.UNKNOWN: "",
}

# Stable class ids for the ``yolo_coordinates`` column: the enum's declaration order.
_CLASS_IDS: dict[DamageClass, int] = {cls: i for i, cls in enumerate(DamageClass)}


@dataclass(frozen=True)
class VehicleRecord:
    id: int
    license_plate: str
    model: str
    color: str
    status: str
    latest_anomaly: str | None


@dataclass(frozen=True)
class DamageRecordResult:
    annotation_ids: list[int]
    alert_id: int
    anomaly_text: str


def anomaly_text(side: ImageSide, detections: list[Detection]) -> str:
    """Summarise detections the way the back office words anomalies, e.g. '右側 刮傷、凹陷'."""

    labels = list(dict.fromkeys(DAMAGE_LABELS_ZH[d.damage_class] for d in detections))
    side_label = SIDE_LABELS_ZH[side]
    return f"{side_label} {'、'.join(labels)}".strip()


def yolo_coordinates(det: Detection, width: int, height: int) -> str:
    """``'<class id> <cx> <cy> <w> <h>'`` normalised to 0-1, as in a YOLO label file."""

    box = det.bounding_box
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    return (
        f"{_CLASS_IDS[det.damage_class]} "
        f"{cx:.6f} {cy:.6f} {box.width / width:.6f} {box.height / height:.6f}"
    )


class VehicleRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def find_vehicle(self, license_plate: str) -> VehicleRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, license_plate, model, color, status, latest_anomaly "
                "FROM vehicles WHERE license_plate = ?",
                (license_plate,),
            ).fetchone()
        finally:
            conn.close()
        return VehicleRecord(**dict(row)) if row else None

    def record_damage(
        self,
        vehicle: VehicleRecord,
        case_id: str,
        side: ImageSide,
        detections: list[Detection],
        image_size: tuple[int, int],
    ) -> DamageRecordResult:
        """Write one annotation per detection, one pending alert, and the vehicle's latest
        anomaly, all in a single transaction. ``detections`` must not be empty."""

        if not detections:
            raise ValueError("record_damage needs at least one detection")
        width, height = image_size
        text = anomaly_text(side, detections)
        confidence = round(max(d.confidence for d in detections) * 100)

        conn = self._connect()
        try:
            with conn:  # commits on success, rolls back on any error
                annotation_ids = [
                    conn.execute(
                        "INSERT INTO damage_annotations (case_id, plate_number, image_side, "
                        "category, x, y, width, height, yolo_coordinates) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            case_id,
                            vehicle.license_plate,
                            side.value,
                            det.damage_class.value,
                            det.bounding_box.x1,
                            det.bounding_box.y1,
                            det.bounding_box.width,
                            det.bounding_box.height,
                            yolo_coordinates(det, width, height),
                        ),
                    ).lastrowid
                    for det in detections
                ]
                alert_id = conn.execute(
                    "INSERT INTO ai_anomaly_alerts (vehicle_id, anomaly_type, confidence) "
                    "VALUES (?, ?, ?)",
                    (vehicle.id, text, confidence),
                ).lastrowid
                conn.execute(
                    "UPDATE vehicles SET latest_anomaly = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    (text, vehicle.id),
                )
        finally:
            conn.close()
        return DamageRecordResult(annotation_ids, alert_id, text)


def build_vehicle_repository(settings: Settings) -> VehicleRepository | None:
    path = settings.db_path
    if path is None:
        logger.info("No iRent ops database configured - results are not linked to vehicles.")
        return None
    if not path.exists():
        logger.warning("iRent ops database not found at %s - results are not linked.", path)
        return None
    return VehicleRepository(path)
