"""Online pre-rental check: the car's status in the ops database plus its current corner photos
(``vehicle_photos``, inspected when they were uploaded), turned into a verdict and saved in
``rental_prechecks`` together with the ids of the photos the customer was shown.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.schemas.common import Severity
from app.schemas.precheck import KnownAlert, KnownRepair, PrecheckReason, ReasonLevel, Verdict
from app.schemas.vehicle import CORNER_LABELS_ZH, Corner
from app.services.vehicle_photos import VehiclePhoto

logger = logging.getLogger(__name__)

REPAIR_DONE = "維修完畢"

_DDL = """
CREATE TABLE IF NOT EXISTS "rental_prechecks" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "case_id" TEXT NOT NULL,
  "vehicle_id" INTEGER NOT NULL,
  "customer_id" INTEGER,
  "verdict" TEXT NOT NULL CHECK ("verdict" IN ('ok', 'warning', 'blocked')),
  "reasons" TEXT NOT NULL DEFAULT '[]',
  "photo_count" INTEGER NOT NULL DEFAULT 0 CHECK ("photo_count" >= 0),
  "damage_count" INTEGER NOT NULL DEFAULT 0 CHECK ("damage_count" >= 0),
  "overall_severity" TEXT NOT NULL DEFAULT 'none',
  "photo_ids" TEXT NOT NULL DEFAULT '[]',
  "created_at" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "rental_prechecks_vehicle_id_fkey"
    FOREIGN KEY ("vehicle_id") REFERENCES "vehicles" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT "rental_prechecks_customer_id_fkey"
    FOREIGN KEY ("customer_id") REFERENCES "customers" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "rental_prechecks_case_id_key"
  ON "rental_prechecks"("case_id");
CREATE INDEX IF NOT EXISTS "rental_prechecks_vehicle_id_created_at_idx"
  ON "rental_prechecks"("vehicle_id", "created_at");
"""


@dataclass(frozen=True)
class VehicleSnapshot:
    id: int
    license_plate: str
    model: str
    color: str
    status: str
    cabin_condition: str
    latest_anomaly: str | None
    station_name: str
    active_rentals: int
    unresolved_alerts: list[KnownAlert]
    open_repairs: list[KnownRepair]


def _reason(code: str, level: ReasonLevel, message: str) -> PrecheckReason:
    return PrecheckReason(code=code, level=level, message=message)


def decide(
    snapshot: VehicleSnapshot, photos: dict[Corner, VehiclePhoto], overall: Severity
) -> tuple[Verdict, list[PrecheckReason]]:
    """Apply the pre-check rules; the verdict is the worst reason level, else ``ok``."""

    blocked, warning = ReasonLevel.BLOCKED, ReasonLevel.WARNING
    reasons: list[PrecheckReason] = []

    if snapshot.status == "maintenance":
        reasons.append(_reason("vehicle_in_maintenance", blocked, "The car is in maintenance."))
    elif snapshot.status == "cleaning":
        reasons.append(_reason("vehicle_in_cleaning", blocked, "The car is being cleaned."))
    if snapshot.active_rentals:
        reasons.append(
            _reason("vehicle_already_rented", blocked, "The car is in an active rental.")
        )
    if overall == Severity.SEVERE:
        reasons.append(
            _reason("severe_damage_detected", blocked, "Severe damage is visible in the photos.")
        )

    if snapshot.open_repairs:
        items = "、".join(r.maintenance_item for r in snapshot.open_repairs)
        reasons.append(
            _reason("open_repair_order", warning, f"Open repair order(s): {items}.")
        )
    if snapshot.unresolved_alerts:
        kinds = "、".join(a.anomaly_type for a in snapshot.unresolved_alerts)
        reasons.append(
            _reason("unresolved_anomaly", warning, f"Unresolved anomaly alert(s): {kinds}.")
        )
    if snapshot.cabin_condition == "dirty":
        reasons.append(_reason("cabin_dirty", warning, "The cabin is reported dirty."))
    damage = sum(len(p.detections) for p in photos.values())
    if damage and overall != Severity.SEVERE:
        reasons.append(
            _reason(
                "damage_detected",
                warning,
                f"{damage} damage finding(s) in the current photos; they are on record as "
                "pre-existing.",
            )
        )
    for corner, photo in photos.items():
        if photo.corner_mismatch and photo.predicted_corner is not None:
            reasons.append(
                _reason(
                    "corner_mismatch",
                    warning,
                    f"The {CORNER_LABELS_ZH[corner]} photo looks like "
                    f"{CORNER_LABELS_ZH[photo.predicted_corner]}; it may be filed under the "
                    "wrong corner.",
                )
            )
    missing = [c for c in Corner if c not in photos]
    if missing:
        corners = "、".join(CORNER_LABELS_ZH[c] for c in missing)
        reasons.append(
            _reason("photos_missing", warning, f"No current photo of: {corners}.")
        )

    if any(r.level == blocked for r in reasons):
        return Verdict.BLOCKED, reasons
    if reasons:
        return Verdict.WARNING, reasons
    return Verdict.OK, reasons


class PrecheckRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def ensure_table(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_DDL)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(rental_prechecks)")}
            if "photo_ids" not in columns:  # table created by an earlier version
                conn.execute(
                    "ALTER TABLE rental_prechecks ADD COLUMN photo_ids TEXT NOT NULL DEFAULT '[]'"
                )
                conn.commit()
        finally:
            conn.close()

    def vehicle_snapshot(self, license_plate: str) -> VehicleSnapshot | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT v.id, v.license_plate, v.model, v.color, v.status, v.cabin_condition, "
                "v.latest_anomaly, s.name AS station_name FROM vehicles v "
                "JOIN stations s ON s.id = v.station_id WHERE v.license_plate = ?",
                (license_plate,),
            ).fetchone()
            if row is None:
                return None
            active = conn.execute(
                "SELECT COUNT(*) FROM rentals WHERE vehicle_id = ? AND status = 'active'",
                (row["id"],),
            ).fetchone()[0]
            alerts = conn.execute(
                "SELECT id, anomaly_type, status, detected_at FROM ai_anomaly_alerts "
                "WHERE vehicle_id = ? AND status != 'resolved' ORDER BY detected_at DESC",
                (row["id"],),
            ).fetchall()
            repairs = conn.execute(
                "SELECT order_number, maintenance_item, status FROM repair_orders "
                "WHERE vehicle_license_plate = ? AND status != ? ORDER BY created_at DESC",
                (license_plate, REPAIR_DONE),
            ).fetchall()
        finally:
            conn.close()
        return VehicleSnapshot(
            **dict(row),
            active_rentals=active,
            unresolved_alerts=[KnownAlert(**dict(a)) for a in alerts],
            open_repairs=[KnownRepair(**dict(r)) for r in repairs],
        )

    def find_customer(self, member_no: str) -> int | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM customers WHERE member_no = ?", (member_no,)
            ).fetchone()
        finally:
            conn.close()
        return row["id"] if row else None

    def record(
        self,
        case_id: str,
        snapshot: VehicleSnapshot,
        customer_id: int | None,
        verdict: Verdict,
        reasons: list[PrecheckReason],
        photos: dict[Corner, VehiclePhoto],
        overall: Severity,
    ) -> int:
        """Save the pre-check with the ids of the photos the customer was shown."""

        conn = self._connect()
        try:
            with conn:
                return conn.execute(
                    "INSERT INTO rental_prechecks (case_id, vehicle_id, customer_id, verdict, "
                    "reasons, photo_count, damage_count, overall_severity, photo_ids) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        case_id,
                        snapshot.id,
                        customer_id,
                        verdict.value,
                        json.dumps([r.model_dump(mode="json") for r in reasons],
                                   ensure_ascii=False),
                        len(photos),
                        sum(len(p.detections) for p in photos.values()),
                        overall.value,
                        json.dumps(sorted(p.id for p in photos.values())),
                    ),
                ).lastrowid
        finally:
            conn.close()


def build_precheck_repository(settings: Settings) -> PrecheckRepository | None:
    path = settings.db_path
    if path is None or not path.exists():
        logger.info("No iRent ops database available - rental pre-checks are unavailable.")
        return None
    repo = PrecheckRepository(path)
    repo.ensure_table()
    return repo
