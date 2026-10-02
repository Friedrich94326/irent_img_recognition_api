"""Schemas for the online pre-rental check (``POST /api/v1/rental/precheck``) and the corner
condition photos behind it (``/api/v1/vehicles/...``)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import Severity
from app.schemas.damage import DamageSummary, Detection
from app.schemas.vehicle import Corner, PhotoSource


class PrecheckRequest(BaseModel):
    plate_number: str = Field(min_length=1, description="Plate of the car, e.g. 'RCW-9206'.")
    member_no: str | None = Field(default=None, description="Customer's member number.")


class Verdict(str, Enum):
    OK = "ok"
    WARNING = "warning"
    BLOCKED = "blocked"


class ReasonLevel(str, Enum):
    WARNING = "warning"
    BLOCKED = "blocked"


class PrecheckReason(BaseModel):
    code: str = Field(description="Stable machine-readable reason, e.g. 'vehicle_in_maintenance'.")
    level: ReasonLevel
    message: str = Field(description="Human-readable explanation for the driver.")


class PrecheckVehicle(BaseModel):
    id: int
    license_plate: str
    model: str
    color: str
    status: str = Field(description="vehicles.status: available / cleaning / maintenance.")
    cabin_condition: str = Field(description="vehicles.cabin_condition: clean / average / dirty.")
    station_name: str


class KnownAlert(BaseModel):
    id: int
    anomaly_type: str
    status: str
    detected_at: str


class KnownRepair(BaseModel):
    order_number: str
    maintenance_item: str
    status: str


class KnownIssues(BaseModel):
    latest_anomaly: str | None
    unresolved_alerts: list[KnownAlert]
    open_repair_orders: list[KnownRepair]


class CornerPhoto(BaseModel):
    """One stored condition photo and the damage detected in it when it was uploaded."""

    photo_id: int
    corner: Corner
    corner_label: str = Field(description="Chinese corner name, e.g. '左前'.")
    source: PhotoSource
    captured_at: str
    image_url: str = Field(description="Path of the JPEG, relative to the API base URL.")
    width: int
    height: int
    detections: list[Detection]
    summary: DamageSummary
    overall_severity: Severity
    model_name: str
    predicted_corner: Corner | None = Field(
        description="Corner the classifier sees in the image; null if it hasn't checked it."
    )
    corner_confidence: float | None = Field(ge=0.0, le=1.0)
    corner_mismatch: bool = Field(
        description="True when the classifier confidently sees a different corner than the "
        "photo is filed under."
    )


class VehicleSummary(BaseModel):
    id: int
    license_plate: str
    model: str
    status: str
    station_name: str
    photo_corners: int = Field(ge=0, le=4, description="Corners that have a current photo.")


class RentalPrecheckResponse(BaseModel):
    precheck_id: int = Field(description="rental_prechecks.id of the saved pre-check.")
    case_id: str = Field(description="rental_prechecks.case_id: quote it when contacting support.")
    verdict: Verdict = Field(description="Worst reason level, or 'ok' when there is none.")
    can_rent: bool = Field(description="False only when the verdict is 'blocked'.")
    reasons: list[PrecheckReason]
    vehicle: PrecheckVehicle
    known_issues: KnownIssues
    photos: list[CornerPhoto] = Field(description="Current photo of each corner that has one.")
    missing_corners: list[Corner]
    overall_severity: Severity = Field(description="Worst damage severity across all photos.")
