"""Schemas that link API results to the iRent ops database."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ImageSide(str, Enum):
    """Which side of the vehicle the photo shows (stored in ``damage_annotations.image_side``)."""

    FRONT = "front"
    REAR = "rear"
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class PlateSource(str, Enum):
    REQUEST = "request"
    OCR = "ocr"


class VehicleLink(BaseModel):
    """The ``vehicles`` row this photo was matched to."""

    id: int
    license_plate: str
    model: str
    color: str
    status: str
    plate_source: PlateSource = Field(
        description="'request' if the client sent plate_number, 'ocr' if it was read from the "
        "photo."
    )


class DamageRecord(BaseModel):
    """What was written to the database for this evaluation."""

    case_id: str = Field(description="damage_annotations.case_id (equals the request_id).")
    annotation_ids: list[int] = Field(description="New damage_annotations row ids.")
    alert_id: int = Field(description="New ai_anomaly_alerts row id (status 'pending').")
    anomaly_text: str = Field(
        description="Text stored as the alert's anomaly_type and, when a vehicle matched, the "
        "vehicle's latest_anomaly."
    )
    vehicle_id: int | None = Field(
        description="ai_anomaly_alerts.vehicle_id; null when the photo matched no vehicle."
    )
    plate_number: str | None = Field(
        description="ai_anomaly_alerts.plate_number: the matched, sent or OCR-read plate; null "
        "when none was found."
    )
