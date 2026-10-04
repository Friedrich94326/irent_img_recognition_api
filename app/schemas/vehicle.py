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


class Corner(str, Enum):
    """Corner of the car a condition photo shows (``vehicle_photos.corner``)."""

    FRONT_LEFT = "front_left"  # 左前
    FRONT_RIGHT = "front_right"  # 右前
    REAR_LEFT = "rear_left"  # 左後
    REAR_RIGHT = "rear_right"  # 右後


CORNER_LABELS_ZH: dict[Corner, str] = {
    Corner.FRONT_LEFT: "左前",
    Corner.FRONT_RIGHT: "右前",
    Corner.REAR_LEFT: "左後",
    Corner.REAR_RIGHT: "右後",
}


class PhotoSource(str, Enum):
    """Who took a condition photo (``vehicle_photos.source``)."""

    RETURN = "return"  # the previous renter's return inspection (還)
    PICKUP = "pickup"  # a renter's pickup inspection (借)
    STATION = "station"  # a station camera or staff


class PlateSource(str, Enum):
    REQUEST = "request"
    FILENAME = "filename"


class VehicleLink(BaseModel):
    """The ``vehicles`` row this photo was matched to."""

    id: int
    license_plate: str
    model: str
    color: str
    status: str
    plate_source: PlateSource = Field(
        description="'request' if the client sent plate_number, 'filename' if it was taken from "
        "the uploaded file's name."
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
        description="ai_anomaly_alerts.plate_number: the matched, sent or file-name plate; null "
        "when none was found."
    )
