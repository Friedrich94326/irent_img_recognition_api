"""Response schema for the tire detection endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import BoundingBox, ImageMetadata
from app.schemas.vehicle import VehicleLink


class TireEllipse(BaseModel):
    """Axis-aligned ellipse in pixel coordinates: centre plus horizontal / vertical semi-axes."""

    cx: float = Field(ge=0, description="Centre x, pixels.")
    cy: float = Field(ge=0, description="Centre y, pixels.")
    rx: float = Field(gt=0, description="Horizontal semi-axis, pixels.")
    ry: float = Field(gt=0, description="Vertical semi-axis, pixels.")


class TireDetection(BaseModel):
    """One tire (the visible wheel: tire plus rim) found in the image."""

    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox
    ellipse: TireEllipse = Field(
        description=(
            "Tire outline: the ellipse inscribed in bounding_box. A tire seen at an angle "
            "appears as an ellipse, and as a circle when it faces the camera."
        )
    )


class TireDetectionResponse(BaseModel):
    request_id: str
    image: ImageMetadata
    tires: list[TireDetection] = Field(description="Every tire found, most confident first.")
    tire_count: int = Field(ge=0)
    model_name: str
    is_mock: bool
    inference_ms: float
    vehicle: VehicleLink | None = Field(
        default=None,
        description="Matched vehicles row, or null (no database, no plate, or plate not "
        "registered).",
    )
