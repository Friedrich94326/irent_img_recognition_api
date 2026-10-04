"""Shared enums and value objects used across request/response schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DamageClass(str, Enum):
    """Vehicle damage categories the detector can report.

    This list mirrors a typical car-damage detection label set. When a real
    YOLOv8 model is trained, re-sync these values with the model's ``names`` map
    (see :class:`app.services.evaluator.YOLOv8Evaluator`).
    """

    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    LAMP_BROKEN = "lamp_broken"
    TIRE_FLAT = "tire_flat"
    MISSING_PART = "missing_part"
    PARTS_BROKEN = "parts_broken"  # a cracked / broken body part, e.g. bumper (Hotai 'parts_boken')
    PAINT_CHIP = "paint_chip"


class Severity(str, Enum):
    """Ordered severity levels. ``ORDER`` below encodes the ranking."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.NONE: 0,
    Severity.MINOR: 1,
    Severity.MODERATE: 2,
    Severity.SEVERE: 3,
}
"""Numeric rank for each :class:`Severity`, lowest (``none``) to highest (``severe``)."""


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (top-left origin)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"x1": 120.0, "y1": 80.5, "x2": 340.0, "y2": 220.0}},
    )

    x1: float = Field(ge=0, description="Left edge, pixels.")
    y1: float = Field(ge=0, description="Top edge, pixels.")
    x2: float = Field(ge=0, description="Right edge, pixels.")
    y2: float = Field(ge=0, description="Bottom edge, pixels.")

    @model_validator(mode="after")
    def _check_ordering(self) -> BoundingBox:
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class ImageMetadata(BaseModel):
    """Properties of the uploaded image, echoed back in the response."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "filename": "front_bumper.jpg",
                "content_type": "image/jpeg",
                "width": 1280,
                "height": 720,
                "size_bytes": 245678,
            }
        },
    )

    filename: str | None = Field(default=None, description="Original client filename, if provided.")
    content_type: str = Field(description="MIME type of the uploaded file.")
    width: int = Field(gt=0, description="Image width in pixels.")
    height: int = Field(gt=0, description="Image height in pixels.")
    size_bytes: int = Field(ge=0, description="Uploaded file size in bytes.")
