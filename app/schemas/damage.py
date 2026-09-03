"""Schemas for the damage-evaluation endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import BoundingBox, DamageClass, ImageMetadata, Severity


class Detection(BaseModel):
    """A single detected area of damage."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "damage_class": "dent",
                "confidence": 0.87,
                "severity": "moderate",
                "bounding_box": {"x1": 120.0, "y1": 80.5, "x2": 340.0, "y2": 220.0},
            }
        },
    )

    damage_class: DamageClass = Field(description="Predicted damage category.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence for this detection.")
    severity: Severity = Field(description="Per-detection severity from class and confidence.")
    bounding_box: BoundingBox = Field(description="Location of the damage within the image.")


class DamageSummary(BaseModel):
    """Aggregate view over all detections."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "total_detections": 2,
                "counts_by_class": {"dent": 1, "scratch": 1},
                "max_confidence": 0.87,
            }
        },
    )

    total_detections: int = Field(ge=0)
    counts_by_class: dict[DamageClass, int] = Field(
        default_factory=dict,
        description="Number of detections per damage class (classes with zero detections omitted).",
    )
    max_confidence: float | None = Field(
        default=None,
        description="Highest confidence across all detections, or null when there are none.",
    )


class DamageEvaluationResponse(BaseModel):
    """Full result returned by ``POST /api/v1/damage/evaluate``."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "request_id": "0f8b3c2e-1c4a-4b6e-9a1d-2f5c9d3e7a11",
                "image": {
                    "filename": "front_bumper.jpg",
                    "content_type": "image/jpeg",
                    "width": 1280,
                    "height": 720,
                    "size_bytes": 245678,
                },
                "detections": [
                    {
                        "damage_class": "dent",
                        "confidence": 0.87,
                        "severity": "moderate",
                        "bounding_box": {"x1": 120.0, "y1": 80.5, "x2": 340.0, "y2": 220.0},
                    }
                ],
                "summary": {
                    "total_detections": 1,
                    "counts_by_class": {"dent": 1},
                    "max_confidence": 0.87,
                },
                "overall_severity": "moderate",
                "model_name": "mock-detector",
                "model_version": "0.1.0",
                "is_mock": True,
                "inference_ms": 3.4,
            }
        }
    )

    request_id: str = Field(description="Unique id for this evaluation request.")
    image: ImageMetadata
    detections: list[Detection]
    summary: DamageSummary
    overall_severity: Severity = Field(description="Overall vehicle damage severity.")
    model_name: str = Field(description="Identifier of the detector that produced this result.")
    model_version: str
    is_mock: bool = Field(description="True when results came from the built-in mock detector.")
    inference_ms: float = Field(ge=0.0, description="Wall-clock inference time in milliseconds.")
