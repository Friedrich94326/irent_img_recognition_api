"""Response schema for the repair-fee estimate endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import BoundingBox, DamageClass, ImageMetadata, Severity
from app.schemas.tire import TireDetection
from app.schemas.vehicle import VehicleLink


class MeasuredDamage(BaseModel):
    """One damage detection, sized against the nearest tire when one is in view."""

    damage_class: DamageClass
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox
    base_severity: Severity = Field(description="Severity from the class and confidence alone.")
    severity: Severity = Field(
        description="base_severity moved one level down for small damage, up for large damage; "
        "equal to base_severity when no tire gave a scale."
    )
    tire_share: float | None = Field(
        description="Box area as a share of the reference tire's area (1.0 = one whole tire); "
        "null without a tire."
    )
    area_cm2: float | None = Field(description="Approximate box area in cm²; null without a tire.")
    size_band: Literal["small", "medium", "large"] | None = Field(
        description="small < 5 % of a tire, large > 25 %; null without a tire."
    )
    tire_index: int | None = Field(
        description="Index into `tires` of the tire used as the ruler (the nearest one)."
    )


class DamagedArea(BaseModel):
    """Body damage merged into one area: overlapping boxes are counted once (flat tires
    excluded - their box is the wheel, not bodywork)."""

    union_px: float = Field(ge=0, description="Union of the body-damage boxes, px².")
    summed_px: float = Field(ge=0, description="Plain sum of the same boxes, px², for contrast.")
    union_share: float | None = Field(
        description="Union as a share of the reference tire's area; null when any damage had "
        "no tire to scale it."
    )
    summed_share: float | None
    union_cm2: float | None = Field(description="Union in approximate cm².")


class ImpactCluster(BaseModel):
    """Touching damage that looks like a collision rather than wear."""

    bounding_box: BoundingBox
    classes: list[DamageClass]
    reason: str


class FeeEstimateResponse(BaseModel):
    request_id: str
    image: ImageMetadata
    detections: list[MeasuredDamage]
    tires: list[TireDetection] = Field(description="Every tire found, most confident first.")
    tire_diameter_cm: float = Field(description="Real tire size the damage was scaled by.")
    damaged_area: DamagedArea
    impacts: list[ImpactCluster]
    overall_severity: Severity = Field(
        description="Worst size-adjusted damage (bumped when widespread); severe on any impact."
    )
    base_overall_severity: Severity = Field(description="overall_severity before impacts.")
    estimated_fee: int | None = Field(
        description="Estimated repair fee in NT$, rounded to NT$100; 0 without damage; null "
        "when no fee model is loaded."
    )
    currency: Literal["TWD"] = "TWD"
    fee_model_name: str | None = Field(description="Fee model used, or null when none is loaded.")
    model_name: str
    tire_model_name: str
    is_mock: bool = Field(description="True when the damage or tire detector is a mock.")
    inference_ms: float
    confidence_threshold: float
    vehicle: VehicleLink | None = Field(
        default=None,
        description="Matched vehicles row (read-only: nothing is written), or null.",
    )
