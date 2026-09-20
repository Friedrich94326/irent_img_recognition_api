"""Response schema for the license-plate recognition endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import BoundingBox, ImageMetadata


class PlateReading(BaseModel):
    """One license plate read from the image."""

    plate_number: str = Field(description="Normalised plate string, e.g. 'ABC-1234'.")
    raw_text: str = Field(description="Text exactly as the OCR engine returned it.")
    confidence: float = Field(ge=0.0, le=1.0)
    valid_format: bool = Field(
        description="True if the text matches a known Taiwan plate layout (e.g. ABC-1234)."
    )
    bounding_box: BoundingBox | None = None


class PlateRecognitionResponse(BaseModel):
    request_id: str
    image: ImageMetadata
    plates: list[PlateReading]
    best_plate: PlateReading | None = Field(
        default=None, description="Highest-confidence valid plate, or null if none was found."
    )
    model_name: str
    is_mock: bool
    inference_ms: float
