"""Tire detection endpoint."""

from __future__ import annotations

import time
import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_settings, get_tire_detector
from app.config import Settings
from app.schemas.common import BoundingBox
from app.schemas.errors import ErrorResponse
from app.schemas.tire import TireDetection, TireDetectionResponse, TireEllipse
from app.services.image_io import load_upload
from app.services.tire_detector import TireDetector

router = APIRouter(prefix="/tire", tags=["tire"])

_ERROR_RESPONSES = {
    413: {"model": ErrorResponse, "description": "Image exceeds the maximum allowed size"},
    415: {"model": ErrorResponse, "description": "Unsupported image content type"},
    422: {"model": ErrorResponse, "description": "Missing or unreadable image"},
}


@router.post(
    "/detect",
    response_model=TireDetectionResponse,
    responses=_ERROR_RESPONSES,
    summary="Find every tire in a vehicle photo",
)
async def detect_tires(
    file: UploadFile = File(..., description="A JPEG, PNG or WebP photo of the vehicle."),
    detector: TireDetector = Depends(get_tire_detector),
    settings: Settings = Depends(get_settings),
) -> TireDetectionResponse:
    image, metadata = await load_upload(file, settings)

    started = time.perf_counter()
    raws = await to_thread.run_sync(detector.detect, image)
    inference_ms = (time.perf_counter() - started) * 1000.0

    width, height = image.size
    tires: list[TireDetection] = []
    for raw in raws:
        x1, y1, x2, y2 = raw.xyxy
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue  # clipped away entirely: nothing left to show
        tires.append(
            TireDetection(
                confidence=max(0.0, min(raw.confidence, 1.0)),
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                # From the clipped box, so the outline never leaves the image.
                ellipse=TireEllipse(
                    cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, rx=(x2 - x1) / 2, ry=(y2 - y1) / 2
                ),
            )
        )

    return TireDetectionResponse(
        request_id=str(uuid.uuid4()),
        image=metadata,
        tires=tires,
        tire_count=len(tires),
        model_name=detector.name,
        is_mock=detector.is_mock,
        inference_ms=round(inference_ms, 3),
    )
