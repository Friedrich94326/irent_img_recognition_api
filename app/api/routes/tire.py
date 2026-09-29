"""Tire detection endpoint."""

from __future__ import annotations

import time
import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import (
    get_plate_recognizer,
    get_settings,
    get_tire_detector,
    get_vehicle_repository,
)
from app.config import Settings
from app.schemas.common import BoundingBox
from app.schemas.errors import ErrorResponse
from app.schemas.tire import TireDetection, TireDetectionResponse, TireEllipse
from app.services.image_io import load_upload
from app.services.plate_recognizer import PlateRecognizer
from app.services.tire_detector import TireDetector
from app.services.vehicle_link import link_vehicle, to_vehicle_link
from app.services.vehicle_repository import VehicleRepository

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
    plate_number: str | None = Form(
        default=None,
        description="Vehicle plate, e.g. 'RAC-4582'. If omitted, it is read from the photo.",
    ),
    detector: TireDetector = Depends(get_tire_detector),
    recognizer: PlateRecognizer = Depends(get_plate_recognizer),
    repo: VehicleRepository | None = Depends(get_vehicle_repository),
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

    # Read-only: tire results only identify the vehicle, they are not written to the database.
    linked = await link_vehicle(image, plate_number, repo, recognizer, settings)
    vehicle_link = None
    if linked is not None:
        vehicle, source = linked
        vehicle_link = to_vehicle_link(vehicle, source)

    return TireDetectionResponse(
        request_id=str(uuid.uuid4()),
        image=metadata,
        tires=tires,
        tire_count=len(tires),
        model_name=detector.name,
        is_mock=detector.is_mock,
        inference_ms=round(inference_ms, 3),
        vehicle=vehicle_link,
    )
