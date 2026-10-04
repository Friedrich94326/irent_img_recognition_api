"""Tire detection endpoint."""

from __future__ import annotations

import time
import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import (
    get_settings,
    get_tire_detector,
    get_vehicle_repository,
)
from app.config import Settings
from app.schemas.errors import ErrorResponse
from app.schemas.tire import TireDetectionResponse
from app.services.image_io import load_upload
from app.services.tire_detector import TireDetector, to_tire_detections
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
        description="Vehicle plate, e.g. 'RAC-4582'. If omitted, it is taken from the "
        "uploaded file's name (e.g. 'RDX-2376.jpg').",
    ),
    detector: TireDetector = Depends(get_tire_detector),
    repo: VehicleRepository | None = Depends(get_vehicle_repository),
    settings: Settings = Depends(get_settings),
) -> TireDetectionResponse:
    image, metadata = await load_upload(file, settings)

    started = time.perf_counter()
    raws = await to_thread.run_sync(detector.detect, image)
    inference_ms = (time.perf_counter() - started) * 1000.0

    tires = to_tire_detections(raws, image.size)

    # Read-only: tire results only identify the vehicle, they are not written to the database.
    match = await link_vehicle(plate_number, file.filename, repo)
    vehicle_link = None
    if match is not None and match.vehicle is not None:
        vehicle_link = to_vehicle_link(match.vehicle, match.source)

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
