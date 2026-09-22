"""License-plate recognition endpoint."""

from __future__ import annotations

import time
import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_plate_recognizer, get_settings
from app.config import Settings
from app.schemas.errors import ErrorResponse
from app.schemas.plate import PlateRecognitionResponse
from app.services.image_io import load_upload
from app.services.plate_recognizer import PlateRecognizer, rectify_plate, resolve_plates

router = APIRouter(prefix="/plate", tags=["plate"])

_ERROR_RESPONSES = {
    413: {"model": ErrorResponse, "description": "Image exceeds the maximum allowed size"},
    415: {"model": ErrorResponse, "description": "Unsupported image content type"},
    422: {"model": ErrorResponse, "description": "Missing or unreadable image"},
}


@router.post(
    "/recognize",
    response_model=PlateRecognitionResponse,
    responses=_ERROR_RESPONSES,
    summary="Read the license plate number from a vehicle photo",
)
async def recognize_plate(
    file: UploadFile = File(..., description="A JPEG, PNG or WebP photo showing the plate."),
    recognizer: PlateRecognizer = Depends(get_plate_recognizer),
    settings: Settings = Depends(get_settings),
) -> PlateRecognitionResponse:
    image, metadata = await load_upload(file, settings)

    started = time.perf_counter()
    raws = await to_thread.run_sync(recognizer.read, image)
    inference_ms = (time.perf_counter() - started) * 1000.0

    plates, best = resolve_plates(raws, image.size, settings.plate_min_confidence)

    return PlateRecognitionResponse(
        request_id=str(uuid.uuid4()),
        image=metadata,
        plates=plates,
        best_plate=best,
        plate_image=rectify_plate(image, best.corners) if best and best.corners else None,
        model_name=recognizer.name,
        is_mock=recognizer.is_mock,
        inference_ms=round(inference_ms, 3),
    )
