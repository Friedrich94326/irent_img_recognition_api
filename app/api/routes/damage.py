"""Damage-evaluation endpoint."""

from __future__ import annotations

import time
import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import (
    get_evaluator,
    get_plate_recognizer,
    get_settings,
    get_vehicle_repository,
)
from app.config import Settings
from app.schemas.damage import DamageEvaluationResponse
from app.schemas.errors import ErrorResponse
from app.schemas.vehicle import DamageRecord, ImageSide
from app.services.evaluator import DamageEvaluator
from app.services.image_io import load_upload
from app.services.plate_recognizer import PlateRecognizer
from app.services.severity import derive_overall_severity, enrich, summarize
from app.services.vehicle_link import link_vehicle, to_vehicle_link
from app.services.vehicle_repository import VehicleRepository

router = APIRouter(prefix="/damage", tags=["damage"])

_ERROR_RESPONSES = {
    413: {"model": ErrorResponse, "description": "Image exceeds the maximum allowed size"},
    415: {"model": ErrorResponse, "description": "Unsupported image content type"},
    422: {"model": ErrorResponse, "description": "Missing or unreadable image"},
}


@router.post(
    "/evaluate",
    response_model=DamageEvaluationResponse,
    responses=_ERROR_RESPONSES,
    summary="Detect damage in a single vehicle photo",
)
async def evaluate_damage(
    file: UploadFile = File(..., description="A single JPEG, PNG or WebP image of the vehicle."),
    plate_number: str | None = Form(
        default=None,
        description="Vehicle plate, e.g. 'RAC-4582'. If omitted, it is read from the photo.",
    ),
    image_side: ImageSide = Form(
        default=ImageSide.UNKNOWN, description="Which side of the vehicle the photo shows."
    ),
    evaluator: DamageEvaluator = Depends(get_evaluator),
    recognizer: PlateRecognizer = Depends(get_plate_recognizer),
    repo: VehicleRepository | None = Depends(get_vehicle_repository),
    settings: Settings = Depends(get_settings),
) -> DamageEvaluationResponse:
    image, metadata = await load_upload(file, settings)

    started = time.perf_counter()
    raw = await to_thread.run_sync(evaluator.predict, image)
    inference_ms = (time.perf_counter() - started) * 1000.0

    detections = enrich(raw)
    summary = summarize(detections)
    overall = derive_overall_severity(detections)
    request_id = str(uuid.uuid4())

    vehicle_link = None
    db_record = None
    linked = await link_vehicle(image, plate_number, repo, recognizer, settings)
    if linked is not None:
        vehicle, source = linked
        vehicle_link = to_vehicle_link(vehicle, source)
        if detections:
            result = await to_thread.run_sync(
                repo.record_damage, vehicle, request_id, image_side, detections, image.size
            )
            db_record = DamageRecord(
                case_id=request_id,
                annotation_ids=result.annotation_ids,
                alert_id=result.alert_id,
                anomaly_text=result.anomaly_text,
            )

    return DamageEvaluationResponse(
        request_id=request_id,
        image=metadata,
        detections=detections,
        summary=summary,
        overall_severity=overall,
        model_name=evaluator.name,
        model_version=evaluator.version,
        is_mock=evaluator.is_mock,
        inference_ms=round(inference_ms, 3),
        vehicle=vehicle_link,
        db_record=db_record,
    )
