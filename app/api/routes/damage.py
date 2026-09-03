"""Damage-evaluation endpoint."""

from __future__ import annotations

import time
import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_evaluator, get_settings
from app.config import Settings
from app.schemas.damage import DamageEvaluationResponse
from app.schemas.errors import ErrorResponse
from app.services.evaluator import DamageEvaluator
from app.services.image_io import load_upload
from app.services.severity import derive_overall_severity, enrich, summarize

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
    evaluator: DamageEvaluator = Depends(get_evaluator),
    settings: Settings = Depends(get_settings),
) -> DamageEvaluationResponse:
    image, metadata = await load_upload(file, settings)

    started = time.perf_counter()
    raw = await to_thread.run_sync(evaluator.predict, image)
    inference_ms = (time.perf_counter() - started) * 1000.0

    detections = enrich(raw)
    summary = summarize(detections)
    overall = derive_overall_severity(detections)

    return DamageEvaluationResponse(
        request_id=str(uuid.uuid4()),
        image=metadata,
        detections=detections,
        summary=summary,
        overall_severity=overall,
        model_name=evaluator.name,
        model_version=evaluator.version,
        is_mock=evaluator.is_mock,
        inference_ms=round(inference_ms, 3),
    )
