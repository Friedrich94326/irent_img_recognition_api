"""Repair-fee estimate endpoint: damage sized against the tires in the same photo."""

from __future__ import annotations

import time
import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import (
    get_evaluator,
    get_fee_model,
    get_settings,
    get_tire_detector,
    get_vehicle_repository,
)
from app.config import Settings
from app.schemas.common import BoundingBox
from app.schemas.errors import ErrorResponse
from app.schemas.fee import DamagedArea, FeeEstimateResponse, ImpactCluster, MeasuredDamage
from app.services.evaluator import DamageEvaluator
from app.services.image_io import load_upload
from app.services.repair_fee import (
    FeeModel,
    damaged_area,
    features,
    impact_clusters,
    measure,
    overall,
)
from app.services.tire_detector import TireDetector, to_tire_detections
from app.services.vehicle_link import link_vehicle, to_vehicle_link
from app.services.vehicle_repository import VehicleRepository

router = APIRouter(prefix="/fee", tags=["fee"])

_ERROR_RESPONSES = {
    413: {"model": ErrorResponse, "description": "Image exceeds the maximum allowed size"},
    415: {"model": ErrorResponse, "description": "Unsupported image content type"},
    422: {"model": ErrorResponse, "description": "Missing or unreadable image"},
}


def _box(xyxy: tuple[float, float, float, float]) -> BoundingBox:
    return BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])


@router.post(
    "/estimate",
    response_model=FeeEstimateResponse,
    responses=_ERROR_RESPONSES,
    summary="Estimate the repair fee of the damage in one vehicle photo",
)
async def estimate_fee(
    file: UploadFile = File(..., description="A JPEG, PNG or WebP photo showing a tire."),
    plate_number: str | None = Form(
        default=None,
        description="Vehicle plate, e.g. 'RAC-4582'. If omitted, it is taken from the "
        "uploaded file's name (e.g. 'RDX-2376.jpg').",
    ),
    confidence_threshold: float | None = Form(
        default=None,
        ge=0.0,
        le=1.0,
        description="Drop damage detections below this confidence (0-1). Default: the server's "
        "IRENT_YOLO_CONFIDENCE_THRESHOLD.",
    ),
    evaluator: DamageEvaluator = Depends(get_evaluator),
    tire_detector: TireDetector = Depends(get_tire_detector),
    fee_model: FeeModel | None = Depends(get_fee_model),
    repo: VehicleRepository | None = Depends(get_vehicle_repository),
    settings: Settings = Depends(get_settings),
) -> FeeEstimateResponse:
    image, metadata = await load_upload(file, settings)
    threshold = (
        settings.yolo_confidence_threshold if confidence_threshold is None
        else confidence_threshold
    )
    tyre_cm = settings.tyre_diameter_cm

    started = time.perf_counter()
    raw_damage = await to_thread.run_sync(evaluator.predict, image, threshold)
    raw_tires = await to_thread.run_sync(tire_detector.detect, image)
    inference_ms = (time.perf_counter() - started) * 1000.0

    # Measure against the clipped tires, so tire_index points into the response's `tires`.
    tires = to_tire_detections(raw_tires, image.size)
    tyre_boxes = [(t.bounding_box.x1, t.bounding_box.y1, t.bounding_box.x2, t.bounding_box.y2)
                  for t in tires]
    measured = measure(raw_damage, tyre_boxes, tyre_cm)
    area = damaged_area(measured, tyre_boxes, tyre_cm)
    impacts = impact_clusters(measured, tyre_boxes, tyre_cm, image.size)
    severity, base_severity = overall(measured, impacts)

    fee = None
    if fee_model is not None:
        fee = 0
        if measured:
            row = features(measured, area, impacts)
            fee = int(round(await to_thread.run_sync(fee_model.predict, row), -2))

    # Read-only, like tire detection: an estimate only identifies the vehicle.
    match = await link_vehicle(plate_number, file.filename, repo)
    vehicle_link = None
    if match is not None and match.vehicle is not None:
        vehicle_link = to_vehicle_link(match.vehicle, match.source)

    return FeeEstimateResponse(
        request_id=str(uuid.uuid4()),
        image=metadata,
        detections=[
            MeasuredDamage(
                damage_class=m.damage_class,
                confidence=round(m.confidence, 4),
                bounding_box=_box(m.xyxy),
                base_severity=m.base_severity,
                severity=m.severity,
                tire_share=m.tyre_share,
                area_cm2=m.area_cm2,
                size_band=m.size_band,
                tire_index=m.tyre_index,
            )
            for m in measured
        ],
        tires=tires,
        tire_diameter_cm=tyre_cm,
        damaged_area=DamagedArea(**vars(area)),
        impacts=[
            ImpactCluster(bounding_box=_box(i.box), classes=i.classes, reason=i.reason)
            for i in impacts
        ],
        overall_severity=severity,
        base_overall_severity=base_severity,
        estimated_fee=fee,
        fee_model_name=fee_model.name if fee_model is not None else None,
        model_name=evaluator.name,
        tire_model_name=tire_detector.name,
        is_mock=evaluator.is_mock or tire_detector.is_mock,
        inference_ms=round(inference_ms, 3),
        confidence_threshold=threshold,
        vehicle=vehicle_link,
    )
