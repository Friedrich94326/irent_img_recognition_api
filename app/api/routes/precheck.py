"""Online pre-rental check endpoint."""

from __future__ import annotations

import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends

from app.api.deps import get_precheck_repository, get_vehicle_photo_repository
from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.schemas.errors import ErrorResponse
from app.schemas.precheck import (
    CornerPhoto,
    KnownIssues,
    PrecheckRequest,
    PrecheckVehicle,
    RentalPrecheckResponse,
    Verdict,
)
from app.schemas.vehicle import CORNER_LABELS_ZH, Corner
from app.services.precheck import PrecheckRepository, decide
from app.services.severity import derive_overall_severity
from app.services.vehicle_link import normalise_plate
from app.services.vehicle_photos import VehiclePhoto, VehiclePhotoRepository

router = APIRouter(prefix="/rental", tags=["rental"])

_ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Unknown plate_number or member_no"},
    422: {"model": ErrorResponse, "description": "Missing plate_number"},
    503: {"model": ErrorResponse, "description": "The ops database is not configured"},
}


def photo_url(photo_id: int) -> str:
    return f"/api/v1/vehicle-photos/{photo_id}/image"


def to_corner_photo(photo: VehiclePhoto) -> CornerPhoto:
    return CornerPhoto(
        photo_id=photo.id,
        corner=photo.corner,
        corner_label=CORNER_LABELS_ZH[photo.corner],
        source=photo.source,
        captured_at=photo.captured_at,
        image_url=photo_url(photo.id),
        width=photo.width,
        height=photo.height,
        detections=photo.detections,
        summary=photo.summary,
        overall_severity=photo.overall_severity,
        model_name=photo.model_name,
        predicted_corner=photo.predicted_corner,
        corner_confidence=photo.corner_confidence,
        corner_mismatch=photo.corner_mismatch,
    )


@router.post(
    "/precheck",
    response_model=RentalPrecheckResponse,
    responses=_ERROR_RESPONSES,
    summary="Check a car's current condition online before renting it",
)
async def precheck_rental(
    body: PrecheckRequest,
    repo: PrecheckRepository | None = Depends(get_precheck_repository),
    photos_repo: VehiclePhotoRepository | None = Depends(get_vehicle_photo_repository),
) -> RentalPrecheckResponse:
    """The car's current corner photos come from the photo store (station cameras / the last
    return inspection) with the damage found when they were uploaded; nothing is uploaded here."""

    if repo is None or photos_repo is None:
        raise ServiceUnavailableError("The iRent ops database is not configured.")

    plate = normalise_plate(body.plate_number)
    snapshot = await to_thread.run_sync(repo.vehicle_snapshot, plate)
    if snapshot is None:
        raise NotFoundError(f"No vehicle with plate '{plate}'.", field="plate_number")
    customer_id = None
    member_no = (body.member_no or "").strip()
    if member_no:
        customer_id = await to_thread.run_sync(repo.find_customer, member_no)
        if customer_id is None:
            raise NotFoundError(f"No customer with member_no '{member_no}'.", field="member_no")

    photos = await to_thread.run_sync(photos_repo.latest_by_corner, snapshot.id)
    overall = derive_overall_severity([d for p in photos.values() for d in p.detections])
    verdict, reasons = decide(snapshot, photos, overall)
    case_id = str(uuid.uuid4())
    precheck_id = await to_thread.run_sync(
        repo.record, case_id, snapshot, customer_id, verdict, reasons, photos, overall
    )

    return RentalPrecheckResponse(
        precheck_id=precheck_id,
        case_id=case_id,
        verdict=verdict,
        can_rent=verdict != Verdict.BLOCKED,
        reasons=reasons,
        vehicle=PrecheckVehicle(
            id=snapshot.id,
            license_plate=snapshot.license_plate,
            model=snapshot.model,
            color=snapshot.color,
            status=snapshot.status,
            cabin_condition=snapshot.cabin_condition,
            station_name=snapshot.station_name,
        ),
        known_issues=KnownIssues(
            latest_anomaly=snapshot.latest_anomaly,
            unresolved_alerts=snapshot.unresolved_alerts,
            open_repair_orders=snapshot.open_repairs,
        ),
        photos=[to_corner_photo(photos[c]) for c in Corner if c in photos],
        missing_corners=[c for c in Corner if c not in photos],
        overall_severity=overall,
    )
