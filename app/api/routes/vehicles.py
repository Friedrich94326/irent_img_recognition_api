"""Vehicle list and corner condition photos (pushed by station cameras / the return app)."""

from __future__ import annotations

from anyio import to_thread
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import (
    get_corner_classifier,
    get_evaluator,
    get_settings,
    get_vehicle_photo_repository,
)
from app.api.routes.precheck import to_corner_photo
from app.config import Settings
from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.schemas.errors import ErrorResponse
from app.schemas.precheck import CornerPhoto, VehicleSummary
from app.schemas.vehicle import Corner, PhotoSource
from app.services.corner_classifier import CornerClassifier, is_mismatch
from app.services.evaluator import DamageEvaluator
from app.services.image_io import load_upload
from app.services.vehicle_link import normalise_plate
from app.services.vehicle_photos import VehiclePhotoRepository, inspect_photo

router = APIRouter(tags=["vehicles"])


def _require(repo: VehiclePhotoRepository | None) -> VehiclePhotoRepository:
    if repo is None:
        raise ServiceUnavailableError("The iRent ops database is not configured.")
    return repo


@router.get(
    "/vehicles",
    response_model=list[VehicleSummary],
    responses={503: {"model": ErrorResponse}},
    summary="List vehicles with how many corners have a current photo",
)
async def list_vehicles(
    repo: VehiclePhotoRepository | None = Depends(get_vehicle_photo_repository),
) -> list[VehicleSummary]:
    rows = await to_thread.run_sync(_require(repo).list_vehicles)
    return [VehicleSummary(**r.__dict__) for r in rows]


@router.post(
    "/vehicles/{plate}/photos",
    response_model=CornerPhoto,
    status_code=201,
    responses={
        404: {"model": ErrorResponse, "description": "Unknown plate"},
        413: {"model": ErrorResponse, "description": "Image exceeds the maximum allowed size"},
        415: {"model": ErrorResponse, "description": "Unsupported image content type"},
        422: {"model": ErrorResponse, "description": "Bad corner / source or unreadable image"},
        503: {"model": ErrorResponse, "description": "The ops database is not configured"},
    },
    summary="Upload a corner condition photo of a car (station camera / return inspection)",
)
async def upload_vehicle_photo(
    plate: str,
    corner: Corner = Form(..., description="Which corner the photo shows."),
    file: UploadFile = File(..., description="A JPEG, PNG or WebP photo of that corner."),
    source: PhotoSource = Form(default=PhotoSource.STATION, description="Who took the photo."),
    captured_at: str | None = Form(
        default=None, description="When it was taken, 'YYYY-MM-DD HH:MM:SS' UTC. Default: now."
    ),
    evaluator: DamageEvaluator = Depends(get_evaluator),
    classifier: CornerClassifier | None = Depends(get_corner_classifier),
    repo: VehiclePhotoRepository | None = Depends(get_vehicle_photo_repository),
    settings: Settings = Depends(get_settings),
) -> CornerPhoto:
    """The photo is filed under the given ``corner``. When the corner classifier is configured
    and confidently sees a different corner, the photo is still stored but flagged
    (``corner_mismatch``)."""

    repo = _require(repo)
    license_plate = normalise_plate(plate)
    vehicle_id = await to_thread.run_sync(repo.find_vehicle_id, license_plate)
    if vehicle_id is None:
        raise NotFoundError(f"No vehicle with plate '{license_plate}'.", field="plate")

    image, _ = await load_upload(file, settings)
    inspection = await to_thread.run_sync(inspect_photo, evaluator, image)
    check = await to_thread.run_sync(classifier.predict, image) if classifier else None
    mismatch = is_mismatch(corner, check, settings.corner_min_confidence)
    photo = await to_thread.run_sync(
        lambda: repo.add_photo(
            vehicle_id, corner, source, image, inspection, evaluator.name, captured_at,
            corner_check=check, corner_mismatch=mismatch,
        )
    )
    return to_corner_photo(photo)


@router.get(
    "/vehicle-photos/{photo_id}/image",
    response_class=FileResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="The stored JPEG of a corner photo",
)
async def vehicle_photo_image(
    photo_id: int,
    repo: VehiclePhotoRepository | None = Depends(get_vehicle_photo_repository),
) -> FileResponse:
    repo = _require(repo)
    photo = await to_thread.run_sync(repo.get, photo_id)
    path = repo.file_for(photo) if photo else None
    if path is None or not path.is_file():
        raise NotFoundError(f"No photo with id {photo_id}.", field="photo_id")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
