"""Find the ``vehicles`` row a photo belongs to: from the client's plate, else by OCR."""

from __future__ import annotations

from anyio import to_thread
from PIL import Image

from app.config import Settings
from app.schemas.vehicle import PlateSource, VehicleLink
from app.services.plate_recognizer import PlateRecognizer, format_plate, resolve_plates
from app.services.vehicle_repository import VehicleRecord, VehicleRepository


def normalise_plate(text: str) -> str:
    """'rac 4582' / 'RAC4582' -> 'RAC-4582'; text that is no known layout is only upper-cased."""

    return format_plate(text) or text.strip().upper()


def to_vehicle_link(vehicle: VehicleRecord, source: PlateSource) -> VehicleLink:
    return VehicleLink(
        id=vehicle.id,
        license_plate=vehicle.license_plate,
        model=vehicle.model,
        color=vehicle.color,
        status=vehicle.status,
        plate_source=source,
    )


async def link_vehicle(
    image: Image.Image,
    plate_number: str | None,
    repo: VehicleRepository | None,
    recognizer: PlateRecognizer,
    settings: Settings,
) -> tuple[VehicleRecord, PlateSource] | None:
    """Return the matched vehicle and where its plate came from, or ``None``.

    Without a database nothing is looked up, and OCR is skipped so requests cost no more than
    before. A plate from the client is trusted as given: OCR only runs when there is none.
    """

    if repo is None:
        return None
    if plate_number and plate_number.strip():
        plate, source = normalise_plate(plate_number), PlateSource.REQUEST
    else:
        raws = await to_thread.run_sync(recognizer.read, image)
        _, best = resolve_plates(raws, image.size, settings.plate_min_confidence)
        if best is None:
            return None
        plate, source = best.plate_number, PlateSource.OCR
    vehicle = await to_thread.run_sync(repo.find_vehicle, plate)
    return (vehicle, source) if vehicle else None
