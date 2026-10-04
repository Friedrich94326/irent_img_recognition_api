"""Find the ``vehicles`` row a photo belongs to: from the client's plate, else from the
uploaded file's name.

A driver rents one car for a fixed period, so its plate is already known: the client names each
upload after it (e.g. ``RDX-2376.jpg``) and no OCR is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

from anyio import to_thread

from app.schemas.vehicle import PlateSource, VehicleLink
from app.services.plate_recognizer import format_plate
from app.services.vehicle_repository import VehicleRecord, VehicleRepository


@dataclass(frozen=True)
class PlateMatch:
    """The plate a photo was attributed to and the ``vehicles`` row it matched, if any."""

    plate: str | None
    source: PlateSource | None
    vehicle: VehicleRecord | None


def normalise_plate(text: str) -> str:
    """'rac 4582' / 'RAC4582' -> 'RAC-4582'; text that is no known layout is only upper-cased."""

    return format_plate(text) or text.strip().upper()


def plate_from_filename(filename: str | None) -> str | None:
    """'RDX-2376.jpg' / 'rdx2376.png' -> 'RDX-2376'; ``None`` when the stem is not a plate."""

    if not filename:
        return None
    # Clients may send a path; either separator style is stripped before taking the stem.
    stem = PurePath(filename.replace("\\", "/")).stem
    return format_plate(stem) if stem else None


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
    plate_number: str | None,
    filename: str | None,
    repo: VehicleRepository | None,
) -> PlateMatch | None:
    """Return the photo's plate and matched vehicle (either may be ``None``), or ``None``
    without a database.

    A plate from the client is trusted as given; otherwise it is taken from ``filename``. A file
    name that is no plate leaves the photo unattributed - nothing is read from the image.
    """

    if repo is None:
        return None
    if plate_number and plate_number.strip():
        plate, source = normalise_plate(plate_number), PlateSource.REQUEST
    else:
        plate = plate_from_filename(filename)
        if plate is None:
            return PlateMatch(None, None, None)
        source = PlateSource.FILENAME
    vehicle = await to_thread.run_sync(repo.find_vehicle, plate)
    return PlateMatch(plate, source, vehicle)
