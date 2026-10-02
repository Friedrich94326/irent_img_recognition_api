"""Import the Hotai iRent corner photos into the photo store, as station cameras / the return
app would push them, so the online pre-rental check has real photos to show.

For every car in ``--source-dir`` that has a photo of all four corners (左前 / 右前 / 左後 / 右後),
the car is registered in ``vehicles`` if it isn't there yet (status available, clean cabin, model
and colour 未知, stations assigned round-robin), and each corner's latest photo is added: the
last return photo (還, 還2, ...) if any, else the last pickup photo (借). Damage detection runs on
each photo with the configured detector (``.env``), exactly as the upload endpoint does, and so
does the corner check when IRENT_CORNER_WEIGHTS_PATH is set: the file name stays the corner of
record, a confident disagreement only flags the photo.

Re-running is safe: photos already imported from the same source file are skipped.

    python scripts/import_hotai_photos.py --dry-run
    python scripts/import_hotai_photos.py
    python scripts/import_hotai_photos.py --recheck   # corner-check every stored photo again
    python scripts/import_hotai_photos.py --reinspect # re-run damage detection on stored photos
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.schemas.vehicle import Corner, PhotoSource  # noqa: E402
from app.services.corner_classifier import (  # noqa: E402
    CornerClassifier,
    CornerPrediction,
    build_corner_classifier,
    is_mismatch,
)
from app.services.evaluator import DamageEvaluator, build_evaluator  # noqa: E402
from app.services.plate_recognizer import format_plate  # noqa: E402
from app.services.vehicle_photos import VehiclePhotoRepository, inspect_photo  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png"}
CORNERS_ZH = {"左前": Corner.FRONT_LEFT, "右前": Corner.FRONT_RIGHT,
              "左後": Corner.REAR_LEFT, "右後": Corner.REAR_RIGHT}
# e.g. 'RCG2235右前 借', 'RCR-7661 左後 還2'
_NAME_RE = re.compile(
    r"^(?P<plate>[A-Za-z0-9-]+?)\s*(?P<corner>左前|右前|左後|右後)\s*(?P<kind>借|還)\s*(?P<n>\d*)$"
)


@dataclass(frozen=True)
class SourcePhoto:
    path: Path
    plate: str
    corner: Corner
    source: PhotoSource
    sequence: int  # 還2 -> 2; no number -> 1


def parse_name(path: Path) -> SourcePhoto | None:
    """Parse '<plate> <corner> <借|還>[n]' file names; ``None`` for anything else."""

    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    m = _NAME_RE.match(path.stem.strip())
    if not m:
        return None
    plate = format_plate(m.group("plate"))
    if plate is None:
        return None
    return SourcePhoto(
        path=path,
        plate=plate,
        corner=CORNERS_ZH[m.group("corner")],
        source=PhotoSource.RETURN if m.group("kind") == "還" else PhotoSource.PICKUP,
        sequence=int(m.group("n") or 1),
    )


def pick_latest(photos: list[SourcePhoto]) -> dict[str, dict[Corner, SourcePhoto]]:
    """Per plate, each corner's latest photo: return (還) beats pickup (借), then the highest
    sequence number. Only plates with all four corners are kept."""

    def rank(p: SourcePhoto) -> tuple[int, int]:
        return (1 if p.source == PhotoSource.RETURN else 0, p.sequence)

    best: dict[str, dict[Corner, SourcePhoto]] = {}
    for p in photos:
        corners = best.setdefault(p.plate, {})
        if p.corner not in corners or rank(p) > rank(corners[p.corner]):
            corners[p.corner] = p
    return {plate: c for plate, c in sorted(best.items()) if len(c) == len(Corner)}


def register_vehicles(db: Path, plates: list[str]) -> dict[str, int]:
    """Return plate -> vehicles.id, inserting the plates that are missing."""

    conn = sqlite3.connect(db)
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON")
            stations = [r[0] for r in conn.execute(
                "SELECT id FROM stations WHERE status = 'active' ORDER BY id")]
            if not stations:
                raise SystemExit("No active stations to assign new vehicles to.")
            ids: dict[str, int] = {}
            added = 0
            for plate in plates:
                row = conn.execute(
                    "SELECT id FROM vehicles WHERE license_plate = ?", (plate,)).fetchone()
                if row:
                    ids[plate] = row[0]
                    continue
                ids[plate] = conn.execute(
                    "INSERT INTO vehicles (license_plate, model, color, station_id) "
                    "VALUES (?, '未知', '未知', ?)",
                    (plate, stations[added % len(stations)]),
                ).lastrowid
                added += 1
    finally:
        conn.close()
    print(f"vehicles: {added} registered, {len(plates) - added} already present")
    return ids


def stable_name(photo: SourcePhoto, source_dir: Path) -> str:
    rel = photo.path.relative_to(source_dir).as_posix()
    return f"hotai_{photo.corner.value}_{hashlib.sha1(rel.encode()).hexdigest()[:12]}"


def describe_mismatch(label: str, corner: Corner, check: CornerPrediction) -> str:
    return (f"  MISMATCH {label}: filed as {corner.value}, looks like {check.corner.value} "
            f"({check.confidence:.0%})")


def recheck(repo: VehiclePhotoRepository, classifier: CornerClassifier, min_conf: float) -> int:
    """Run the corner check on every stored photo and record the result."""

    photos = repo.all_photos()
    flagged = 0
    for photo in photos:
        with Image.open(repo.file_for(photo)) as stored:
            check = classifier.predict(stored.convert("RGB"))
        mismatch = is_mismatch(photo.corner, check, min_conf)
        repo.set_corner_check(photo.id, check, mismatch)
        if mismatch:
            flagged += 1
            print(describe_mismatch(f"photo {photo.id} ({photo.file_path})", photo.corner, check))
    print(f"corner check: {len(photos)} photos checked, {flagged} flagged")
    return 0


def reinspect(repo: VehiclePhotoRepository, evaluator: DamageEvaluator) -> int:
    """Re-run damage detection on every stored photo (after a model or threshold change)."""

    photos = repo.all_photos()
    before = after = 0
    for photo in photos:
        with Image.open(repo.file_for(photo)) as stored:
            inspection = inspect_photo(evaluator, stored.convert("RGB"))
        repo.set_inspection(photo.id, inspection, evaluator.name)
        before += len(photo.detections)
        after += len(inspection.detections)
    print(f"damage re-inspection: {len(photos)} photos, findings {before} -> {after}")
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=Path, default=Path("data/Hotai_iRent_cars/Raw"))
    parser.add_argument("--db", type=Path, default=settings.db_path
                        or Path("data/irent_op_backend.sqlite"))
    parser.add_argument("--photo-dir", type=Path, default=settings.vehicle_photo_dir)
    parser.add_argument("--dry-run", action="store_true", help="Only report what would change")
    parser.add_argument("--recheck", action="store_true",
                        help="Only re-run the corner check on the photos already stored")
    parser.add_argument("--reinspect", action="store_true",
                        help="Only re-run damage detection on the photos already stored")
    args = parser.parse_args()

    if args.reinspect:
        evaluator = build_evaluator(settings)
        print(f"detector: {evaluator.name} (mock={evaluator.is_mock}), "
              f"confidence >= {settings.yolo_confidence_threshold}")
        repo = VehiclePhotoRepository(args.db, args.photo_dir)
        repo.ensure_schema()
        return reinspect(repo, evaluator)

    classifier = build_corner_classifier(settings)
    min_conf = settings.corner_min_confidence
    if args.recheck:
        if classifier is None:
            raise SystemExit("Set IRENT_CORNER_WEIGHTS_PATH to a trained corner classifier first.")
        repo = VehiclePhotoRepository(args.db, args.photo_dir)
        repo.ensure_schema()
        return recheck(repo, classifier, min_conf)

    found = [p for f in sorted(args.source_dir.rglob("*")) if (p := parse_name(f))]
    cars = pick_latest(found)
    print(f"{len(found)} corner photos found, {len(cars)} cars with all four corners")
    if args.dry_run:
        for plate, corners in cars.items():
            picks = ", ".join(f"{c.value}={p.path.name}" for c, p in corners.items())
            print(f"  {plate}: {picks}")
        return 0

    repo = VehiclePhotoRepository(args.db, args.photo_dir)
    repo.ensure_schema()
    ids = register_vehicles(args.db, list(cars))

    evaluator = build_evaluator(settings)
    print(f"detector: {evaluator.name} (mock={evaluator.is_mock}), "
          f"corner check: {classifier.name if classifier else 'off'}")
    added = skipped = 0
    for plate, corners in cars.items():
        for corner, photo in corners.items():
            name = stable_name(photo, args.source_dir)
            if repo.has_file(f"{ids[plate]}/{name}.jpg"):
                skipped += 1
                continue
            with Image.open(photo.path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            captured = datetime.fromtimestamp(photo.path.stat().st_mtime, UTC)
            inspection = inspect_photo(evaluator, image)
            check = classifier.predict(image) if classifier else None
            mismatch = is_mismatch(corner, check, min_conf)
            repo.add_photo(
                ids[plate], corner, photo.source, image, inspection, evaluator.name,
                captured.strftime("%Y-%m-%d %H:%M:%S"), name,
                corner_check=check, corner_mismatch=mismatch,
            )
            added += 1
            if mismatch:
                print(describe_mismatch(photo.path.name, corner, check))
            print(f"  {plate} {corner.value}: {len(inspection.detections)} finding(s), "
                  f"{inspection.overall_severity.value}")
    print(f"photos: {added} added, {skipped} already imported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
