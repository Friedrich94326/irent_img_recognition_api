#!/usr/bin/env python
"""Score the license-plate recogniser against the real Hotai iRent photos.

Ground truth comes from the filename: every photo is named after the car's plate, e.g.
``RDU-5572右前 借.jpg`` or ``RCG2235右前 借.jfif``. The plate is extracted with
``([A-Z]{3})-?(\\d{4})`` and compared with the recogniser's ``best_plate`` after both are
normalised to ``ABC-1234``.

Usage:
    python scripts/evaluate_plate_recognition.py
    python scripts/evaluate_plate_recognition.py --limit 40 --csv runs/plate_eval.csv
    python scripts/evaluate_plate_recognition.py --no-locator     # full-image OCR baseline
    python scripts/evaluate_plate_recognition.py --plate-detector weights/tyre_plate.pt \
        --dataset data/Hotai_iRent_cars/auto_labelled/Test_data   # held-out cars only
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.services.plate_recognizer import (  # noqa: E402
    EasyOCRPlateRecognizer,
    build_plate_detector,
    resolve_plates,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}
_GT_RE = re.compile(r"([A-Z]{3})-?(\d{4})")


def ground_truth(path: Path) -> str | None:
    m = _GT_RE.search(path.name.upper())
    return f"{m.group(1)}-{m.group(2)}" if m else None


def find_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=Path("data/Hotai_iRent_cars/Raw"))
    parser.add_argument("--limit", type=int, default=0, help="Only score the first N images.")
    parser.add_argument("--csv", type=Path, default=Path("runs/plate_eval.csv"))
    parser.add_argument("--no-locator", action="store_true", help="OCR the full image only.")
    parser.add_argument(
        "--plate-detector",
        type=Path,
        default=None,
        help="YOLO plate weights to try before the locator (e.g. weights/tyre_plate.pt).",
    )
    args = parser.parse_args()

    settings = get_settings()
    detector = None
    if args.plate_detector:
        settings = settings.model_copy(update={"plate_detector_weights_path": args.plate_detector})
        detector = build_plate_detector(settings)
        if detector is None:
            sys.exit(f"Could not load the plate detector from {args.plate_detector}")
    recognizer = EasyOCRPlateRecognizer(use_locator=not args.no_locator, plate_detector=detector)

    images = [p for p in find_images(args.dataset) if ground_truth(p)]
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"No images with a plate in the filename under {args.dataset}")

    rows: list[dict[str, str]] = []
    exact = found = 0
    total_ms = 0.0
    for i, path in enumerate(images, 1):
        truth = ground_truth(path)
        try:
            with Image.open(path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except OSError as exc:
            print(f"[{i}/{len(images)}] SKIP {path.name}: {exc}")
            continue

        started = time.perf_counter()
        raws = recognizer.read(image)
        total_ms += (time.perf_counter() - started) * 1000.0
        _, best = resolve_plates(raws, image.size, settings.plate_min_confidence)

        predicted = best.plate_number if best else ""
        found += bool(best)
        hit = predicted == truth
        exact += hit
        rows.append(
            {
                "file": path.name,
                "truth": truth or "",
                "predicted": predicted,
                "confidence": f"{best.confidence:.2f}" if best else "",
                "match": "1" if hit else "0",
            }
        )
        status = "OK  " if hit else "MISS"
        print(f"[{i}/{len(images)}] {status} truth={truth} pred={predicted or '-'}")

    n = len(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["file", "truth", "predicted", "confidence", "match"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Plate recognition ===")
    print(f"recogniser      : {recognizer.name}")
    print(f"images scored   : {n}")
    print(f"plate found     : {found}/{n} ({found / n:.1%})")
    print(f"exact match     : {exact}/{n} ({exact / n:.1%})")
    print(f"mean latency    : {total_ms / n:.0f} ms/image")
    print(f"per-image CSV   : {args.csv}")


if __name__ == "__main__":
    main()
