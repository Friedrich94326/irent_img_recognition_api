#!/usr/bin/env python
"""Train the 4-class corner classifier (左前 / 右前 / 左後 / 右後) that checks which corner a
condition photo really shows.

Labels come from the Hotai file names ('<plate> <corner> <借|還>[n]', parsed exactly like
import_hotai_photos.py). Each photo is EXIF-corrected and turned landscape the way the API
does before classifying (app.services.image_io.orient_landscape), then written to an Ultralytics
classify layout split into train/val **by car** (all of a car's photos land on the same side, so
the val score isn't inflated by near-duplicates). Horizontal flips are disabled: a mirrored left
view is a right view.

Usage:
    python scripts/train_corner_classifier.py --device 0
Then set IRENT_CORNER_WEIGHTS_PATH=./weights/corner.pt in .env and restart the API.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_hotai_photos import SourcePhoto, parse_name  # noqa: E402

from app.schemas.vehicle import Corner  # noqa: E402
from app.services.image_io import orient_landscape  # noqa: E402

_MAX_SIDE = 640  # the classifier trains at a few hundred px; no need to copy full-size photos


def split_by_car(
    photos: list[SourcePhoto], val_fraction: float, seed: int
) -> tuple[list[SourcePhoto], list[SourcePhoto]]:
    groups: dict[str, list[SourcePhoto]] = defaultdict(list)
    for p in photos:
        groups[p.plate].append(p)
    cars = sorted(groups)
    random.Random(seed).shuffle(cars)
    val_cars = set(cars[: max(1, round(len(cars) * val_fraction))])
    train = [p for c in cars if c not in val_cars for p in groups[c]]
    val = [p for c in cars if c in val_cars for p in groups[c]]
    return train, val


def load_landscape(path: Path) -> Image.Image:
    with Image.open(path) as raw:
        image = orient_landscape(ImageOps.exif_transpose(raw).convert("RGB"))
    image.thumbnail((_MAX_SIDE, _MAX_SIDE))
    return image


def write_split(photos: list[SourcePhoto], output: Path, split: str) -> None:
    for corner in Corner:
        (output / split / corner.value).mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(photos):
        load_landscape(p.path).save(output / split / p.corner.value / f"{i:05d}.jpg", quality=90)


def print_confusion(weights: Path, val: list[SourcePhoto], device: str) -> None:
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights))
    confusion: Counter[tuple[str, str]] = Counter()
    wrong: list[str] = []
    for p in val:
        probs = model.predict(load_landscape(p.path), device=device, verbose=False)[0].probs
        predicted = model.names[probs.top1]
        confusion[(p.corner.value, predicted)] += 1
        if predicted != p.corner.value:
            wrong.append(f"  {p.path.name}: predicted {predicted} ({float(probs.top1conf):.2f})")
    names = [c.value for c in Corner]
    correct = sum(confusion[(c, c)] for c in names)
    print(f"\nval top-1: {correct}/{len(val)} = {correct / max(1, len(val)):.1%}")
    print("rows = file name, columns = predicted")
    print(f"{'':>12}" + "".join(f"{n:>12}" for n in names))
    for actual in names:
        print(f"{actual:>12}" + "".join(f"{confusion[(actual, n)]:>12}" for n in names))
    if wrong:
        print("misclassified:\n" + "\n".join(wrong))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/Hotai_iRent_cars/Raw"))
    parser.add_argument("--output", type=Path, default=Path("data/Hotai_corner_cls"))
    parser.add_argument("--model", default="yolov8n-cls.pt", help="ImageNet-pretrained start")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="'cpu' or a CUDA index such as '0'")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("weights/corner.pt"))
    args = parser.parse_args()

    photos = [p for f in sorted(args.source_dir.rglob("*")) if (p := parse_name(f))]
    if not photos:
        raise SystemExit(f"No '<plate> <corner> <借|還>' photos under {args.source_dir}")
    if args.output.exists():
        shutil.rmtree(args.output)  # rebuilt from the source photos every run
    train, val = split_by_car(photos, args.val_fraction, args.seed)
    write_split(train, args.output, "train")
    write_split(val, args.output, "val")
    print(f"train: {len(train)} photos   val: {len(val)} photos   "
          f"({len({p.plate for p in photos})} cars)")

    from ultralytics import YOLO  # noqa: PLC0415 - heavy; only needed once the split is written

    results = YOLO(args.model).train(
        data=str(args.output.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        fliplr=0.0,  # a mirrored left view is a right view
        name="corner",
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, args.out)
    print(f"\nCopied {best} -> {args.out}")
    print_confusion(args.out, val, args.device)


if __name__ == "__main__":
    main()
