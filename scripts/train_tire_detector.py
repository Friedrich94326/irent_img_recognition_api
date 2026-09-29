#!/usr/bin/env python
"""Train the one-class YOLO tire detector from an auto-labelled draft dataset.

Takes the flat draft written by auto_annotate.py with scripts/ontology_tire.yaml, splits it
into train/val **by car** (every iRent car is photographed from four corners at pickup and
return, so a random split would leak the same car into both and inflate the val score),
trains from a COCO-pretrained checkpoint, and copies the best weights to weights/tire.pt.

Usage:
    python scripts/auto_annotate.py --input data/Hotai_iRent_cars/Raw \
        --output data/Hotai_tire_draft --ontology scripts/ontology_tire.yaml \
        --max-box-frac 0.25
    python scripts/train_tire_detector.py --device 0
Then set IRENT_TIRE_WEIGHTS_PATH=./weights/tire.pt in .env and restart the API.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

# Staged names are "{index:05d}_{original stem}"; the stem starts with the plate, e.g.
# "00012_RDU-5572右前 借" or "00040_RFB5176左後 還". The plate (hyphen optional) is the car id.
_CAR_ID_RE = re.compile(r"^\d{5}_([A-Z]{2,3})-?(\d{3,4})")


def car_id(stem: str) -> str:
    m = _CAR_ID_RE.match(stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem  # unknown naming: its own group


def split_by_car(
    images: list[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[car_id(image.stem)].append(image)
    cars = sorted(groups)
    random.Random(seed).shuffle(cars)
    n_val = max(1, round(len(cars) * val_fraction))
    val_cars = set(cars[:n_val])
    train = [p for c in cars if c not in val_cars for p in groups[c]]
    val = [p for c in cars if c in val_cars for p in groups[c]]
    return train, val


def write_split(images: list[Path], labels_dir: Path, output: Path, split: str) -> int:
    (output / "images" / split).mkdir(parents=True, exist_ok=True)
    (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for image in images:
        shutil.copyfile(image, output / "images" / split / image.name)
        label = labels_dir / f"{image.stem}.txt"
        text = label.read_text(encoding="utf-8") if label.exists() else ""
        n_boxes += sum(1 for line in text.splitlines() if line.strip())
        (output / "labels" / split / label.name).write_text(text, encoding="utf-8")
    return n_boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--draft", type=Path, default=Path("data/Hotai_tire_draft"))
    parser.add_argument("--output", type=Path, default=Path("data/Hotai_tire"))
    parser.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained starting point")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="'cpu' or a CUDA index such as '0'")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("weights/tire.pt"))
    args = parser.parse_args()

    images = sorted((args.draft / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"No staged images under {args.draft / 'images'}; run auto_annotate first")
    if args.output.exists():
        shutil.rmtree(args.output)  # rebuilt from the draft every run, never hand-edited
    train, val = split_by_car(images, args.val_fraction, args.seed)
    n_train = write_split(train, args.draft / "labels", args.output, "train")
    n_val = write_split(val, args.draft / "labels", args.output, "val")
    print(f"train: {len(train)} images / {n_train} tires   val: {len(val)} images / {n_val} tires")

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(args.output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "tire"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    from ultralytics import YOLO  # noqa: PLC0415 - heavy; only needed once the split is written

    results = YOLO(args.model).train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        name="tire",
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, args.out)
    print(f"\nCopied {best} -> {args.out}")


if __name__ == "__main__":
    main()
