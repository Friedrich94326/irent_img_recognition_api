#!/usr/bin/env python
"""Train the two-class YOLO tyre + licence-plate detector on the by-car Hotai split.

Reads the labelme JSONs that ``split_hotai_dataset.py`` wrote under
``auto_labelled/<Training_data|Validation_data|Test_data>/<Good|Damaged>/``, converts the
``tyre`` and ``license_plate`` shapes to YOLO boxes (other labels, e.g. damage, are ignored),
trains from a COCO-pretrained checkpoint, reports per-class val and test metrics, and copies
the best weights to weights/tyre_plate.pt.

Images are EXIF-transposed and saved without EXIF so the pixels YOLO trains on are exactly
the frame the labelme polygons (and the API's uploads, see app/services/image_io.py) use.

Usage:
    python scripts/train_tyre_plate_detector.py --device cpu
Then set IRENT_TIRE_WEIGHTS_PATH and IRENT_PLATE_DETECTOR_WEIGHTS_PATH to
./weights/tyre_plate.pt in .env and restart the API.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageOps

SOURCE = Path("data/Hotai_iRent_cars/auto_labelled")
SPLITS = {"train": "Training_data", "val": "Validation_data", "test": "Test_data"}
CLASS_NAMES = ["tyre", "license_plate"]
CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def to_yolo_lines(shapes: list[dict], width: int, height: int) -> list[str]:
    lines = []
    for shape in shapes:
        cls = CLASS_IDS.get(shape["label"])
        if cls is None:
            continue
        xs = [min(max(p[0], 0.0), width) for p in shape["points"]]
        ys = [min(max(p[1], 0.0), height) for p in shape["points"]]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(
            f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return lines


def write_split(source: Path, output: Path, split: str) -> tuple[int, list[int]]:
    images_dir, labels_dir = output / "images" / split, output / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    n_images, counts = 0, [0] * len(CLASS_NAMES)
    for j in sorted(source.glob("*/*.json")):
        data = json.loads(j.read_text(encoding="utf-8"))
        image = ImageOps.exif_transpose(Image.open(j.parent / data["imagePath"])).convert("RGB")
        lines = to_yolo_lines(data["shapes"], *image.size)
        # Good and Damaged never share a car, but prefix anyway so names can never collide.
        stem = f"{j.parent.name}_{j.stem}"
        image.save(images_dir / f"{stem}.jpg", quality=95)
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_images += 1
        for line in lines:
            counts[int(line.split()[0])] += 1
    return n_images, counts


def print_metrics(title: str, metrics) -> None:
    print(f"\n{title}\n{'class':<15}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
    for i, name in enumerate(CLASS_NAMES):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{name:<15}{p:>7.3f}{r:>7.3f}{ap50:>8.3f}{ap:>10.3f}")
    print(f"{'all':<15}{metrics.box.mp:>7.3f}{metrics.box.mr:>7.3f}{metrics.box.map50:>8.3f}{metrics.box.map:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=Path("data/Hotai_tyre_plate"))
    parser.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained starting point")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="'cpu' or a CUDA index such as '0'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("weights/tyre_plate.pt"))
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)  # rebuilt from the labelme split every run, never hand-edited
    for split, folder in SPLITS.items():
        n_images, counts = write_split(args.source / folder, args.output, split)
        if not n_images:
            raise SystemExit(
                f"No labelme JSONs under {args.source / folder}; run split_hotai_dataset.py first"
            )
        per_class = ", ".join(f"{c} {n}" for n, c in zip(CLASS_NAMES, counts, strict=True))
        print(f"{split}: {n_images} images / {per_class}")

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(args.output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": dict(enumerate(CLASS_NAMES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    if args.device == "cpu":
        # torch's optimizer still probes the accelerator on CPU runs and crashes when the GPU is
        # busy; hide it before torch is imported. ("-1", not "": Windows drops empty variables.)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from ultralytics import YOLO  # noqa: PLC0415 - heavy; only needed once the dataset is written

    results = YOLO(args.model).train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        fliplr=0.5,
        name="tyre_plate",
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    model = YOLO(str(best))
    for split in ("val", "test"):
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=args.imgsz,
            device=args.device,
            plots=False,
            verbose=False,
        )
        print_metrics(f"{split} split", metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, args.out)
    print(f"\nCopied {best} -> {args.out}")


if __name__ == "__main__":
    main()
