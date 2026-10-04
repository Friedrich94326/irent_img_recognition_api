#!/usr/bin/env python
"""Build a YOLO dataset for fine-tuning the damage detector on real Hotai iRent photos.

Two real, reliably-labeled groups are used:

* The 32 images under ``--annotated`` that have a labelme ``.json`` sidecar - full
  box/polygon damage annotations.
* All images under ``--negatives`` (the "no claim filed" folder) - real photos known to
  have no damage, used as hard negatives (empty YOLO label files). No auto-labeling is
  needed since the ground truth is already known from the folder.

Each image is oriented to landscape first (see ``hotai_common.orient_horizontal``), and
damage boxes are carried through the same rotation so they stay aligned with the pixels
the detector will actually see (see the plan's box-rotation formula: for a portrait image
of size ``old_w x old_h`` rotated 90 degrees into landscape, a point ``(x, y)`` maps to
``(y, old_w - x)``).

labelme ``parts_boken`` (sic) is the ``parts_broken`` class - a cracked/broken body part such
as a bumper, low severity - and ``parts_missing`` maps onto ``missing_part``;
``dent``/``scratch``/``crack`` pass through unchanged. The undamaged negatives carry the iRent
hood logo, which teaches the model it is not a scratch.

The damaged and negative groups are split into train/val separately (stratified) so the
tiny damaged class isn't starved out of either split. The val split's *original* source
images are recorded in ``held_out_manifest.csv`` so evaluation can later be run on
exactly the held-out set.

Usage:
    python scripts/prepare_hotai_finetune_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

from hotai_common import find_images, labelme_shapes, orient_horizontal

LABEL_MAP = {
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack",
    "parts_boken": "parts_broken",  # Hotai's spelling
    "parts_broken": "parts_broken",
    "parts_missing": "missing_part",
}
# Same id order as data/CarDD_yolo/data.yaml for classes 0-5, so those ids keep
# lining up with the checkpoint being fine-tuned; missing_part (6) and parts_broken (7) are new.
CLASS_NAMES = [
    "dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat", "missing_part",
    "parts_broken",
]
CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def rotate_point(x: float, y: float, old_w: int) -> tuple[float, float]:
    """Map a point from the pre-rotation (EXIF-corrected) frame into the post-ROTATE_90
    landscape frame: (x, y) -> (y, old_w - x)."""

    return y, old_w - x


def shape_to_box(shape: dict, rotated: bool, old_w: int) -> tuple[float, float, float, float] | None:
    label = LABEL_MAP.get(shape["label"])
    if label is None:
        return None
    points = shape["points"]
    if rotated:
        points = [rotate_point(x, y, old_w) for x, y in points]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return label, min(xs), min(ys), max(xs), max(ys)


def to_yolo_line(label: str, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{CLASS_IDS[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def build_damaged_labels(oriented, json_path: Path) -> list[str]:
    width, height = oriented.image.size
    lines = []
    for shape in labelme_shapes(json_path):
        result = shape_to_box(shape, oriented.rotated, oriented.pre_rotation_size[0])
        if result is None:
            continue
        label, x1, y1, x2, y2 = result
        lines.append(to_yolo_line(label, x1, y1, x2, y2, width, height))
    return lines


def write_sample(
    image_path: Path,
    oriented,
    label_lines: list[str],
    split: str,
    output: Path,
) -> Path:
    images_dir = output / "images" / split
    labels_dir = output / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    dest_image = images_dir / f"{image_path.stem}.jpg"
    oriented.image.save(dest_image, "JPEG", quality=95)
    (labels_dir / f"{image_path.stem}.txt").write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
    )
    return dest_image


def split_group(items: list[Path], val_fraction: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    shuffled = items[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotated", type=Path, default=Path("data/Hotai_iRent_cars/Annotated"))
    parser.add_argument("--negatives", type=Path, default=Path("data/Hotai_iRent_cars/Raw/沒有進行索賠"))
    parser.add_argument("--output", type=Path, default=Path("data/Hotai_finetune"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)

    rng = random.Random(args.seed)

    damaged_images = [p for p in find_images(args.annotated) if p.with_suffix(".json").exists()]
    negative_images = find_images(args.negatives)

    damaged_train, damaged_val = split_group(damaged_images, args.val_fraction, rng)
    negative_train, negative_val = split_group(negative_images, args.val_fraction, rng)

    manifest_rows: list[dict[str, str]] = []

    def process(image_path: Path, split: str, is_damaged: bool) -> None:
        with Image.open(image_path) as im:
            oriented = orient_horizontal(im)
        lines = build_damaged_labels(oriented, image_path.with_suffix(".json")) if is_damaged else []
        dest_image = write_sample(image_path, oriented, lines, split, args.output)
        if split == "val":
            manifest_rows.append(
                {
                    "filename": image_path.name,
                    "oriented_image_path": str(dest_image),
                    "ground_truth": "damaged" if is_damaged else "not_damaged",
                }
            )

    for image_path in damaged_train:
        process(image_path, "train", True)
    for image_path in damaged_val:
        process(image_path, "val", True)
    for image_path in negative_train:
        process(image_path, "train", False)
    for image_path in negative_val:
        process(image_path, "val", False)

    data_yaml = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (args.output / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    manifest_path = args.output / "held_out_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "oriented_image_path", "ground_truth"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Damaged: {len(damaged_train)} train / {len(damaged_val)} val")
    print(f"Negative: {len(negative_train)} train / {len(negative_val)} val")
    print(f"Total: {len(damaged_train) + len(negative_train)} train / {len(damaged_val) + len(negative_val)} val")
    print(f"Wrote dataset to {args.output}")
    print(f"Wrote held-out manifest to {manifest_path}")


if __name__ == "__main__":
    main()
