#!/usr/bin/env python
"""Auto-annotate a folder of images into YOLO-format labels without drawing any boxes
by hand, using an open-vocabulary zero-shot detector (Grounding DINO) driven by a
text-prompt ontology.

Requires requirements-autolabel.txt. See .claude/skills/auto-annotate-dataset for the
full workflow (choosing categories, reviewing output, training) this script fits into.

Usage:
    python scripts/auto_annotate.py \
        --input Datasets/Hotai_iRent_cars/Raw \
        --output Datasets/Hotai_yolo_draft \
        --ontology scripts/ontology.yaml \
        --limit 20          # dry run on a random sample first
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".webp"}


def find_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def stage_images(images: list[Path], images_out: Path) -> dict[Path, Path]:
    """Normalize every source image to a uniquely-named .jpg inside images_out.

    Source folders here mix .jpg/.jfif and have images across multiple subfolders that
    can share a filename, so re-encoding via PIL both guards against corrupt/odd files
    and guarantees unique, YOLO-friendly filenames. The EXIF orientation is applied
    (and not re-saved), so staged images are upright exactly as the API sees uploads.
    """
    images_out.mkdir(parents=True, exist_ok=True)
    mapping: dict[Path, Path] = {}
    for i, src in enumerate(images):
        dest = images_out / f"{i:05d}_{src.stem}.jpg"
        try:
            with Image.open(src) as im:
                ImageOps.exif_transpose(im).convert("RGB").save(dest, "JPEG", quality=95)
        except Exception as exc:  # noqa: BLE001 - skip unreadable/corrupt images
            print(f"  skip (unreadable): {src} ({exc})", file=sys.stderr)
            continue
        mapping[dest] = src
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Folder of images to label (searched recursively)")
    parser.add_argument("--output", required=True, type=Path, help="Output YOLO dataset folder")
    parser.add_argument("--ontology", required=True, type=Path, help="YAML file mapping text prompts -> class names")
    parser.add_argument("--conf", type=float, default=0.3, help="Box/text confidence threshold (default 0.3)")
    parser.add_argument("--limit", type=int, default=None, help="Randomly sample N images instead of labeling everything (for a quick dry run)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --limit sampling")
    parser.add_argument(
        "--max-box-frac", type=float, default=1.0,
        help="Drop boxes covering more than this fraction of the image (e.g. 0.25 for tires, "
             "where Grounding DINO sometimes boxes the whole car); default 1.0 keeps all",
    )
    args = parser.parse_args()

    try:
        from autodistill.detection import CaptionOntology
        from autodistill_grounding_dino import GroundingDINO
    except ImportError:
        raise SystemExit(
            "Missing auto-labeling dependencies. Run: pip install -r requirements-autolabel.txt"
        )

    ontology_map = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    if not ontology_map:
        raise SystemExit(f"Ontology file {args.ontology} is empty")

    ontology = CaptionOntology(ontology_map)
    class_names = ontology.classes()
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    images = find_images(args.input)
    if not images:
        raise SystemExit(f"No images found under {args.input}")
    if args.limit is not None:
        random.seed(args.seed)
        images = random.sample(images, min(args.limit, len(images)))
    print(f"Labeling {len(images)} image(s) from {args.input}")

    images_out = args.output / "images"
    labels_out = args.output / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)
    mapping = stage_images(images, images_out)

    model = GroundingDINO(ontology=ontology, box_threshold=args.conf, text_threshold=args.conf)

    class_counts = {name: 0 for name in class_names}
    zero_detection: list[str] = []

    for staged_path, original_path in mapping.items():
        detections = model.predict(str(staged_path))
        with Image.open(staged_path) as im:
            w, h = im.size

        lines = []
        for i in range(len(detections.class_id)):
            class_id = detections.class_id[i]
            if class_id is None:
                continue
            x1, y1, x2, y2 = detections.xyxy[i]
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw * bh > args.max_box_frac:
                continue
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counts[class_names[class_id]] += 1

        (labels_out / f"{staged_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if not lines:
            zero_detection.append(str(original_path))

    data_yaml = {
        "path": str(args.output.resolve()),
        "train": "images",
        "val": "images",
        "names": dict(enumerate(class_names)),
    }
    (args.output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print("\nPer-class detection counts:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    print(f"\n{len(zero_detection)} image(s) with zero detections (review these first):")
    for path in zero_detection[:20]:
        print(f"  {path}")
    if len(zero_detection) > 20:
        print(f"  ... and {len(zero_detection) - 20} more")

    print(f"\nWrote draft YOLO dataset to {args.output}")
    print("train/val both point at the same 'images' folder - this is a review draft, not a")
    print("training-ready split. Split into train/val after correcting labels (step 6/7).")


if __name__ == "__main__":
    main()
