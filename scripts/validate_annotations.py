#!/usr/bin/env python
"""Render a draft YOLO dataset's boxes onto its images and report QA stats, so most of
the review prescribed by the auto-annotate-dataset skill's Step 6 can happen by flipping
through a folder of images instead of importing into CVAT/Label Studio first.

See .claude/skills/auto-annotate-dataset for the full workflow this fits into.

Usage:
    python scripts/validate_annotations.py \
        --dataset data/Hotai_yolo_draft_sample \
        --claim-root data/Hotai_iRent_cars/Raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

TINY_BOX_FRACTION = 0.02
HUGE_BOX_FRACTION = 0.9
NO_CLAIM_FOLDER = "沒有進行索賠"
CLAIM_FOLDER = "進行索賠資料"

BOX_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
]


def load_class_names(dataset: Path) -> dict[int, str]:
    data_yaml = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    return {int(k): v for k, v in data_yaml["names"].items()}


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        class_id, cx, cy, bw, bh = line.split()
        boxes.append((int(class_id), float(cx), float(cy), float(bw), float(bh)))
    return boxes


def build_claim_index(claim_root: Path) -> dict[str, set[str]]:
    """Map each original image's lowercased stem to the set of top-level subfolders
    (e.g. 進行索賠資料 / 沒有進行索賠) it appears under in claim_root."""
    index: dict[str, set[str]] = {}
    for path in claim_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(claim_root).parts
        if not rel_parts:
            continue
        index.setdefault(path.stem.lower(), set()).add(rel_parts[0])
    return index


def classify_claim_status(staged_stem: str, claim_index: dict[str, set[str]]) -> str:
    """Recover whether a staged image (named '{i:05d}_{original_stem}.jpg' by
    auto_annotate.py) came from the no-claim folder, using the original stem embedded
    after the 5-digit prefix."""
    original_stem = staged_stem[6:]  # strip "NNNNN_"
    folders = claim_index.get(original_stem.lower())
    if not folders:
        return "unknown"
    if folders == {NO_CLAIM_FOLDER}:
        return "no_claim"
    if folders == {CLAIM_FOLDER}:
        return "claim"
    return "ambiguous"


def draw_boxes(image: Image.Image, boxes: list[tuple[int, float, float, float, float]], names: dict[int, str]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    w, h = annotated.size
    for class_id, cx, cy, bw, bh in boxes:
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = names.get(class_id, str(class_id))
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color, font=font)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="Draft YOLO dataset folder (images/, labels/, data.yaml)")
    parser.add_argument("--claim-root", type=Path, default=None, help="Original raw image tree, to flag no-claim images with detections first")
    args = parser.parse_args()

    images_dir = args.dataset / "images"
    labels_dir = args.dataset / "labels"
    review_dir = args.dataset / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    names = load_class_names(args.dataset)
    claim_index = build_claim_index(args.claim_root) if args.claim_root else None

    class_counts = {name: 0 for name in names.values()}
    zero_detection: list[str] = []
    suspicious: list[str] = []
    no_claim_with_detections: list[str] = []

    image_paths = sorted(p for p in images_dir.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No images found under {images_dir}")

    for image_path in image_paths:
        boxes = parse_label_file(labels_dir / f"{image_path.stem}.txt")
        with Image.open(image_path) as im:
            annotated = draw_boxes(im, boxes, names)
        annotated.save(review_dir / image_path.name, "JPEG", quality=90)

        if not boxes:
            zero_detection.append(image_path.name)

        for class_id, _cx, _cy, bw, bh in boxes:
            class_counts[names.get(class_id, str(class_id))] += 1
            if bw < TINY_BOX_FRACTION or bh < TINY_BOX_FRACTION or bw > HUGE_BOX_FRACTION or bh > HUGE_BOX_FRACTION:
                suspicious.append(image_path.name)

        if claim_index is not None and boxes:
            if classify_claim_status(image_path.stem, claim_index) == "no_claim":
                no_claim_with_detections.append(image_path.name)

    suspicious = sorted(set(suspicious))

    lines = []
    lines.append(f"Reviewed {len(image_paths)} image(s) from {args.dataset}\n")
    lines.append("Per-class detection counts:")
    for name, count in class_counts.items():
        lines.append(f"  {name}: {count}")

    lines.append(f"\n{len(zero_detection)} image(s) with zero detections:")
    lines.extend(f"  {name}" for name in zero_detection)

    if claim_index is not None:
        lines.append(f"\n{len(no_claim_with_detections)} '{NO_CLAIM_FOLDER}' image(s) with detections (likely false positives):")
        lines.extend(f"  {name}" for name in no_claim_with_detections)

    lines.append(f"\n{len(suspicious)} image(s) with a suspiciously tiny/huge box:")
    lines.extend(f"  {name}" for name in suspicious)

    lines.append("\nSuggested review order (open these in review/ first):")
    lines.append("  1. zero-detection images")
    if claim_index is not None:
        lines.append(f"  2. {NO_CLAIM_FOLDER} images with detections")
    lines.append("  3. suspicious tiny/huge boxes")
    lines.append("  4. a random spot-check of the rest")

    report = "\n".join(lines)
    print(report)
    (args.dataset / "review_report.txt").write_text(report, encoding="utf-8")
    print(f"\nWrote annotated overlays to {review_dir}")
    print(f"Wrote report to {args.dataset / 'review_report.txt'}")


if __name__ == "__main__":
    main()
