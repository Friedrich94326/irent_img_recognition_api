#!/usr/bin/env python
"""Evaluate the trained YOLOv8 damage detector against the real Hotai iRent dataset.

Two modes:

* ``--dataset DIR`` (default): scan a flat folder of images, where ground truth follows
  the dataset's own convention - an image is "damaged" if a same-named labelme ``.json``
  sidecar exists next to it, "not damaged" otherwise. Each image is oriented to landscape
  (see :func:`hotai_common.orient_horizontal`) before being handed to the detector.
* ``--manifest CSV``: evaluate exactly the images listed in a
  ``held_out_manifest.csv`` produced by ``prepare_hotai_finetune_dataset.py`` (columns
  ``filename,oriented_image_path,ground_truth``). Those images are already oriented, so
  they're used as-is - this is how a fine-tuned checkpoint is scored only on data it
  never trained on.

This scores image-level "any damage detected" against the ground truth
(precision/recall/F1 + confusion matrix), not box-level IoU/mAP: the labelme
boxes/polygons live in the pre-rotation coordinate space, and geometric scoring is out
of scope for this pass.

Usage:
    python scripts/evaluate_hotai_dataset.py \
        --dataset "Datasets/Hotai_iRent_cars/Annotated"

    python scripts/evaluate_hotai_dataset.py \
        --manifest Datasets/Hotai_finetune/held_out_manifest.csv \
        --weights weights/car_damage_hotai.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
from hotai_common import draw_predictions, find_images, labelme_labels, orient_horizontal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.services.evaluator import build_evaluator  # noqa: E402


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _load_image(image_path: Path, already_oriented: bool) -> Image.Image:
    with Image.open(image_path) as im:
        if already_oriented:
            return im.convert("RGB")
        return orient_horizontal(im).image


def _samples_from_dataset(dataset: Path) -> list[tuple[Path, bool, str]]:
    images = find_images(dataset)
    if not images:
        raise SystemExit(f"No images found under {dataset}")
    return [(p, p.with_suffix(".json").exists(), p.name) for p in images]


def _samples_from_manifest(manifest: Path) -> list[tuple[Path, bool, str]]:
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in {manifest}")
    return [
        (Path(row["oriented_image_path"]), row["ground_truth"] == "damaged", row["filename"])
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Datasets/Hotai_iRent_cars/Annotated"),
        help="Folder of images with optional same-named labelme .json sidecars",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="held_out_manifest.csv from prepare_hotai_finetune_dataset.py; evaluates "
        "exactly its rows instead of scanning --dataset",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Override the configured YOLOv8 weights file for this run only",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.weights is not None:
        settings = settings.model_copy(update={"yolo_weights_path": args.weights})
    evaluator = build_evaluator(settings)

    if args.manifest is not None:
        samples = _samples_from_manifest(args.manifest)
        output_dir = args.manifest.parent
        source_label = f"manifest {args.manifest}"
    else:
        samples = _samples_from_dataset(args.dataset)
        output_dir = args.dataset
        source_label = str(args.dataset)

    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    tp = fp = fn = tn = 0

    for image_path, is_damaged, filename in samples:
        gt_labels = (
            labelme_labels(image_path.with_suffix(".json"))
            if args.manifest is None and is_damaged
            else []
        )

        oriented = _load_image(image_path, already_oriented=args.manifest is not None)
        detections = evaluator.predict(oriented)
        predicted_damaged = len(detections) > 0
        predicted_classes = sorted({d.damage_class.value for d in detections})

        if is_damaged and predicted_damaged:
            tp += 1
            category = "tp"
        elif not is_damaged and predicted_damaged:
            fp += 1
            category = "fp"
        elif is_damaged and not predicted_damaged:
            fn += 1
            category = "fn"
        else:
            tn += 1
            category = "tn"

        annotated = draw_predictions(oriented, detections)
        annotated.save(review_dir / f"{Path(filename).stem}.jpg", "JPEG", quality=90)

        rows.append(
            {
                "filename": filename,
                "ground_truth": "damaged" if is_damaged else "not_damaged",
                "predicted": "damaged" if predicted_damaged else "not_damaged",
                "category": category,
                "num_detections": len(detections),
                "predicted_classes": ";".join(predicted_classes),
                "labelme_labels": ";".join(gt_labels),
            }
        )

    total = len(samples)
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision == precision and recall == recall else float("nan")

    false_positives = [r["filename"] for r in rows if r["ground_truth"] == "not_damaged" and r["predicted"] == "damaged"]
    false_negatives = [r["filename"] for r in rows if r["ground_truth"] == "damaged" and r["predicted"] == "not_damaged"]

    lines = [
        f"Evaluator: {evaluator.name} v{evaluator.version} (mock={evaluator.is_mock})",
        f"Source: {source_label} ({total} images: {tp + fn} damaged, {tn + fp} not damaged)",
        "",
        f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}",
        f"Accuracy:  {accuracy:.3f}",
        f"Precision: {precision:.3f}",
        f"Recall:    {recall:.3f}",
        f"F1:        {f1:.3f}",
        "",
        f"{len(false_positives)} false positive(s) (flagged damaged, no ground-truth JSON):",
        *(f"  {name}" for name in false_positives),
        "",
        f"{len(false_negatives)} false negative(s) (missed known damage):",
        *(f"  {name}" for name in false_negatives),
    ]

    report = "\n".join(lines)
    print(report)

    report_path = output_dir / "eval_report.txt"
    report_path.write_text(report, encoding="utf-8")

    csv_path = output_dir / "eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote report to {report_path}")
    print(f"Wrote per-image results to {csv_path}")
    print(f"Wrote {len(rows)} annotated overlay(s) to {review_dir}")


if __name__ == "__main__":
    main()
