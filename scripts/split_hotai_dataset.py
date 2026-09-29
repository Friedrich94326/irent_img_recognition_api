#!/usr/bin/env python
"""Split the good-condition and damaged Hotai iRent photos into train/val/test by car.

Every iRent car is photographed from four corners at pickup and return, so a per-image
shuffle would put near-identical shots of one car into several splits and inflate the
val/test scores. Cars are therefore kept whole, and the good and damaged groups are split
separately (stratified) so the small damaged group reaches every split.

Each labelme JSON is copied with its image into
``<output>/<Training_data|Validation_data|Test_data>/<Good|Damaged>/``, with ``imagePath``
rewritten to the bare filename so every split folder opens on its own in labelme. A
``split_manifest.csv`` records where every image went; ``has_damage_labels`` is false for
damaged photos that only carry tyre/plate shapes (do not use those as damage negatives).

Usage:
    python scripts/split_hotai_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

DATASET = Path("data/Hotai_iRent_cars")
SPLITS = {"train": "Training_data", "val": "Validation_data", "test": "Test_data"}
NON_DAMAGE = {"tyre", "license_plate"}

# Stems start with the plate, hyphen optional: "RCG2235右前 借", "RDJ-0772左前 還".
_CAR_ID_RE = re.compile(r"^([A-Z]{2,3})-?(\d{3,4})")


def car_id(stem: str) -> str:
    m = _CAR_ID_RE.match(stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem  # unknown naming: its own group


def split_cars(
    groups: dict[str, list[Path]], fraction: float, seed: int
) -> dict[str, list[str]]:
    cars = sorted(groups)
    random.Random(seed).shuffle(cars)
    n_hold = max(1, round(len(cars) * fraction))
    return {
        "val": cars[:n_hold],
        "test": cars[n_hold : 2 * n_hold],
        "train": cars[2 * n_hold :],
    }


def balanced(groups: dict[str, list[Path]], split: dict[str, list[str]], fraction: float) -> bool:
    """Val and test each hold roughly ``fraction`` of the images, not just of the cars."""

    total = sum(len(v) for v in groups.values())
    for name in ("val", "test"):
        share = sum(len(groups[c]) for c in split[name]) / total
        if not 0.6 * fraction <= share <= 1.4 * fraction:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--good", type=Path, default=DATASET / "Raw" / "沒有進行索賠")
    parser.add_argument("--damaged", type=Path, default=DATASET / "auto_labelled" / "Damaged")
    parser.add_argument("--output", type=Path, default=DATASET / "auto_labelled")
    parser.add_argument(
        "--fraction", type=float, default=0.15, help="share of cars for val and for test"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="clear the split folders first")
    args = parser.parse_args()

    for folder in SPLITS.values():
        dest = args.output / folder
        if dest.exists() and any(dest.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"{dest} is not empty; pass --overwrite to rebuild it")
            shutil.rmtree(dest)

    rows = []
    for condition, source in (("Good", args.good), ("Damaged", args.damaged)):
        groups: dict[str, list[Path]] = defaultdict(list)
        for j in sorted(source.glob("*.json")):
            groups[car_id(j.stem)].append(j)
        seed = args.seed
        split = split_cars(groups, args.fraction, seed)
        while not balanced(groups, split, args.fraction) and seed < args.seed + 1000:
            seed += 1
            split = split_cars(groups, args.fraction, seed)
        print(f"{condition}: {len(groups)} cars, seed {seed}")

        for name, cars in split.items():
            dest = args.output / SPLITS[name] / condition
            dest.mkdir(parents=True, exist_ok=True)
            for car in sorted(cars):
                for j in groups[car]:
                    data = json.loads(j.read_text(encoding="utf-8"))
                    image = (j.parent / data["imagePath"]).resolve()
                    shutil.copyfile(image, dest / image.name)
                    data["imagePath"] = image.name
                    data["imageData"] = None
                    text = json.dumps(data, ensure_ascii=False, indent=2)
                    (dest / j.name).write_text(text, encoding="utf-8")
                    labels = [s["label"] for s in data["shapes"]]
                    n_damage = sum(1 for label in labels if label not in NON_DAMAGE)
                    rows.append({
                        "split": SPLITS[name], "condition": condition, "car_id": car,
                        "file": image.name, "n_tyre": labels.count("tyre"),
                        "n_plate": labels.count("license_plate"), "n_damage": n_damage,
                        "has_damage_labels": condition == "Good" or n_damage > 0,
                    })

    with (args.output / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'split':<16}{'condition':<10}{'cars':>5}{'images':>8}{'tyres':>7}{'plates':>8}{'damage':>8}")
    for folder in SPLITS.values():
        for condition in ("Good", "Damaged"):
            sel = [r for r in rows if r["split"] == folder and r["condition"] == condition]
            print(
                f"{folder:<16}{condition:<10}{len({r['car_id'] for r in sel}):>5}{len(sel):>8}"
                f"{sum(r['n_tyre'] for r in sel):>7}{sum(r['n_plate'] for r in sel):>8}"
                f"{sum(r['n_damage'] for r in sel):>8}"
            )


if __name__ == "__main__":
    main()
