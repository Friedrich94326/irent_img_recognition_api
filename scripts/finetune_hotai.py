#!/usr/bin/env python
"""Fine-tune the CarDD-trained YOLOv8 checkpoint on real Hotai iRent data.

Continues training from an existing checkpoint (default: weights/car_damage.pt) using a
low learning rate, on the dataset produced by prepare_hotai_finetune_dataset.py. Adding
the missing_part and parts_broken classes (not present in CarDD) means Ultralytics reinitializes the
detection head - the backbone/neck weights still transfer, only the head starts fresh.

Writes the resulting weights to a new file (default: weights/car_damage_hotai.pt) rather
than overwriting the existing weights/car_damage.pt or .env, so the current model stays
in place until the fine-tuned one is evaluated and accepted.

Usage:
    python scripts/finetune_hotai.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/Hotai_finetune/data.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("weights/car_damage.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--out", type=Path, default=Path("weights/car_damage_hotai.pt"))
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_weights, args.out)
    print(f"\nCopied {best_weights} -> {args.out}")


if __name__ == "__main__":
    main()
