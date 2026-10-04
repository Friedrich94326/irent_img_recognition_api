#!/usr/bin/env python
"""Trace how the plate recogniser turns photos into a plate number, step by step.

For every photo: each raw EasyOCR read, how ``format_plate`` rewrote it (cleaning, leading-noise
strip, which layout regex matched, letter/digit swaps), the vote count and ranking that
``resolve_plates`` applied, and the final best plate - compared with the expected plate.

Run from the repository root (it imports ``app``):

    python .claude/skills/diagnose-plate-misread/scripts/trace_plate.py --plate RDX-2376
    python .claude/skills/diagnose-plate-misread/scripts/trace_plate.py --expect RDX-2376 a.jpg b.jpg
    python .claude/skills/diagnose-plate-misread/scripts/trace_plate.py --text ROI270 RDX2376

``--plate`` collects the car's photos itself: its stored corner photos (vehicles ->
vehicle_photos in the ops DB) and every dataset image whose file name contains the plate.
``--text`` skips OCR and only explains how ``format_plate`` treats the given raw strings.
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "app" / "services" / "plate_recognizer.py").exists():
    sys.exit("Run this from the repository root (the folder that contains app/).")
sys.path.insert(0, str(ROOT))

from app.services import plate_recognizer as pr  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}
DATASET_DIRS = [ROOT / "data" / "Hotai_iRent_cars", ROOT / "data" / "mock_photos"]


def explain_format(text: str) -> tuple[str | None, list[str]]:
    """Mirror ``format_plate`` and narrate each step; returns (result, steps)."""
    steps = []
    cleaned = pr._clean(text)
    if cleaned != text:
        steps.append(f"clean {text!r} -> {cleaned!r}")
    stripped = pr._strip_leading_noise(cleaned)
    if stripped != cleaned:
        steps.append(f"strip leading I/L/J noise -> {stripped!r}")
    m = pr._PLATE_RE.fullmatch(stripped)
    result = None
    if m:
        if m.group("letters"):
            letters, digits = m.group("letters"), m.group("digits")
            layout = f"{len(letters)} letters + {len(digits)} digits"
            fixed = letters.translate(pr._FIX_LETTERS)
            result = f"{fixed}-{digits}"
        else:
            digits, letters = m.group("digits2"), m.group("letters2")
            layout = f"{len(digits)} digits + {len(letters)} letters"
            fixed = letters.translate(pr._FIX_LETTERS)
            result = f"{digits}-{fixed}"
        steps.append(f"matched layout {layout}")
        if fixed != letters:
            steps.append(f"letter 'O' forced to 'Q': {letters} -> {fixed}")
        if not pr._STANDARD_PLATE_RE.fullmatch(result):
            steps.append("NOT the current ABC-1234 layout (older/motorcycle layout accepted)")
    else:
        for n_letters in (3, 2):
            for n_digits in (4, 3):
                if result is None and len(stripped) == n_letters + n_digits:
                    letters = stripped[:n_letters].translate(pr._TO_LETTER)
                    digits = stripped[n_letters:].translate(pr._TO_DIGIT)
                    if letters.isalpha() and digits.isdigit():
                        result = f"{letters}-{digits}"
                        steps.append(
                            f"no direct match; confusion swap {stripped[:n_letters]}"
                            f"|{stripped[n_letters:]} -> {letters}|{digits}"
                        )
        if result is None:
            steps.append("rejected: no plate layout fits")
    if result and set(result.split("-")[0] + result.split("-")[-1]) & {"I", "O"}:
        steps.append("contains I/O, letters Taiwan plates never use")
    actual = pr.format_plate(text)
    if actual != result:  # the mirror drifted from the real code: trust the real result
        steps.append(f"WARNING: trace says {result!r} but format_plate returns {actual!r}; "
                     "re-read format_plate, it has changed")
        result = actual
    return result, steps


def norm(plate: str) -> str:
    return pr.format_plate(plate) or plate.strip().upper()


def photos_for_plate(plate: str, db_path: Path | None) -> list[Path]:
    found: list[Path] = []
    if db_path and db_path.exists():
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT p.file_path FROM vehicle_photos p JOIN vehicles v ON v.id = p.vehicle_id "
                "WHERE v.license_plate = ? ORDER BY p.created_at DESC",
                (plate,),
            ).fetchall()
        except sqlite3.Error as exc:
            print(f"(ops DB lookup failed: {exc})")
            rows = []
        finally:
            conn.close()
        photo_dir = ROOT / os.environ.get("IRENT_VEHICLE_PHOTO_DIR", "data/vehicle_photos")
        found += [photo_dir / r[0] for r in rows if (photo_dir / r[0]).exists()]
    keys = {plate.upper(), plate.replace("-", "").upper()}
    seen = {p.name for p in found}
    for base in DATASET_DIRS:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if "review" in p.parts:  # rendered copies with boxes drawn on them
                continue
            if (p.suffix.lower() in IMAGE_SUFFIXES and p.name not in seen
                    and any(k in p.stem.upper() for k in keys)):
                seen.add(p.name)  # the same photo is often copied into several splits
                found.append(p)
    return found


def trace_image(recognizer, path: Path, expect: str | None, min_conf: float) -> str | None:
    from PIL import Image, ImageOps  # noqa: PLC0415

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    raws = recognizer.read(image)
    plates, best = pr.resolve_plates(raws, image.size, min_conf)
    print(f"\n=== {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
          f"  ({image.width}x{image.height})")
    if not raws:
        print("  no text found at all (locator, detector and whole-frame scan all empty)")
    for r in raws:
        result, steps = explain_format(r.text)
        where = "located crop" if r.corners else "whole-frame scan"
        print(f"  raw {r.text!r:<14} conf {r.confidence:.2f}  [{where}]  -> {result or '-'}")
        for s in steps:
            print(f"      . {s}")
    merged = pr._merge_split_plate(raws)
    if merged:
        print(f"  split boxes joined: {merged.text!r} -> {pr.format_plate(merged.text)}")
    votes = Counter(p.plate_number for p in plates if p.valid_format)
    if votes:
        print("  votes: " + ", ".join(f"{k} x{v}" for k, v in votes.most_common()))
    got = best.plate_number if best else None
    verdict = ""
    if expect:
        verdict = "  OK" if got == expect else f"  MISMATCH (expected {expect})"
    print(f"  BEST: {got}{verdict}")
    return got


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--plate", help="the real plate: collect its photos automatically")
    parser.add_argument("--expect", help="the real plate, when images are given explicitly")
    parser.add_argument("--text", nargs="+", help="only explain format_plate on raw strings")
    parser.add_argument("--max", type=int, default=12, help="max photos to OCR (slow on CPU)")
    args = parser.parse_args()

    if args.text:
        for t in args.text:
            result, steps = explain_format(t)
            print(f"{t!r} -> {result}")
            for s in steps:
                print(f"    . {s}")
        return

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # GPU is often busy with training
    from app.config import Settings  # noqa: PLC0415

    settings = Settings()
    expect = norm(args.plate or args.expect) if (args.plate or args.expect) else None
    images = list(args.images)
    if args.plate:
        images += photos_for_plate(expect, settings.db_path)
    if not images:
        sys.exit("No photos to read. Pass image paths, or a --plate whose photos exist.")
    if len(images) > args.max:
        print(f"{len(images)} photos found; reading the first {args.max} (--max to change)")
        images = images[: args.max]

    if settings.plate_use_mock:
        sys.exit("IRENT_PLATE_USE_MOCK is true: the mock invents plates. Unset it to trace OCR.")
    recognizer = pr.build_plate_recognizer(settings)
    print(f"recogniser: {recognizer.name}  (plate detector: "
          f"{settings.plate_detector_weights_path or 'none'}, "
          f"OpenCV locator: {settings.plate_use_opencv_locator})")

    results = Counter()
    for path in images:
        results[trace_image(recognizer, path, expect, settings.plate_min_confidence)] += 1
    print("\n=== summary: " + ", ".join(f"{k or 'no plate'} x{v}" for k, v in results.most_common()))


if __name__ == "__main__":
    main()
