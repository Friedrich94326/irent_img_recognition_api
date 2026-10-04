#!/usr/bin/env python
"""Build web/demo.html: damage detections sized against a tyre in the same photo.

A tyre is a ruler of known size (~63 cm for a 195/65 R15 rental car tyre), so each damage box is
reported as a share of the tyre's area and as an approximate cm² figure, and its severity is
nudged up or down by that size. This is a demo-only calculation - the API is unchanged.

Runs the damage model from .env (IRENT_YOLO_WEIGHTS_PATH) and the tyre model given by
--tire-weights on the Damaged photos of the Hotai split (test and validation first, so the tyre
model has not seen them), draws the boxes and tyre ellipses, and writes a static page that opens
straight from disk.

Usage:
    python scripts/build_demo_page.py
    python scripts/build_demo_page.py --tire-weights weights/tyre_plate.pt
"""
# ruff: noqa: E501 - the page template below keeps its HTML/CSS lines whole
from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.repair_fee import (  # noqa: E402 - needs the sys.path entry above
    TYRE_DIAMETER_CM,
    Measured,
    boxes_overlap,
    damaged_area,
    impact_clusters,
    measure,
    overall,
)

SOURCE = Path("data/Hotai_iRent_cars/auto_labelled")
SPLIT_ORDER = ("Test_data", "Validation_data", "Training_data")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".jfif"}
DEFAULT_PHOTOS = (
    "RDJ-0793左前 還",  # bumper dent beside the front tyre
    "RDX-2376右後 還",  # broken tail lamp
    "RCR-7661 左後 借",  # small scratch above the wheel arch
    "RCR-7661 左後 還",  # scratches, dent and a flat tyre
    "RDE-0073 車況回報",  # no tyre in view: shows the unscaled fallback
)
"""Photos car_damage.pt gets right; most others have false detections (logos, glare, tarmac)."""
MAX_WIDTH = 1280
DAMAGE_COLOUR, TYRE_COLOUR, IMPACT_COLOUR = (255, 138, 76), (56, 189, 248), (239, 68, 68)


def font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _dashed_rectangle(pen, box, colour, width: int, dash: int) -> None:
    x0, y0, x1, y1 = box
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                               ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = math.dist((ax, ay), (bx, by))
        for start in range(0, int(length), 2 * dash):
            t0, t1 = start / length, min(start + dash, length) / length
            pen.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0),
                      (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=colour, width=width)


def draw(image: Image.Image, measured: list[Measured], tyres, tyre_cm: float,
         impacts: list) -> Image.Image:
    scale = min(1.0, MAX_WIDTH / image.width)
    canvas = image.resize((round(image.width * scale), round(image.height * scale)))
    pen = ImageDraw.Draw(canvas)
    stroke = max(3, round(canvas.width / 320))
    label_font = font(max(16, round(canvas.width / 55)))

    used = {m.tyre_index for m in measured}
    labels = []  # (box, text, colour, preferred positions); placed after every outline is drawn
    for impact in impacts:
        margin = 3 * stroke
        box = [impact.box[0] * scale - margin, impact.box[1] * scale - margin,
               impact.box[2] * scale + margin, impact.box[3] * scale + margin]
        box = [max(0, box[0]), max(0, box[1]), min(canvas.width - 1, box[2]),
               min(canvas.height - 1, box[3])]
        _dashed_rectangle(pen, box, IMPACT_COLOUR, stroke, 6 * stroke)
        labels.append((box, "impact", IMPACT_COLOUR, ("below", "inside", "above")))
    for i, tyre in enumerate(tyres):
        box = [v * scale for v in tyre.xyxy]
        pen.ellipse(box, outline=TYRE_COLOUR, width=stroke if i in used else max(1, stroke // 2))
        if i in used:
            labels.append((box, f"tyre ≈ {tyre_cm:.0f} cm", TYRE_COLOUR, ("below", "above")))
    for m in measured:
        box = [v * scale for v in m.xyxy]
        pen.rectangle(box, outline=DAMAGE_COLOUR, width=stroke)
        text = m.damage_class.replace("_", " ")
        if m.tyre_share is not None:
            text += f" · {m.tyre_share:.0%} tyre · ~{m.area_cm2:,.0f} cm²"
        labels.append((box, text, DAMAGE_COLOUR, ("above", "inside", "below")))

    placed: list[tuple[float, float, float, float]] = []
    for box, text, colour, positions in labels:
        left, top, right, bottom = pen.textbbox((0, 0), text, font=label_font)
        w, h = right - left + 12, bottom - top + 10
        x = max(0, min(box[0], canvas.width - w))
        ys = {"above": box[1] - h, "inside": box[1] + stroke, "below": box[3]}
        rects = [(x, min(max(0, ys[p]), canvas.height - h)) for p in positions]
        rects = [(rx, ry, rx + w, ry + h) for rx, ry in rects]
        # First spot that doesn't cover an earlier label; overlapping labels leave clipped text.
        rect = next((r for r in rects if not any(boxes_overlap(r, q) for q in placed)), rects[0])
        placed.append(rect)
        pen.rectangle(rect, fill=colour)
        pen.text((rect[0] + 6, rect[1] + 5 - top), text, fill=(20, 22, 26), font=label_font)
    return canvas


def candidates(source: Path) -> list[Path]:
    paths = []
    for split in SPLIT_ORDER:
        folder = source / split / "Damaged"
        paths += sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--tire-weights", type=Path, default=Path("runs/detect/tyre_plate/weights/best.pt")
    )
    parser.add_argument("--tyre-cm", type=float, default=TYRE_DIAMETER_CM)
    parser.add_argument("--max", type=int, default=8, help="photos with both damage and a tyre")
    parser.add_argument("--no-tyre-examples", type=int, default=1)
    parser.add_argument(
        "--only", nargs="*", default=list(DEFAULT_PHOTOS),
        help="photo file stems to use (default: a hand-checked set; pass no names to scan them all)",
    )
    parser.add_argument(
        "--ip-photo", default="RDU-6956 右前 借",
        help="photo file stem for the image-processing section (plate location, glare)",
    )
    parser.add_argument(
        "--fee-model", type=Path, default=Path("weights/repair_fee_xgb.json"),
        help="XGBoost model from scripts/repair_fee_model.py; skipped when the file is missing",
    )
    parser.add_argument("--out", type=Path, default=Path("web/demo.html"))
    args = parser.parse_args()

    # Keep off the GPU (see train_tyre_plate_detector.py); "-1" because Windows drops "" values.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from app.config import Settings  # noqa: PLC0415
    from app.services.evaluator import build_evaluator  # noqa: PLC0415
    from app.services.tire_detector import YOLOTireDetector  # noqa: PLC0415

    settings = Settings()
    evaluator = build_evaluator(settings)
    if evaluator.is_mock:
        raise SystemExit("Damage model fell back to the mock; check IRENT_YOLO_WEIGHTS_PATH")
    tyre_detector = YOLOTireDetector(
        settings.model_copy(update={"tire_weights_path": args.tire_weights})
    )

    fee_model = None
    if args.fee_model.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import repair_fee_model  # noqa: PLC0415 - sibling script, needs xgboost

        fee_model = repair_fee_model.load_model(args.fee_model)

    image_dir = args.out.parent / "demo"
    if image_dir.exists():
        shutil.rmtree(image_dir)  # generated output only
    image_dir.mkdir(parents=True)

    with_tyre, without_tyre = [], []
    for path in candidates(args.source):
        if args.only and path.stem not in args.only:
            continue
        if len(with_tyre) >= args.max and len(without_tyre) >= args.no_tyre_examples:
            break
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        raw = evaluator.predict(image)
        if not raw:
            continue
        tyres = tyre_detector.detect(image)
        bucket, limit = (with_tyre, args.max) if tyres else (without_tyre, args.no_tyre_examples)
        if len(bucket) >= limit:
            continue
        boxes = [t.xyxy for t in tyres]
        measured = measure(raw, boxes, args.tyre_cm)
        area = damaged_area(measured, boxes, args.tyre_cm)
        impacts = impact_clusters(measured, boxes, args.tyre_cm, image.size)
        severity, base_severity = (s.value for s in overall(measured, impacts))
        fee = None
        if fee_model is not None:
            row = repair_fee_model.features(measured, area, impacts)
            fee = round(repair_fee_model.predict_fee(fee_model, row), -2)
        n = len(with_tyre) + len(without_tyre) + 1
        name = f"{n:02d}.jpg"
        draw(image, measured, tyres, args.tyre_cm, impacts).save(image_dir / name, quality=88)
        bucket.append({
            "image": f"demo/{name}",
            "source": f"{path.parent.parent.name}/{path.name}",
            "tyre_confidence": tyres[0].confidence if tyres else None,
            "overall": severity,
            "base_overall": base_severity,
            "area": vars(area),
            "impacts": [vars(i) for i in impacts],
            "fee": fee,
            "detections": [m.__dict__ for m in measured],
        })
        print(f"{name} {path.name}: {len(raw)} damage, {len(tyres)} tyre -> "
              + ", ".join(f"{m.damage_class} {m.tyre_share:.0%}" if m.tyre_share is not None
                          else m.damage_class for m in measured)
              + f" | union {area.union_px:.0f}px vs summed {area.summed_px:.0f}px"
              + (f" ({area.union_share:.0%} vs {area.summed_share:.0%} tyre)"
                 if area.union_share is not None else "")
              + f" | overall {base_severity}"
              + (f" -> {severity} [{'; '.join(i.reason for i in impacts)}]" if impacts else "")
              + (f" | fee NT${fee:,.0f}" if fee is not None else ""))

    cards = with_tyre + without_tyre
    if not cards:
        raise SystemExit("No photo produced a damage detection")
    explainers = []
    if fee_model is not None and repair_fee_model.SIMULATED_CSV.exists():
        demo_fees = [(Path(c["source"]).stem, c["fee"]) for c in cards]
        explainers = [f"demo/{p.name}" for p in
                      repair_fee_model.plot_explainers(fee_model, image_dir, demo_fees=demo_fees)]

    ip_steps = {"plate": [], "glare": []}
    ip_path = next((p for p in candidates(args.source) if p.stem == args.ip_photo), None)
    if ip_path is None:
        print(f"Image-processing photo {args.ip_photo!r} not found; section skipped")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import image_processing_demo  # noqa: PLC0415 - sibling script

        ip_image = ImageOps.exif_transpose(Image.open(ip_path)).convert("RGB")
        from app.services.plate_recognizer import YOLOPlateDetector  # noqa: PLC0415

        plate_detector = YOLOPlateDetector(
            settings.model_copy(update={"plate_detector_weights_path": args.tire_weights})
        )
        ip_steps = image_processing_demo.build_steps(ip_image, evaluator, image_dir,
                                                     plate_detector)
        ip_steps["source"] = f"{ip_path.parent.parent.name}/{ip_path.name}"
        print(f"Image processing on {ip_path.name}: {len(ip_steps['plate'])} plate steps; "
              f"glare before: {ip_steps['glare_effect']['before']} | "
              f"after: {ip_steps['glare_effect']['after']}")

    page = TEMPLATE.replace("__DATA__", json.dumps(cards, ensure_ascii=False)).replace(
        "__META__",
        html.escape(
            f"Damage model {settings.yolo_weights_path} · tyre model {args.tire_weights} · "
            + (f"fee model {args.fee_model} (XGBoost, trained on simulated history) · "
               if fee_model is not None else "")
            + f"generated {datetime.now():%Y-%m-%d %H:%M}"
        ),
    ).replace("__TYRE_CM__", f"{args.tyre_cm:.0f}").replace(
        "__EXPLAINERS__", json.dumps(explainers)
    ).replace("__IP_STEPS__", json.dumps(ip_steps, ensure_ascii=False))
    args.out.write_text(page, encoding="utf-8")
    print(f"\nWrote {args.out} with {len(cards)} photos ({len(with_tyre)} with a tyre)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Damage Size Demo</title>
<style>
  :root {
    --bg: #f5f6f8; --panel: #ffffff; --border: #e2e5ea; --text: #1b2129; --text-dim: #667085;
    --accent: #c1410c; --accent-dim: #f7e6dc; --tyre: #0369a1;
    --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    --sev-none: #1a7f37; --sev-none-bg: #dcfce7;
    --sev-minor: #a16207; --sev-minor-bg: #fef3c7;
    --sev-moderate: #c2410c; --sev-moderate-bg: #ffedd5;
    --sev-severe: #b91c1c; --sev-severe-bg: #fee2e2;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --panel: #1c1f26; --border: #2b2f38; --text: #e6e8ec; --text-dim: #9aa2b1;
      --accent: #ff8a4c; --accent-dim: #3a2417; --tyre: #38bdf8;
      --sev-none: #3fb950; --sev-none-bg: #123321;
      --sev-minor: #e0b23d; --sev-minor-bg: #3a2f10;
      --sev-moderate: #ff8a4c; --sev-moderate-bg: #3a2417;
      --sev-severe: #f85149; --sev-severe-bg: #3b1618;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1200px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .meta { color: var(--text-dim); font: 12px var(--mono); overflow-wrap: anywhere; }
  .lede { max-width: 70ch; color: var(--text-dim); margin: 12px 0 24px; }
  .steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px;
           list-style: none; padding: 0; margin: 0 0 12px; counter-reset: step; }
  .steps li { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
              padding: 12px 14px; counter-increment: step; }
  .steps li::before { content: counter(step); display: inline-grid; place-items: center;
                      width: 22px; height: 22px; border-radius: 50%; margin-right: 8px;
                      background: var(--accent-dim); color: var(--accent); font-weight: 700;
                      font-size: 12px; }
  .steps b { display: block; margin-top: 6px; }
  .steps span { color: var(--text-dim); font-size: 13px; }
  .note { font-size: 13px; color: var(--text-dim); margin: 0 0 28px; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; margin-bottom: 16px; }
  .legend i { display: inline-block; width: 14px; height: 14px; margin-right: 6px;
              vertical-align: -2px; border: 3px solid; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr));
          gap: 20px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
          overflow: hidden; display: flex; flex-direction: column; }
  .card img { width: 100%; display: block; background: #000; cursor: zoom-in; }
  .card header { display: flex; justify-content: space-between; align-items: center; gap: 12px;
                 padding: 12px 16px 4px; }
  .card header span { font: 12px var(--mono); color: var(--text-dim); overflow-wrap: anywhere; }
  .badge { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
           padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
  .sev-none { color: var(--sev-none); background: var(--sev-none-bg); }
  .sev-minor { color: var(--sev-minor); background: var(--sev-minor-bg); }
  .sev-moderate { color: var(--sev-moderate); background: var(--sev-moderate-bg); }
  .sev-severe { color: var(--sev-severe); background: var(--sev-severe-bg); }
  .table-wrap { overflow-x: auto; padding: 0 16px 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { color: var(--text-dim); font-weight: 600; }
  td.num, th.num { text-align: right; }
  .arrow { color: var(--text-dim); margin: 0 4px; }
  .notyre { padding: 0 16px 8px; font-size: 13px; color: var(--text-dim); }
  .area { padding: 0 16px 8px; font-size: 14px; }
  .area small { color: var(--text-dim); }
  .fee { padding: 0 16px 10px; margin: 0; font-size: 15px; }
  .fee b { font-size: 20px; color: var(--accent); font-variant-numeric: tabular-nums; }
  .impact { margin: 0 16px 10px; padding: 8px 12px; border-radius: 8px; font-size: 13px;
            color: var(--sev-severe); background: var(--sev-severe-bg); }
  .up { font-size: 11px; font-weight: 700; color: var(--sev-severe); margin-left: 6px; }
  h2 { font-size: 22px; margin: 48px 0 4px; }
  .concept { display: grid; gap: 16px; }
  .concept figure { margin: 0; background: #fcfcfb; border: 1px solid var(--border);
                    border-radius: 12px; padding: 12px; }
  .concept img { width: 100%; display: block; cursor: zoom-in; }
  h3 { font-size: 17px; margin: 28px 0 10px; }
  h3 small { font-weight: 400; color: var(--text-dim); font-size: 13px; }
  .steps-row { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(260px, 300px);
               gap: 14px; overflow-x: auto; padding-bottom: 10px; scroll-snap-type: x mandatory; }
  .step { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
          overflow: hidden; scroll-snap-align: start; display: flex; flex-direction: column; }
  .step img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: #000;
              display: block; cursor: zoom-in; }
  .step div { padding: 10px 12px 12px; font-size: 13px; color: var(--text-dim); }
  .step b { display: block; color: var(--text); font-size: 14px; margin-bottom: 2px; }
  .step b span { color: var(--accent); margin-right: 6px; }
  @media (max-width: 640px) {
    .steps-row { grid-auto-flow: row; grid-auto-columns: auto; overflow-x: visible; }
  }
  .concept .step-arrow { text-align: center; color: var(--text-dim); font-size: 22px; line-height: 1; }
  dialog { border: 0; padding: 0; background: transparent; max-width: 96vw; }
  dialog img { max-width: 96vw; max-height: 92vh; display: block; }
  dialog::backdrop { background: rgba(0,0,0,.8); }
</style>
</head>
<body>
<main>
  <h1>Damage size, measured against a tyre</h1>
  <div class="meta">__META__</div>
  <p class="lede">A photo has no scale on its own, so a scratch box of 200 pixels could be a scuff
    or half a door. Every rental car photo also shows a tyre of a roughly known size, so we use
    the tyre as a ruler. Each detected damage is reported as a share of the tyre's area and as
    an approximate area in cm², and its severity is moved up or down by that size.</p>
  <ol class="steps">
    <li><b>Detect damage</b><span>YOLO damage model: class, confidence and box</span></li>
    <li><b>Detect the tyres</b><span>Tyre + plate model; each damage uses the nearest wheel</span></li>
    <li><b>Tyre = __TYRE_CM__ cm ruler</b><span>Longer ellipse axis ≈ true diameter, even at an angle</span></li>
    <li><b>Size the damage</b><span>Box area ÷ tyre area, and cm² from the pixels-per-cm scale</span></li>
    <li><b>Adjust severity</b><span>&lt; 5 % of tyre: one level down · &gt; 25 %: one level up</span></li>
    <li><b>Merge &amp; check for impact</b><span>Overlaps counted once; a collision pattern makes the photo severe</span></li>
    <li><b>Estimate fee (XGBoost)</b><span>Damage mix, sizes and impact → NT$; trained on simulated repair history for now</span></li>
  </ol>
  <p class="note">The cm² figure assumes the damage is about as far from the camera as the tyre,
    and it uses the box rather than the exact damage outline, so read it as an estimate.</p>
  <div class="legend"><span><i style="border-color:var(--accent)"></i>Damage box</span>
    <span><i style="border-color:var(--tyre);border-radius:50%"></i>Tyre used as reference</span>
    <span><i style="border-color:var(--sev-severe);border-style:dashed"></i>Impact (likely collision)</span></div>
  <div class="grid" id="grid"></div>
  <section id="ip" hidden>
    <h2>Image processing behind the scenes</h2>
    <p class="lede">Classic computer-vision steps that fit rental-car photos: one finds the
      licence plate for OCR, the other removes light reflections before damage detection.
      <span class="meta" id="ip-source"></span></p>
    <h3>Locating the licence plate <small>· the OpenCV pipeline the plate API uses</small></h3>
    <div class="steps-row" id="ip-plate"></div>
    <h3>Filtering light reflections <small>· concept, not in the API yet</small></h3>
    <div class="steps-row" id="ip-glare"></div>
  </section>
  <section id="fee-concept" hidden>
    <h2>How the fee model works</h2>
    <p class="lede">Past repair cases teach an XGBoost regression model what each kind, size and
      combination of damage costs; the model then prices any new photo from the same features.
      It is trained on simulated history until real repair invoices are exported.</p>
    <div class="concept" id="concept"></div>
  </section>
</main>
<dialog id="zoom"><img alt="" /></dialog>
<script>
  const cards = __DATA__;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]);
  const badge = s => `<span class="badge sev-${s}">${s}</span>`;
  const pct = v => v == null ? '–' : (v * 100).toFixed(v < 0.1 ? 1 : 0) + ' %';
  const cm2 = v => v == null ? '–' : '~' + Math.round(v).toLocaleString() + ' cm²';
  const areaLine = a => {
    if (!a.union_px) return '<small>Only a flat tyre, so no bodywork area.</small>';
    if (a.union_share == null) return `<b>Damaged area (union):</b> ${Math.round(a.union_px).toLocaleString()} px² <small>· unscaled, no tyre in view${a.summed_px > a.union_px * 1.01 ? ` · boxes summed: ${Math.round(a.summed_px).toLocaleString()} px²` : ''}</small>`;
    return `<b>Damaged area (union):</b> ${pct(a.union_share)} of tyre · ${cm2(a.union_cm2)}` +
      (a.summed_share > a.union_share * 1.01 ? ` <small>· boxes summed: ${pct(a.summed_share)}</small>` : '');
  };
  document.getElementById('grid').innerHTML = cards.map(c => `
    <article class="card">
      <img src="${esc(c.image)}" alt="Annotated photo ${esc(c.source)}" loading="lazy" />
      <header><span>${esc(c.source)}</span><span>${c.overall !== c.base_overall ? '<span class="up">↑ impact</span> ' : ''}${badge(c.overall)}</span></header>
      <p class="area">${areaLine(c.area)}</p>
      ${c.fee == null ? '' : `<p class="fee">Estimated repair fee: <b>NT$${Math.round(c.fee).toLocaleString()}</b></p>`}
      ${c.impacts.map(i => `<p class="impact"><b>Impact:</b> ${esc(i.classes.map(k => k.replace(/_/g, ' ')).join(' + '))}. ${esc(i.reason)}. Raised to severe (likely collision or broken bumper).</p>`).join('')}
      ${c.tyre_confidence == null ? '<p class="notyre">No tyre in view, so there is no scale; severity comes from class and confidence only.</p>' : ''}
      <div class="table-wrap"><table>
        <thead><tr><th>Damage</th><th class="num">Conf.</th><th class="num">Of tyre</th>
          <th class="num">Est. area</th><th>Severity</th></tr></thead>
        <tbody>${c.detections.map(d => `<tr>
          <td>${esc(d.damage_class.replace(/_/g, ' '))}</td>
          <td class="num">${d.confidence.toFixed(2)}</td>
          <td class="num">${pct(d.tyre_share)}</td>
          <td class="num">${cm2(d.area_cm2)}</td>
          <td>${d.base_severity === d.severity ? badge(d.severity)
               : badge(d.base_severity) + '<span class="arrow">→</span>' + badge(d.severity)}</td>
        </tr>`).join('')}</tbody>
      </table></div>
    </article>`).join('');
  const ip = __IP_STEPS__;
  if (ip.plate.length || ip.glare.length) {
    document.getElementById('ip').hidden = false;
    document.getElementById('ip-source').textContent = `Photo: ${ip.source}`;
    const row = steps => steps.map((s, i) => `<article class="step">
      <img src="${esc(s.src)}" alt="${esc(s.title)}" loading="lazy" />
      <div><b><span>${i + 1}</span>${esc(s.title)}</b>${esc(s.caption)}</div></article>`).join('');
    document.getElementById('ip-plate').innerHTML = row(ip.plate);
    document.getElementById('ip-glare').innerHTML = row(ip.glare);
    document.getElementById('ip').addEventListener('click', e => {
      if (e.target.tagName !== 'IMG') return;
      zoomImage(e.target.src);
    });
  }
  function zoomImage(src) {
    const dialog = document.getElementById('zoom');
    dialog.querySelector('img').src = src;
    dialog.showModal();
  }
  const explainers = __EXPLAINERS__;
  if (explainers.length) {
    document.getElementById('fee-concept').hidden = false;
    document.getElementById('concept').innerHTML = explainers.map(src =>
      `<figure><img src="${esc(src)}" alt="Fee model step" loading="lazy" /></figure>`
    ).join('<div class="step-arrow">↓</div>');
  }
  const zoom = document.getElementById('zoom');
  document.getElementById('concept').addEventListener('click', e => {
    if (e.target.tagName !== 'IMG') return;
    zoom.querySelector('img').src = e.target.src;
    zoom.showModal();
  });
  document.getElementById('grid').addEventListener('click', e => {
    if (e.target.tagName !== 'IMG') return;
    zoom.querySelector('img').src = e.target.src;
    zoom.showModal();
  });
  zoom.addEventListener('click', () => zoom.close());
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
