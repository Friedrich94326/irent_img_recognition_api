#!/usr/bin/env python
"""Train an XGBoost repair-fee model on damage features; used by build_demo_page.py.

There is no real history linking damage photos to repair invoices yet, so by default the model is
trained on *simulated* past cases: damages drawn with Hotai-like class frequencies, sized against
a tyre, priced by the repair a body shop would do (touch-up, repaint, dent repair, replacement...)
plus an impact premium and noise. The price ranges are tuned so single repairs land near the
body-repair rows in irent-project's repair_orders table (printed as a check).

Replace the simulation with real records by passing --data: a CSV with the FEATURES columns
(empty cell = missing) plus ``cost`` in NT$.

Usage:
    python scripts/repair_fee_model.py
    python scripts/repair_fee_model.py --data path/to/past_repairs.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

CLASSES = ("dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat", "missing_part")
FEATURES = (
    [f"n_{c}" for c in CLASSES]
    + [f"max_share_{c}" for c in CLASSES]
    + ["union_share", "n_detections", "impact", "n_impacts"]
)
MODEL_PATH = Path("weights/repair_fee_xgb.json")
SIMULATED_CSV = Path("Datasets/repair_fee_simulated.csv")

# --- features --------------------------------------------------------------------------------


def features(measured, area: dict, impacts: list) -> list[float]:
    """Feature row for one photo from build_demo_page's Measured rows, damaged_area and impacts.

    Shares are NaN when no tyre gave a scale; XGBoost routes missing values natively.
    """
    counts = dict.fromkeys(CLASSES, 0)
    shares: dict[str, float] = dict.fromkeys(CLASSES, math.nan)
    for m in measured:
        if m.damage_class not in counts:
            continue
        counts[m.damage_class] += 1
        if m.tyre_share is not None:
            prev = shares[m.damage_class]
            shares[m.damage_class] = m.tyre_share if math.isnan(prev) else max(prev, m.tyre_share)
    union = area.get("union_share")
    return (
        [counts[c] for c in CLASSES]
        + [shares[c] for c in CLASSES]
        + [math.nan if union is None else union, len(measured), int(bool(impacts)), len(impacts)]
    )


# --- simulated history -----------------------------------------------------------------------

PRICES = {  # NT$ (low, high) per repair method; placeholders until real invoices exist
    "touch_up": (800, 1500),
    "partial_repaint": (2500, 4500),
    "full_repaint": (4500, 7500),
    "paintless_dent": (1500, 4000),
    "panel_beating": (5000, 9000),
    "panel_replacement": (9000, 18000),
    "plastic_weld": (2700, 6000),
    "lamp": (3500, 9000),
    "glass": (8000, 20000),
    "tyre": (3000, 5500),
    "part": (2500, 10000),
}
BAND_METHODS = {  # class -> method for small / medium / large (share of tyre area)
    "scratch": ("touch_up", "partial_repaint", "full_repaint"),
    "dent": ("paintless_dent", "panel_beating", "panel_replacement"),
    "crack": ("plastic_weld", "plastic_weld", "panel_replacement"),
}
FIXED_METHODS = {  # size-independent jobs: class -> (method, share at which price tops out)
    "lamp_broken": ("lamp", 1.0),
    "glass_shatter": ("glass", 2.0),
    "tire_flat": ("tyre", math.inf),
    "missing_part": ("part", 0.5),
}
SMALL, LARGE = 0.05, 0.25
CLASS_P = {"scratch": .45, "dent": .25, "crack": .07, "lamp_broken": .08, "glass_shatter": .04,
           "tire_flat": .05, "missing_part": .06}
SHARE_MEDIAN = {"scratch": .18, "dent": .22, "crack": .3, "lamp_broken": .45, "glass_shatter": 1.5,
                "tire_flat": 1.0, "missing_part": .3}


def job_cost(damage_class: str, share: float) -> float:
    """Price of the repair a shop would do for one damage, positioned within its range by size."""
    if damage_class in BAND_METHODS:
        if share < SMALL:
            band, pos = 0, share / SMALL
        elif share <= LARGE:
            band, pos = 1, (share - SMALL) / (LARGE - SMALL)
        else:
            band, pos = 2, (share - LARGE) / 0.75
        method = BAND_METHODS[damage_class][band]
    else:
        method, top = FIXED_METHODS[damage_class]
        pos = 0.5 if math.isinf(top) else share / top
    low, high = PRICES[method]
    return low + (high - low) * min(max(pos, 0.0), 1.0)


def simulate(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, probs = list(CLASS_P), np.array(list(CLASS_P.values()))
    rows, costs = [], []
    for _ in range(n):
        n_damage = int(min(5, 1 + rng.poisson(0.9)))
        n_panels = int(rng.integers(1, min(3, n_damage) + 1))
        damages = [
            (c, float(SHARE_MEDIAN[c] * rng.lognormal(0, 0.8)), int(rng.integers(n_panels)))
            for c in rng.choice(classes, size=n_damage, p=probs / probs.sum())
        ]
        cost, n_impacts, union = 0.0, 0, 0.0
        for panel in range(n_panels):
            on_panel = [(c, s) for c, s, p in damages if p == panel and c != "tire_flat"]
            if not on_panel:
                continue
            kinds = {c for c, _ in on_panel}
            shares = sorted((s for _, s in on_panel), reverse=True)
            panel_union = shares[0] + 0.5 * sum(shares[1:])  # damage on one panel overlaps
            union += panel_union
            cost += max(job_cost(c, s) for c, s in on_panel)  # one job per panel
            if (kinds & {"dent", "crack"} and kinds & {"crack", "lamp_broken", "glass_shatter",
                                                         "missing_part"} and len(on_panel) >= 2) \
                    or (len(on_panel) >= 3 and len(kinds) >= 2) or panel_union >= 1.0:
                n_impacts += 1
                cost += rng.uniform(3000, 8000)  # structure / extra labour after a collision
        cost += sum(job_cost(c, s) for c, s, _ in damages if c == "tire_flat")
        cost = round(cost * rng.lognormal(0, 0.15) / 50) * 50

        counts = {c: sum(1 for d in damages if d[0] == c) for c in CLASSES}
        max_share = {c: max((s for d, s, _ in damages if d == c), default=math.nan)
                     for c in CLASSES}
        if rng.random() < 0.15:  # no tyre in the photo: sizes unknown
            max_share = dict.fromkeys(CLASSES, math.nan)
            union = math.nan
        rows.append([counts[c] for c in CLASSES] + [max_share[c] for c in CLASSES]
                    + [union, n_damage, int(n_impacts > 0), n_impacts])
        costs.append(cost)
    return np.array(rows, dtype=float), np.array(costs, dtype=float)


# --- io / training ---------------------------------------------------------------------------


def write_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([*FEATURES, "cost"])
        for row, cost in zip(x, y, strict=True):
            w.writerow(["" if math.isnan(v) else f"{v:g}" for v in row] + [f"{cost:.0f}"])


def read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    missing = [c for c in [*FEATURES, "cost"] if c not in records[0]]
    if missing:
        raise SystemExit(f"{path} lacks columns: {', '.join(missing)}")
    x = [[float(r[c]) if r[c] != "" else math.nan for c in FEATURES] for r in records]
    return np.array(x, dtype=float), np.array([float(r["cost"]) for r in records])


def load_model(path: Path = MODEL_PATH):
    import xgboost as xgb  # noqa: PLC0415 - optional dependency (requirements-fee.txt)

    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def predict_fee(model, row: list[float]) -> float:
    return float(np.expm1(model.predict(np.array([row], dtype=float))[0]))


def single(damage_class: str, share: float) -> list[float]:
    """Feature row for a photo with exactly one damage of the given size."""
    row = dict.fromkeys(FEATURES, 0.0)
    for c in CLASSES:
        row[f"max_share_{c}"] = math.nan
    row[f"n_{damage_class}"] = 1
    row[f"max_share_{damage_class}"] = share
    row["union_share"] = 0.0 if damage_class == "tire_flat" else share
    row["n_detections"] = 1
    return [row[f] for f in FEATURES]


CALIBRATION = [  # body-repair rows in irent-project/data/irent.sqlite repair_orders
    ("右後保桿鈑噴 (bumper repaint)", "scratch", 0.20, 4200),
    ("左側車門鈑金 (door panel beating)", "dent", 0.20, 7800),
    ("右後燈更換 (tail lamp replacement)", "lamp_broken", 0.50, 6500),
    ("左前燈修復 (headlamp repair)", "lamp_broken", 0.10, 3600),
    ("後視鏡更換 (mirror replacement)", "missing_part", 0.20, 4100),
    ("輪胎更換 (tyre replacement)", "tire_flat", 1.00, 4400),
]


def split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train / held-out indices (80 / 20); shared by training and the explainer plots."""
    order = np.random.default_rng(seed).permutation(n)
    cut = int(0.8 * n)
    return order[:cut], order[cut:]


# --- explainer pictures ----------------------------------------------------------------------

READABLE = {
    **{f"n_{c}": f"Number of {c.replace('_', ' ')}s" for c in CLASSES},
    **{f"max_share_{c}": f"Largest {c.replace('_', ' ')} size" for c in CLASSES},
    "n_tire_flat": "Number of flat tyres",
    "max_share_tire_flat": "Largest flat tyre size",
    "n_glass_shatter": "Number of shattered glass",
    "n_missing_part": "Number of missing parts",
    "union_share": "Total damaged area",
    "n_detections": "Number of damages",
    "impact": "Impact found",
    "n_impacts": "Number of impacts",
}
# Reference dataviz palette (validated: series blue + highlight orange), light slide surface.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES, HIGHLIGHT = "#2a78d6", "#eb6834"


def _style():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import font_manager  # noqa: PLC0415

    family = "DejaVu Sans"
    font = Path("C:/Windows/Fonts/msjh.ttc")  # Microsoft JhengHei: renders the Chinese labels
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID, "xtick.color": INK_2, "ytick.color": INK_2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _share(v: float) -> str:
    return "–" if math.isnan(v) else f"{v:.0%}"


def plot_history(x: np.ndarray, y: np.ndarray, path: Path, rows: int = 8) -> None:
    plt = _style()
    col = {f: i for i, f in enumerate(FEATURES)}
    # Cheap -> expensive cases with different damage mixes, plus one photo without a tyre.
    order = np.argsort(y)
    picks: list[int] = []
    seen: set[tuple] = set()
    for q in np.linspace(0.04, 0.97, rows * 6):
        i = int(order[int(q * (len(y) - 1))])
        mix = tuple(x[i, col[f"n_{c}"]] > 0 for c in CLASSES)
        if mix not in seen and not math.isnan(x[i, col["union_share"]]):
            seen.add(mix)
            picks.append(i)
    picks = [picks[int(k)] for k in np.linspace(0, len(picks) - 1, rows - 1)]
    no_tyre = np.flatnonzero(np.isnan(x[:, col["union_share"]]))
    picks.append(int(no_tyre[np.argmin(np.abs(y[no_tyre] - np.median(y)))]))
    picks.sort(key=lambda i: y[i])
    header = ["Case", "Dents", "Scratches", "Lamps", "Flat tyres", "Other",
              "Largest damage\n(% of tyre)", "Damaged area\n(% of tyre)", "Impact", "Cost (NT$)"]
    cells = []
    for n, i in enumerate(picks, 1):
        r = x[i]
        other = r[col["n_crack"]] + r[col["n_glass_shatter"]] + r[col["n_missing_part"]]
        shares = [r[col[f"max_share_{c}"]] for c in CLASSES]
        largest = math.nan if all(math.isnan(s) for s in shares) else np.nanmax(shares)
        cells.append([f"#{n}", *(f"{r[col[k]]:.0f}" for k in
                                 ("n_dent", "n_scratch", "n_lamp_broken", "n_tire_flat")),
                      f"{other:.0f}", _share(largest), _share(r[col["union_share"]]),
                      "yes" if r[col["impact"]] else "no", f"{y[i]:,.0f}"])

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    widths = [0.07, 0.08, 0.09, 0.08, 0.09, 0.08, 0.14, 0.14, 0.08, 0.11]
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center",
                     colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=INK, weight="bold")
            cell.set_facecolor("#f0efea")
            cell.set_height(cell.get_height() * 1.5)
        elif c == len(header) - 1:
            cell.set_text_props(color=INK, weight="bold")
            cell.set_facecolor("#fdeee7")  # the label (target) column
        elif c == len(header) - 2 and cells[r - 1][c] == "yes":
            cell.set_text_props(color="#c1410c", weight="bold")
    fig.suptitle("① Input: past repair cases (features → cost)", x=0.02, ha="left",
                 fontsize=16, weight="bold")
    fig.text(0.02, 0.87, f"{rows} of {len(y):,} simulated cases · each row = damage found in the "
             "photos + what the repair cost · “–” = no tyre in view, size unknown",
             color=INK_2, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _top_split(tree_dump: str) -> tuple[str, str]:
    """Readable root condition of one tree from Booster.get_dump(), e.g. ('Impact found', '< 1')."""
    head = tree_dump.splitlines()[0]
    cond = head[head.index("[") + 1:head.index("]")]
    name, value = cond.split("<")
    feature = FEATURES[int(name[1:])] if name.startswith("f") and name[1:].isdigit() else name
    v = float(value)
    shown = f"{v:.0%}" if "share" in feature else ("yes?" if feature == "impact" else f"≥ {v:g}")
    return READABLE.get(feature, feature), shown


def plot_model(model, path: Path) -> None:
    plt = _style()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: PLC0415

    booster = model.get_booster()
    dump = booster.get_dump()
    fig = plt.figure(figsize=(14, 5.4))
    ax = fig.add_axes((0.0, 0.0, 0.56, 0.86))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x0, y0, w, h, text, face, ink=INK, size=10.5, bold=False):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=face, edgecolor=GRID, linewidth=1))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", color=ink, fontsize=size,
                weight="bold" if bold else "normal")

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=MUTED,
                                     linewidth=1.4))

    box(1, 24, 15, 12, "Damage features\nof one photo", "#eef4fc", bold=True)
    trees = [(0, "Tree 1"), (1, "Tree 2"), (len(dump) - 1, f"Tree {len(dump)}")]
    for k, (idx, title) in enumerate(trees):
        cx, top = 30 + k * 20, 52
        name, value = _top_split(dump[idx])
        ax.text(cx, top + 5, title, ha="center", color=INK_2, fontsize=10, weight="bold")
        box(cx - 8.5, top - 6, 17, 7, f"{name}\n{value}", SURFACE, size=9)
        for dx, lab in ((-5, "yes"), (5, "no")):
            ax.plot([cx, cx + dx], [top - 6.4, top - 16], color=MUTED, linewidth=1.2)
            ax.text(cx + dx * 0.55, top - 11, lab, fontsize=8, color=MUTED, ha="center")
            ax.add_patch(FancyBboxPatch((cx + dx - 3.2, top - 21), 6.4, 5,
                                        boxstyle="round,pad=0.3,rounding_size=1",
                                        facecolor="#f0efea", edgecolor=GRID))
            ax.text(cx + dx, top - 18.5, "…", ha="center", va="center", color=INK_2)
        arrow((cx, top - 22), (cx, 18))
        if k == 1:
            ax.text(cx + 10, top - 10, "···", fontsize=22, color=MUTED, ha="center", va="center")
    arrow((16.8, 32), (21.2, 48))
    ax.text(50, 12.5, f"Σ  add up all {len(dump)} trees' small corrections", ha="center",
            fontsize=11.5, weight="bold")
    box(34, 1, 32, 7, "→  predicted repair fee (NT$)", "#fdeee7", ink="#c1410c", bold=True)
    fig.text(0.02, 0.9, "Each tree asks a few yes/no questions and nudges the price; "
             "the next tree learns from the previous trees' mistakes.", color=INK_2, fontsize=11)

    gain = booster.get_score(importance_type="gain")
    names = {f"f{i}": f for i, f in enumerate(FEATURES)}
    top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:8]
    total = sum(gain.values())
    labels = [READABLE.get(names.get(k, k), names.get(k, k)) for k, _ in top][::-1]
    values = [v / total for _, v in top][::-1]
    bx = fig.add_axes((0.72, 0.1, 0.26, 0.72))
    bars = bx.barh(labels, values, height=0.55, color=SERIES)
    for bar, v in zip(bars, values, strict=True):
        bx.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.0%}",
                va="center", color=INK, fontsize=10)
    bx.set_xticks([])
    bx.spines["bottom"].set_visible(False)
    bx.tick_params(axis="y", length=0, labelsize=10.5)
    bx.set_title("What drives the price (share of gain)", loc="left", fontsize=12,
                 weight="bold", color=INK, pad=10)
    fig.suptitle("② Model: XGBoost — gradient-boosted decision trees", x=0.02, ha="left",
                 fontsize=16, weight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regression(model, x: np.ndarray, y: np.ndarray, test: np.ndarray, path: Path,
                    demo_fees: list[tuple[str, float]] | None = None) -> None:
    plt = _style()
    pred = np.expm1(model.predict(x[test]))
    actual = y[test]
    mae = np.mean(np.abs(pred - actual))
    mape = np.mean(np.abs(pred - actual) / actual)
    base = np.mean(np.abs(np.median(y) - actual))

    widths = (1.35, 1) if demo_fees else (1,)
    fig, axes = plt.subplots(1, len(widths), figsize=(13 if demo_fees else 7.5, 5.8),
                             gridspec_kw={"width_ratios": widths})
    ax = axes[0] if demo_fees else axes
    hi = float(np.percentile(np.concatenate([pred, actual]), 99.5)) * 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(hi * 0.8, hi * 0.97, "perfect prediction", color=MUTED, fontsize=9.5, ha="right",
            va="top")
    ax.scatter(actual, pred, s=22, color=SERIES, alpha=0.55, edgecolors=SURFACE, linewidths=0.6,
               zorder=2)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Actual repair cost (NT$)")
    ax.set_ylabel("Predicted by XGBoost (NT$)")
    fmt = plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{len(test)} held-out cases the model never saw", loc="left", fontsize=12,
                 weight="bold", pad=10)
    ax.text(0.03, 0.97, f"Average error  NT${mae:,.0f}  ({mape:.0%})\n"
            f"Guessing the median  NT${base:,.0f}", transform=ax.transAxes, va="top",
            fontsize=10.5, color=INK, bbox={"boxstyle": "round,pad=0.5", "facecolor": "#ffffff",
                                            "edgecolor": GRID})

    if demo_fees:
        dx = axes[1]
        labels = [label for label, _ in demo_fees][::-1]
        fees = [fee for _, fee in demo_fees][::-1]
        bars = dx.barh(labels, fees, height=0.55, color=HIGHLIGHT)
        for bar, fee in zip(bars, fees, strict=True):
            dx.text(bar.get_width() + max(fees) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"NT${fee:,.0f}", va="center", color=INK, fontsize=10.5)
        dx.set_xlim(0, max(fees) * 1.3)
        dx.set_xticks([])
        dx.spines["bottom"].set_visible(False)
        dx.tick_params(axis="y", length=0, labelsize=10.5)
        dx.set_title("Applied to the demo photos", loc="left", fontsize=12, weight="bold",
                     pad=10)
    fig.suptitle("③ Output: regression — a fee in NT$ for any new damage photo", x=0.02,
                 ha="left", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_explainers(model, out_dir: Path, seed: int = 42,
                    demo_fees: list[tuple[str, float]] | None = None) -> list[Path]:
    """Write the three design-concept pictures (input, model, output) from the training table."""
    x, y = read_csv(SIMULATED_CSV)
    _, test = split(len(y), seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "fee_1_history.png", out_dir / "fee_2_xgboost.png",
             out_dir / "fee_3_regression.png"]
    plot_history(x, y, paths[0])
    plot_model(model, paths[1])
    plot_regression(model, x, y, test, paths[2], demo_fees)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, help="real CSV (FEATURES + cost); default: simulate")
    parser.add_argument("--n", type=int, default=3000, help="simulated cases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    parser.add_argument("--plots", type=Path, help="also write the explainer PNGs to this folder")
    args = parser.parse_args()

    import xgboost as xgb  # noqa: PLC0415

    if args.data:
        x, y = read_csv(args.data)
        print(f"Loaded {len(y)} real cases from {args.data}")
    else:
        x, y = simulate(args.n, args.seed)
        write_csv(SIMULATED_CSV, x, y)
        print(f"Simulated {len(y)} cases -> {SIMULATED_CSV} "
              f"(median NT${np.median(y):,.0f}, p90 NT${np.percentile(y, 90):,.0f})")

    train, test = split(len(y), args.seed)
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.9, random_state=args.seed,
    )
    model.fit(x[train], np.log1p(y[train]))
    pred = np.expm1(model.predict(x[test]))
    mae = np.mean(np.abs(pred - y[test]))
    mape = np.mean(np.abs(pred - y[test]) / y[test])
    base = np.mean(np.abs(np.median(y[train]) - y[test]))
    print(f"Held-out {len(test)} cases: MAE NT${mae:,.0f} (MAPE {mape:.0%}) "
          f"vs NT${base:,.0f} for always predicting the median")

    print(f"\n{'repair order':<36}{'actual':>9}{'model':>9}")
    for label, cls, share, actual in CALIBRATION:
        print(f"{label:<36}{actual:>9,}{predict_fee(model, single(cls, share)):>9,.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out)
    print(f"\nSaved {args.out}")
    if args.plots and not args.data:
        for path in plot_explainers(model, args.plots, args.seed):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
