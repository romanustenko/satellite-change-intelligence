"""Figures for the GitHub README / REPORT (both studies), written to ``assets/readme/``.

Reads only existing artifacts (no training, no test re-read): Research 1 HPO summaries, Research 2
run records, the frozen test analysis, the exploratory window-share check and the web-demo export
(SECOND validation examples).

Usage: python src/scripts/make_readme_figures.py --demo ../satellite-change-lab/public
"""

import argparse
import ast
import datetime as dt
import glob
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from export_r2_v2_web_demo import results_block  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/readme"
V2 = ROOT / "artifacts/research2/v2"
MODELS = {"siam_unet_resnet18": "Siamese U-Net + ResNet-18", "scannet": "SCanNet + ResNet-34",
          "mamba_scd": "MambaSCD-Tiny"}
COLORS = {"siam_unet_resnet18": "#4C78A8", "scannet": "#E45756", "mamba_scd": "#54A24B"}
CLASS_COLORS = np.array([[255, 255, 255], [176, 164, 146], [62, 112, 76], [150, 178, 98],
                         [70, 124, 168], [146, 78, 96], [214, 166, 64]], dtype=np.uint8)
CLASSES = ["no change", "non-veg. ground", "tree", "low vegetation", "water", "building",
           "playground"]

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print("written", OUT / name, flush=True)


def r1_hpo() -> None:
    """Validation F1 of every completed Research 1 HPO trial over time, per tier and regime."""
    tiers = {"fc_siam_diff": ("Light · FC-Siam-Diff", "#4C78A8"),
             "bit": ("Medium · BIT", "#F58518"),
             "siam_mit_b3_diff": ("Heavy · MiT-B3", "#54A24B")}
    points: dict[tuple[str, bool], list[tuple[dt.datetime, float]]] = {}
    for path in glob.glob(str(ROOT / "artifacts/hyperparameter_search/**/summary.json"),
                          recursive=True):
        data = json.loads(Path(path).read_text())
        model = next((p for p in Path(path).parts if p in tiers), None)
        if data.get("status") != "completed" or data.get("best_f1") is None or model is None:
            continue
        parameters = data["parameters"]
        if isinstance(parameters, str):
            parameters = ast.literal_eval(parameters)
        resized = parameters.get("image_size") == 256
        when = dt.datetime.fromtimestamp(Path(path).stat().st_mtime)
        points.setdefault((model, resized), []).append((when, data["best_f1"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    total = 0
    for (model, resized), values in sorted(points.items()):
        label, color = tiers[model]
        when, f1 = zip(*values, strict=True)
        total += len(values)
        ax.scatter(when, f1, s=12, alpha=0.65, color=color, marker="^" if resized else "o",
                   label=f"{label} · {'Resized 256' if resized else 'Full 1024'} ({len(values)})")
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("validation F1 (LEVIR-CD)")
    ax.set_title(f"Research 1 — {total} hyperparameter trials, run and inspected by hand")
    ax.legend(fontsize=8, ncol=2, loc="lower right", frameon=False)
    fig.autofmt_xdate()
    save(fig, "r1_hpo_trials.png")


def r2_training() -> None:
    """Validation SeK against training FLOPs: HPO trials and the three final seeds per model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for model, label in MODELS.items():
        for index, run in enumerate(sorted((V2 / "runs/final" / model).glob("*_s*"))):
            record = json.loads((run / "run_record.json").read_text())
            curve = record["curve"]
            axes[0].plot([c["training_flops"] for c in curve], [c["sek"] for c in curve],
                         color=COLORS[model], alpha=0.8, lw=1.2,
                         label=label if index == 0 else None)
        trials = []
        for run in sorted((V2 / "runs/hpo" / model).glob("*_hpo_t*")):
            record = json.loads((run / "run_record.json").read_text())
            trials.append((record["lr"], record["weight_decay"], record["best_sek"]))
        lr, wd, sek = zip(*trials, strict=True)
        axes[1].plot(sorted(lr), [s for _, s in sorted(zip(lr, sek, strict=True))],
                     color=COLORS[model], alpha=0.4, lw=1)
        axes[1].scatter(lr, sek, color=COLORS[model], s=[20 + 2000 * w for w in wd], alpha=0.85)
    axes[0].axvline(3.2e17, color="grey", ls=":", lw=1)
    axes[0].text(3.15e17, 0.08, "equal budget\n3.2·10¹⁷ FLOPs", ha="right", fontsize=8,
                 color="grey")
    axes[0].set_xlabel("training FLOPs")
    axes[0].set_ylabel("validation SeK")
    axes[0].set_title("Final training, 3 seeds per model")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("learning rate (marker size = weight decay)")
    axes[1].set_title("HPO: 6 trials × 8·10¹⁶ FLOPs per model")
    save(fig, "r2_training.png")


def r2_examples(demo: Path) -> None:
    """Three SECOND validation tiles (sparse / medium / dense): images, truth, both systems."""
    manifest = json.loads((demo / "demo/manifest.json").read_text())
    examples = {e["id"]: e for e in manifest["examples"]}
    chosen = ["03", "09", "15"]
    columns = ["Before (T1)", "After (T2)", "Ground truth (T2)", "Single-stage · SCanNet",
               "Two-stage · SCanNet + ROI"]
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(13, 8.2))
    for row, key in enumerate(chosen):
        folder = demo / "demo" / key
        layers = [np.asarray(Image.open(demo / "data" / f"{key}_t1.jpg").convert("RGB")),
                  np.asarray(Image.open(demo / "data" / f"{key}_t2.jpg").convert("RGB")),
                  CLASS_COLORS[np.asarray(Image.open(folder / "gt_t2.png"))],
                  CLASS_COLORS[np.asarray(Image.open(folder / "scannet_single_t2.png"))],
                  CLASS_COLORS[np.asarray(Image.open(folder / "scannet_two_stage_t2.png"))]]
        example = examples[key]
        for column, image in enumerate(layers):
            ax = axes[row, column]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(column >= 2)
                spine.set_color("#cccccc")
            if row == 0:
                ax.set_title(columns[column], fontsize=10)
            if column == 4:
                for x0, y0, x1, y1 in example["roi_boxes"]:
                    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                               ec="#222222", lw=1.2, ls="--"))
                ax.set_xlabel(f"processed by SCD: ROI boxes {example['roi_area']:.0%} of tile",
                              fontsize=8)
        axes[row, 0].set_ylabel(f"changed {example['density']:.0%}", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i] / 255, ec="#999999")
               for i in range(1, 7)]
    fig.legend(handles, CLASSES[1:], loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0.04, hspace=0.12, bottom=0.06)
    save(fig, "r2_examples.png")


def r2_tradeoff(results: dict) -> None:
    """Test SeK against forward GFLOPs per tile for every system (and the oracle diagnostic)."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    markers = {"single": ("o", "single-stage"), "case2__frontier0.5": ("s", "two-stage · case 2"),
               "case3__frontier0.5": ("D", "two-stage · case 3"),
               "case4__frontier0.5": ("^", "two-stage · case 4 (OR)"),
               "case2__frontier0.3": ("v", "two-stage · case 2, frontier0.3"),
               "oracle": ("*", "oracle ROI (diagnostic)")}
    for model, label in MODELS.items():
        systems = results["models"][model]["systems"]
        base = systems["single"]["gflops"]
        for key, (marker, _) in markers.items():
            system = systems[key]
            gflops = system.get("gflops") or system["flops_ratio"] * base
            ax.scatter(gflops, system["sek"], marker=marker, s=70 if marker == "*" else 40,
                       color=COLORS[model], edgecolor="white" if marker != "*" else COLORS[model],
                       facecolor="none" if key == "oracle" else COLORS[model], zorder=3)
        ax.annotate(label, (base, systems["single"]["sek"]), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=COLORS[model])
    for marker, name in markers.values():
        ax.scatter([], [], marker=marker, color="grey", label=name)
    ax.set_xlabel("forward GFLOPs per 512² tile (detector included)")
    ax.set_ylabel("test SeK")
    ax.set_title("Accuracy vs compute — every two-stage point lies right of its single stage")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "r2_accuracy_vs_flops.png")


def r2_density(results: dict) -> None:
    """ΔSeK (case 2 − single) per changed-pixel stratum with 95 % CI."""
    edges = results["density_edges"]
    names = [f"sparse\nρ < {edges[0]:.3f}", f"medium\n{edges[0]:.3f}–{edges[1]:.3f}",
             f"dense\nρ ≥ {edges[1]:.3f}"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.26
    for offset, (model, label) in enumerate(MODELS.items()):
        strata = results["models"][model]["strata"]
        diff = np.array([s["case2"]["diff"] for s in strata])
        ci = np.array([s["case2"]["ci95"] for s in strata])
        x = np.arange(3) + (offset - 1) * width
        ax.bar(x, diff, width, color=COLORS[model], label=label)
        ax.errorbar(x, diff, yerr=[diff - ci[:, 0], ci[:, 1] - diff], fmt="none",
                    ecolor="#333333", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-results["delta"], color="grey", lw=0.8, ls="--")
    ax.text(2.45, -results["delta"], " −δ", va="center", fontsize=8, color="grey")
    ax.set_xticks(range(3), names)
    ax.set_ylabel("ΔSeK two-stage − single")
    ax.set_title("Where the two-stage system gains or loses (test, 95 % CI)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "r2_density.png")


def r2_cost(window: dict) -> None:
    """FLOPs ratio against the changed share of the scene, SECOND tiles and LEVIR-CD scenes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sources = {"second_val__oracle": ("SECOND 512² (ground-truth boxes)", "-"),
               "levir_val__oracle": ("LEVIR-CD 1024² (ground-truth boxes)", "--"),
               "levir_val__r1_medium": ("LEVIR-CD 1024² (in-domain detector)", ":")}
    for key, (_, style) in sources.items():
        bands = window[key]["by_density"]
        x = [100 * (b["density"][0] + min(b["density"][1], 0.4)) / 2 for b in bands]
        for model in MODELS:
            ax.plot(x, [b[model] for b in bands], style, marker="o", ms=3, color=COLORS[model],
                    lw=1.3)
    ax.axhline(1, color="black", lw=0.9)
    ax.text(31, 0.93, "below this line two-stage is cheaper", fontsize=8, ha="right")
    for model, label in MODELS.items():
        ax.plot([], [], color=COLORS[model], label=label)
    for name, style in sources.values():
        ax.plot([], [], style, color="grey", label=name)
    ax.set_xlabel("changed share of the scene, % (band midpoint)")
    ax.set_ylabel("FLOPs two-stage / single")
    ax.set_title("Two-stage pays off only when changes are rare — image size does not matter")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    save(fig, "r2_when_cheaper.png")


THEMES = {"light": {"ink": "#1f2328", "muted": "#59636e", "scd": "#0969da", "det": "#bc4c00"},
          "dark": {"ink": "#e6edf3", "muted": "#9198a1", "scd": "#4493f8", "det": "#db6d28"}}


def architecture(demo: Path, key: str = "04") -> None:
    """Single-stage vs two-stage pipeline on one SECOND validation tile, transparent background,
    one variant per GitHub theme (used with a ``<picture>`` element)."""
    example = next(e for e in json.loads((demo / "demo/manifest.json").read_text())["examples"]
                   if e["id"] == key)
    folder = demo / "demo" / key
    t1, t2 = (np.asarray(Image.open(demo / "data" / f"{key}_{d}.jpg").convert("RGB"))
              for d in ("t1", "t2"))
    probability = np.asarray(Image.open(folder / "detector.png"), dtype=float) / 255

    def classes(name: str) -> np.ndarray:
        return CLASS_COLORS[np.asarray(Image.open(folder / f"{name}.png"))]

    for theme in THEMES:
        _architecture_figure(theme, example, t1, t2, probability, classes)


def _architecture_figure(theme: str, example: dict, t1: np.ndarray, t2: np.ndarray,
                         probability: np.ndarray, classes: Any) -> None:
    """One theme variant of the pipeline figure (see ``architecture``)."""
    c = THEMES[theme]
    fig = plt.figure(figsize=(12, 5.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 47)
    ax.set_aspect("equal")
    ax.axis("off")

    def tile(image: np.ndarray, x: float, y: float, layer: int = 2, **kw: Any) -> None:
        ax.imshow(image, extent=(x, x + 12, y, y + 12), zorder=layer, **kw)
        ax.add_patch(plt.Rectangle((x, y), 12, 12, fill=False, ec=c["muted"], lw=0.6,
                                   zorder=layer + 0.5))

    def pair(first: np.ndarray, second: np.ndarray, x: float, y: float, top: str,
             bottom: str) -> None:
        tile(first, x, y + 2.2, layer=2)
        tile(second, x + 2.2, y, layer=3)
        ax.text(x + 7.1, y + 15.6, top, ha="center", fontsize=9, color=c["muted"])
        ax.text(x + 7.1, y - 1.6, bottom, ha="center", va="top", fontsize=9,
                color=c["muted"])

    def stage(x: float, y: float, width: float, title: str, note: str, color: str) -> None:
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, y), width, 9, boxstyle="round,pad=0,rounding_size=1.2", fc=color,
            alpha=0.13, ec="none", zorder=1))
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, y), width, 9, boxstyle="round,pad=0,rounding_size=1.2", fill=False,
            ec=color, lw=1.4, zorder=2))
        ax.text(x + width / 2, y + 5.6, title, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=c["ink"])
        ax.text(x + width / 2, y + 2.6, note, ha="center", va="center", fontsize=8.5,
                color=c["muted"])

    def arrow(x0: float, x1: float, y: float) -> None:
        ax.annotate("", (x1, y), (x0, y), arrowprops={"arrowstyle": "-|>", "lw": 1.3,
                                                     "color": c["muted"]})

    rows = {"single": 29.5, "two": 4.5}
    for name, y in rows.items():
        label = "Single-stage" if name == "single" else "Two-stage"
        ax.text(0, y + 16.2, label, fontsize=13, fontweight="bold", color=c["ink"])
        pair(t1, t2, 0, y, "", "image pair T1 / T2")
        stage(60, y + 2.7, 17, "SCD model", "what changed into what", c["scd"])
        arrow(77.5, 82.5, y + 7.2)
        prefix = "scannet_single" if name == "single" else "scannet_two_stage"
        pair(classes(f"{prefix}_t1"), classes(f"{prefix}_t2"), 83, y, "",
             "class maps T1 / T2")
    arrow(15, 59.5, rows["single"] + 7.2)
    ax.text(37, rows["single"] + 8.4, "whole tile", ha="center", fontsize=9,
            color=c["muted"])
    y = rows["two"]
    arrow(15, 19.5, y + 7.2)
    stage(20, y + 2.7, 15, "Detector (BIT)", "where? (binary)", c["det"])
    arrow(35.5, 40.5, y + 7.2)
    tile(probability, 41, y + 1.2, cmap="magma", vmin=0, vmax=1)
    for x0, y0, x1, y1 in example["roi_boxes"]:
        scale = 12 / 512
        ax.add_patch(plt.Rectangle((41 + x0 * scale, y + 1.2 + 12 - y1 * scale),
                                   (x1 - x0) * scale, (y1 - y0) * scale, fill=False,
                                   ec="#ffffff", lw=1.0, ls="--", zorder=5))
    ax.text(47, y - 1.6, "probability + ROI boxes", ha="center", va="top", fontsize=9,
            color=c["muted"])
    arrow(53.5, 59.5, y + 7.2)
    ax.text(68.5, y + 0.6, "same weights, only ROI windows;\noutside = no change",
            ha="center", va="top", fontsize=8, color=c["muted"])
    fig.savefig(OUT / f"architecture_{theme}.png", transparent=True, dpi=170)
    plt.close(fig)
    print("written", OUT / f"architecture_{theme}.png", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--only", choices=["architecture"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    architecture(args.demo)
    if args.only:
        return
    results = results_block()
    window = json.loads((V2 / "window_share/window_share.json").read_text())
    r1_hpo()
    r2_training()
    r2_examples(args.demo)
    r2_tradeoff(results)
    r2_density(results)
    r2_cost(window)


if __name__ == "__main__":
    main()
