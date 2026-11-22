"""Export the web-demo examples (R2-W-T03, final models): masks only, no RGB tiles.

For 15 SECOND *validation* tiles chosen before any prediction is looked at (stratified by the
ground-truth change density: 5 quantile bins x 3 tiles, fixed seed) the script writes, per tile:

* ``gt_t1.png`` / ``gt_t2.png``      ground-truth class maps (pixel value = class id 0..6)
* ``detector.png``                   Medium (BIT fine-tuned, case 2) change probability x 255
* per SCD model and system (``single`` = whole tile, ``two_stage`` = case 2, frontier0.5 ROI policy,
  ``batch_ctx`` path): ``<model>_<system>_t1.png`` / ``_t2.png`` (class ids), ``_change.png``
  (change probability x 255) and ``_conf.png`` (class confidence = max softmax over classes 1..6,
  x 255), plus the ROI boxes and per-tile metrics in ``manifest.json``.

No RGB image is exported (SECOND redistribution licence unconfirmed, R2-W-T05); no weights are
exported. SCD checkpoints are the frozen finals of seed 42 (HPO winners, b* = 3.2e17, the
``best.pt`` named in ``v9/freeze_manifest_v2.json``); the detector is ``medium_ft`` seed 42 (D-075).

Verification: every exported tile is checked against the frozen validation evaluation
(``v9/eval/val/<model>_s42.npz``): the pooled 7x7 histogram of the exported single-stage and
two-stage maps and the ROI box area must equal the stored rows. The outcome (exact or the number of
differing histogram counts, e.g. from CPU vs GPU arithmetic) is written to ``manifest.json``.

The manifest also carries ``results``: dataset-level test numbers (frozen protocol v0.4) copied
from ``v9/analysis_test.json`` and ``v9/latency_val.json`` for the site's summary tables; no number
is typed by hand. Runs on CPU by default.

Usage: python src/scripts/export_r2_v2_web_demo.py --out <site>/public/demo [--models ...]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_r2_v2_two_stage_speedups as bench  # noqa: E402
import eval_r2_v2_two_stage_exploratory as base  # noqa: E402
import r2_v2_systems_lib as lib  # noqa: E402

from satchange.data.levir_cd import create_dataset_dataloader  # noqa: E402
from satchange.semantic.freeze import sha256_file  # noqa: E402
from satchange.semantic.second import SECOND_CLASSES  # noqa: E402
from satchange.semantic.system_eval import load_semantic_checkpoint  # noqa: E402

SEED = 42
FINAL_RUNS = {name: lib.scd_run_dir(name, SEED) for name in lib.SCD_MODELS}
LABELS = {"siam_unet_resnet18": "Siamese U-Net", "scannet": "SCanNet", "mamba_scd": "MambaSCD"}
DETECTOR = ("medium_ft", SEED)
# case-2 area-budget policy declared on validation (roi_policies.json); "balanced" covers
# almost every tile with one full-tile box, which shows no difference in a demo
POLICY_NAME = "frontier0.5"
POLICY = lib.policies_for("case2")[POLICY_NAME]
SELECTION_SEED = 2026
BINS, PER_BIN = 5, 3


def select_tiles(densities: np.ndarray) -> list[int]:
    """Indices of 15 tiles: quantile bins of GT change density, 3 random tiles per bin."""
    order = np.argsort(densities, kind="stable")
    rng = np.random.default_rng(SELECTION_SEED)
    chosen: list[int] = []
    for part in np.array_split(order, BINS):
        chosen.extend(int(i) for i in sorted(rng.choice(part, size=PER_BIN, replace=False)))
    return chosen


def verdict_code(system: dict, comparison: dict, delta: float) -> str:
    """Decision rule of ``make_r2_v2_results.verdict`` (frozen protocol v0.4) as a short code."""
    non_inferior = comparison["ci95"][0] > -delta
    cheaper = system["flops_ratio_ci95"][1] < 1
    better, worse = comparison["ci95"][0] > 0, comparison["ci95"][1] < 0
    dearer = system["flops_ratio_ci95"][0] > 1
    if non_inferior and cheaper:
        return "two_stage"
    if (cheaper and not non_inferior) or (better and dearer):
        return "trade_off"
    if (worse and dearer) or (non_inferior and dearer):
        return "single"
    return "inconclusive"


def results_block() -> dict:
    """Test-split summary for the site (copied from the frozen analysis files)."""
    test = json.loads((lib.V9 / "analysis_test.json").read_text())
    latency = json.loads((lib.V9 / "latency_val.json").read_text())
    delta, policy = float(test["delta"]), str(test["primary_policy"])
    models = {}
    for name in lib.SCD_MODELS:
        entry = test["models"][name]
        single = entry["systems"]["single"]
        systems = {"single": {
            "sek": single["sek"]["mean"], "sek_sd": single["sek"]["sd"],
            "miou": single["miou"]["mean"], "fscd": single["fscd"]["mean"],
            "flops_ratio": 1.0, "gflops": single["flops_per_tile"] / 1e9,
            "latency_ms": latency[name]["single"]["median_ms"], "latency_ratio": 1.0,
        }}
        for key in [*(f"{case}__{policy}" for case in lib.CASES), "case2__frontier0.3"]:
            system = entry["systems"][key]
            comparison = entry["comparisons"][f"{key}_vs_single"]
            timing = latency[name].get(key)
            systems[key] = {
                "sek": system["sek"]["mean"], "miou": system["miou"]["mean"],
                "fscd": system["fscd"]["mean"], "diff": comparison["diff_sek"],
                "diff_ci95": comparison["ci95"], "non_inferior": comparison["non_inferior"],
                "flops_ratio": system["flops_ratio_vs_single"],
                "flops_ratio_ci95": system["flops_ratio_ci95"],
                "gflops": system["flops_per_tile"] / 1e9,
                "box_area": system["box_area_ratio"], "window_area": system["window_area_ratio"],
                "latency_ms": timing["median_ms"] if timing else None,
                "latency_ratio": timing["ratio_vs_single"]["median"] if timing else None,
                "verdict": verdict_code(system, comparison, delta),
            }
        oracle = entry["systems"]["oracle__balanced"]
        systems["oracle"] = {"sek": oracle["sek"]["mean"],
                             "diff": oracle["sek"]["mean"] - single["sek"]["mean"],
                             "flops_ratio": oracle["flops_ratio_vs_single"]}
        strata = [{"tiles": entry["strata"][str(k)]["tiles"],
                   **{case: entry["strata"][str(k)][f"{case}_minus_single"]
                      for case in lib.CASES}} for k in range(3)]
        models[name] = {"systems": systems, "strata": strata}
    return {"split": "SECOND test", "tiles": test["tiles"], "seeds": list(lib.SEEDS),
            "delta": delta, "primary_policy": policy, "density_edges": test["density_edges"],
            "latency_split": "SECOND validation, batch 1, fp32, A100", "models": models}


def save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(array.astype(np.uint8)), mode="L").save(path,
                                                                                optimize=True)


def to_u8(probability: torch.Tensor) -> np.ndarray:
    return (probability.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


class CapturingMerger(bench.Merger):
    """Merger that keeps its last instance (averaged probabilities are read after ``scene``)."""

    last: "CapturingMerger | None" = None

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        CapturingMerger.last = self


def run_system(runner, sample, boxes, device, *, single: bool):
    """(T1, T2, change probability, class confidence) maps [512,512] for one tile."""
    before, after = sample["before"].to(device), sample["after"].to(device)
    if single:
        probability, log_prob = bench.forward_batch(runner, before[None], after[None])
        change, log = probability[0], log_prob[0]
        first, second = base._decode(change, log)
    else:
        (first, second), _ = bench.scene("batch_ctx", runner, sample, boxes, device)
        merger = CapturingMerger.last
        assert merger is not None
        seen = merger.covered.clamp(min=1)
        change, log = merger.change / seen, merger.log / seen
        change = change * (merger.covered > 0)
    confidence = log[:, 1:].exp().amax(dim=1)  # [2,H,W]: per-date class confidence
    changed = (first > 0) | (second > 0)
    confidence = torch.where(changed, confidence.amax(dim=0), torch.zeros_like(change))
    return first, second, change, confidence


def scores(first, second, label_1, label_2) -> dict[str, float]:
    """Per-tile pixel agreement numbers (descriptive only; dataset-level metrics are elsewhere)."""
    predicted = (first > 0) | (second > 0)
    truth = (label_1 > 0) | (label_2 > 0)
    tp = float((predicted & truth).sum())
    union = float((predicted | truth).sum())
    correct = float(((first == label_1) & (second == label_2) & truth).sum())
    return {
        "change_iou": tp / union if union else 1.0,
        "changed_class_accuracy": correct / float(truth.sum()) if truth.any() else 1.0,
        "predicted_area": float(predicted.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=list(FINAL_RUNS))
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset, _ = create_dataset_dataloader(lib.DATA, "val", None, 1, 0, semantic_labels=True)
    masks = [dataset[i]["mask"] for i in range(len(dataset))]
    densities = np.array([float((m > 0.5).float().mean()) for m in masks])
    chosen = select_tiles(densities)
    detector = np.load(lib.map_path("val", *DETECTOR), mmap_mode="r")
    if detector.shape[0] != len(dataset):
        raise ValueError("Detector maps do not match the validation tile count.")

    args.out.mkdir(parents=True, exist_ok=True)
    samples = {index: dataset[index] for index in chosen}
    examples = []
    for rank, index in enumerate(chosen):
        sample = samples[index]
        folder = args.out / f"{rank + 1:02d}"
        folder.mkdir(exist_ok=True)
        save_png(folder / "gt_t1.png", sample["before_label"].numpy())
        save_png(folder / "gt_t2.png", sample["after_label"].numpy())
        probabilities = np.asarray(detector[index], dtype=np.float32)
        save_png(folder / "detector.png", np.round(np.clip(probabilities, 0, 1) * 255))
        boxes = base._boxes(probabilities, POLICY)
        examples.append({
            "id": f"{rank + 1:02d}", "sample_id": str(dataset.sample_ids[index]),
            "density": densities[index], "density_bin": rank // PER_BIN,
            "roi_boxes": boxes.tolist(),
            "roi_area": float(sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes.tolist()))
            / lib.TILE ** 2,
            "systems": {},
        })

    models_meta = []
    verification: dict[str, dict] = {}
    for name in args.models:
        run = FINAL_RUNS[name]
        checkpoint = run / "best.pt"
        evaluated = json.loads((lib.V9 / "eval/val" / f"{name}_s{SEED}.json").read_text())
        if evaluated["checkpoint_sha256"] != sha256_file(checkpoint):
            raise ValueError(f"{name}: checkpoint differs from the one evaluated on validation")
        stored = np.load(lib.V9 / "eval/val" / f"{name}_s{SEED}.npz")
        row_of = {str(sample_id): row for row, sample_id in enumerate(stored["ids"])}
        checks = {"tiles": 0, "exact": 0, "differing_counts": 0, "box_area_equal": 0}
        started = time.perf_counter()
        model = load_semantic_checkpoint(checkpoint, device)
        runner = bench.Runner(model, bf16=False)
        bench.Merger = CapturingMerger  # scene() builds its merger through the module attribute
        for rank, index in enumerate(chosen):
            sample, entry = samples[index], examples[rank]
            folder = args.out / entry["id"]
            boxes = np.asarray(entry["roi_boxes"], dtype=np.int64).reshape(-1, 4)
            for system in ("single", "two_stage"):
                with torch.inference_mode():
                    first, second, change, confidence = run_system(
                        runner, sample, boxes, device, single=system == "single")
                stem = f"{name}_{system}"
                save_png(folder / f"{stem}_t1.png", first.cpu().numpy())
                save_png(folder / f"{stem}_t2.png", second.cpu().numpy())
                save_png(folder / f"{stem}_change.png", to_u8(change))
                save_png(folder / f"{stem}_conf.png", to_u8(confidence))
                entry["systems"][stem] = scores(first.cpu(), second.cpu(),
                                                sample["before_label"], sample["after_label"])
                key = "single" if system == "single" else f"case2__{POLICY_NAME}"
                row = row_of[entry["sample_id"]]
                hist = base._hist((first.cpu(), second.cpu()),
                                  (sample["before_label"], sample["after_label"]))
                differing = int(np.abs(hist - stored[f"hist__{key}"][row]).sum())
                checks["tiles"] += 1
                checks["exact"] += differing == 0
                checks["differing_counts"] += differing
                box_area = int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum())
                checks["box_area_equal"] += (system == "single"
                                             or box_area == int(stored[f"box_area__{key}"][row]))
            print(name, f"{rank + 1}/{len(chosen)}", f"{time.perf_counter() - started:.0f}s",
                  flush=True)
        verification[name] = checks
        print(name, "verification", checks, flush=True)
        if checks["box_area_equal"] != checks["tiles"]:
            raise ValueError(f"{name}: ROI boxes differ from the frozen validation evaluation")
        models_meta.append({
            "id": name, "label": LABELS[name], "run": run.name, "pilot": False, "seed": SEED,
            "checkpoint_sha256": evaluated["checkpoint_sha256"],
        })
        del model, runner

    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=lib.ROOT, check=True).stdout.strip()
    manifest = {
        "version": 2, "kind": "satchange_web_demo", "pilot": False, "images_included": False,
        "split": "SECOND validation", "tile_size": lib.TILE,
        "selection": {"rule": "GT change-density quantile bins, random tiles per bin",
                      "bins": BINS, "per_bin": PER_BIN, "seed": SELECTION_SEED},
        "classes": ["no change", *SECOND_CLASSES],
        "detector": {"variant": DETECTOR[0], "seed": DETECTOR[1],
                     "label": "BIT (Medium), LEVIR-CD -> SECOND fine-tune"},
        "roi_policy": {"name": POLICY_NAME, **POLICY._asdict()} if hasattr(POLICY, "_asdict")
        else {"name": POLICY_NAME, **vars(POLICY)},
        "models": models_meta, "examples": examples, "verification": verification,
        "results": results_block(), "git_revision": revision,
        "device": str(device), "torch": torch.__version__,
    }
    temporary = args.out / "manifest.json.tmp"
    with temporary.open("w") as handle:
        json.dump(manifest, handle, indent=1, default=float)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.out / "manifest.json")
    print("written", args.out / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
