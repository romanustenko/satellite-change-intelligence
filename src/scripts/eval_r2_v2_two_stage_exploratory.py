"""Exploratory single-stage vs two-stage SCD comparison on SECOND validation (user request 2026-11-13).

Single stage = one SCD model on the whole 512x512 tile. Two stage = Research 1 Medium (BIT)
detector fine-tuned on SECOND (screening finals, saved probability maps) -> ROI boxes from the
screening generator -> the SAME SCD checkpoint applied to every crop -> crops merged back into
native-resolution T1/T2 maps (outside every ROI = 0 = no change). Nothing is trained or tuned here:
ROI policies are the three fixed illustrative settings of the screening report and the SCD
checkpoints are pilot checkpoints trained on whole tiles (case-1 models, D-050 option "reuse"). Both
variants are evaluated on the SAME 444 validation tiles; the test split is never read. Results are
exploratory (not a frozen-protocol result, pilot lr/budget, checkpoints picked by validation SeK).

Usage: python src/scripts/eval_r2_v2_two_stage_exploratory.py --device cuda:1 --out <dir>
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.data.normalization import normalize_image
from satchange.execution import resolve_device
from satchange.semantic import roi_screening as rs
from satchange.semantic.scd_metrics import SemanticChangeMetrics
from satchange.semantic.system_eval import load_semantic_checkpoint

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/SECOND"
RUNS = ROOT / "artifacts/research2/v2/runs/pilot"
SCREENING = ROOT / "artifacts/research2/v2/screening"
NUM_CLASSES = 7
PAD = 32  # all three encoders need H, W divisible by 32 (reflect padding, removed afterwards)

# Pilot checkpoints (seed 42, lr 3e-4). "best": the budget with the highest validation SeK;
# "largest": the largest finished budget (closest to the common b* = 3.2e17, D-069).
CHECKPOINTS = {
    "best": {
        "siam_unet_resnet18": "siam_unet_resnet18/unet_4e16_lr3e-4",
        "scannet": "scannet/scannet_1.6e17_lr3e-4_rerun",
        "mamba_scd": "mamba_scd/mamba_8e16_lr3e-4",
    },
    "largest": {
        "siam_unet_resnet18": "siam_unet_resnet18/unet_8e16_lr3e-4",
        "scannet": "scannet/scannet_3.2e17_lr3e-4",
        "mamba_scd": "mamba_scd/mamba_1.6e17_lr3e-4",
    },
}
POLICIES = {
    "balanced": rs.ScreeningSettings(0.5, 16, 32, 64),
    "recall-oriented": rs.ScreeningSettings(0.35, 32, 64, 64),
    "one box": rs.ScreeningSettings(0.35, 16, None, 128),
}
DETECTOR_SEEDS = (42, 123, 777)
FRONTIER_BUDGETS = (0.3, 0.5, 0.7)
IMAGENET = "imagenet"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _change_probability(change: torch.Tensor) -> torch.Tensor:
    """Change probability [B,H,W] from a 1-channel (sigmoid) or 2-channel (softmax) head."""
    change = change.float()
    return change[:, 0].sigmoid() if change.shape[1] == 1 else change.softmax(dim=1)[:, 1]


def _decode(
    change_prob: torch.Tensor, log_prob: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maps [H,W] from change probability [H,W] and per-date log-probabilities [2,7,H,W].

    Changed = probability > 0.5; class = argmax over classes 1..6 (`land_cover`, D-049).
    """
    changed = change_prob > 0.5
    maps = log_prob[:, 1:].argmax(dim=1) + 1
    maps = torch.where(changed[None], maps, torch.zeros_like(maps))
    return maps[0], maps[1]


@torch.inference_mode()
def _forward(model: torch.nn.Module, before: torch.Tensor, after: torch.Tensor):
    """(change probability [H,W], per-date log-softmax [2,7,H,W]) for one [3,h,w] crop pair."""
    height, width = before.shape[-2:]
    pad_h, pad_w = (-height) % PAD, (-width) % PAD

    def prepare(image: torch.Tensor) -> torch.Tensor:
        batch = normalize_image(image, IMAGENET)[None]
        return F.pad(batch, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else batch

    change, semantic_1, semantic_2 = model(prepare(before), prepare(after))
    probability = _change_probability(change)[0, :height, :width]
    semantic = torch.stack([semantic_1.float(), semantic_2.float()], dim=1)[0]
    log_prob = F.log_softmax(semantic, dim=1)[:, :, :height, :width]
    if log_prob.shape[1] != NUM_CLASSES:
        raise ValueError("Expected 7 semantic channels.")
    return probability, log_prob


def _hist(maps: tuple[torch.Tensor, torch.Tensor], labels: tuple[torch.Tensor, ...]) -> np.ndarray:
    metrics = SemanticChangeMetrics(NUM_CLASSES)
    for prediction, label in zip(maps, labels, strict=True):
        metrics.update(prediction.cpu(), label)
    return metrics.hist.copy()


def _metrics(hist: np.ndarray) -> dict[str, float]:
    metrics = SemanticChangeMetrics(NUM_CLASSES)
    metrics.hist = hist.astype(np.int64)
    return metrics.compute()


def _boxes(probabilities: np.ndarray, settings: rs.ScreeningSettings) -> np.ndarray:
    detections = rs.detections_by_threshold(probabilities, (settings.threshold,))
    return rs.generate_boxes(detections, settings, *probabilities.shape)


def run_scene_single(model, sample, device):
    before, after = sample["before"].to(device), sample["after"].to(device)
    probability, log_prob = _forward(model, before, after)
    return _decode(probability, log_prob)


def run_scene_two_stage(model, sample, probabilities, settings, device):
    """Detector map (numpy [H,W]) -> boxes -> SCD per crop -> merged maps; returns also crop stats."""
    height, width = probabilities.shape
    boxes = _boxes(probabilities, settings)
    if len(boxes) == 0:
        empty = torch.zeros(height, width, dtype=torch.long, device=device)
        return (empty, empty.clone()), 0, 0
    before, after = sample["before"].to(device), sample["after"].to(device)
    change_sum = torch.zeros(height, width, device=device)
    log_sum = torch.zeros(2, NUM_CLASSES, height, width, device=device)
    covered = torch.zeros(height, width, device=device)
    for x0, y0, x1, y1 in boxes.tolist():
        probability, log_prob = _forward(
            model, before[:, y0:y1, x0:x1].contiguous(), after[:, y0:y1, x0:x1].contiguous()
        )
        change_sum[y0:y1, x0:x1] += probability
        log_sum[:, :, y0:y1, x0:x1] += log_prob
        covered[y0:y1, x0:x1] += 1
    seen = covered.clamp(min=1)
    first, second = _decode(change_sum / seen, log_sum / seen)
    inside = covered > 0
    first, second = first * inside, second * inside
    area = int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum())
    return (first, second), len(boxes), area


def _bootstrap_delta(
    hist_a: np.ndarray, hist_b: np.ndarray, resamples: int, seed: int = 0
) -> tuple[float, float]:
    """95 % tile-bootstrap CI of SeK(b) - SeK(a); both [N,7,7] pooled per tile."""
    rng = np.random.default_rng(seed)
    count = len(hist_a)
    deltas = np.empty(resamples)
    for index in range(resamples):
        pick = rng.integers(0, count, count)
        deltas[index] = (
            _metrics(hist_b[pick].sum(0))["sek"] - _metrics(hist_a[pick].sum(0))["sek"]
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-set", choices=sorted(CHECKPOINTS), default="best")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument(
        "--mode", choices=("fixed", "frontier"), default="fixed",
        help="fixed: three illustrative ROI policies + oracle; frontier: per-seed best-recall "
        "setting under each area budget (selected on the same validation tiles)",
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: first N tiles only")
    args = parser.parse_args()
    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)

    dataset, _ = create_dataset_dataloader(DATA, "val", None, 1, 0, semantic_labels=True)
    count = args.limit or len(dataset)
    samples = [dataset[i] for i in range(count)]  # raw [0,1] RGB, native 512x512, dense labels
    labels = [(s["before_label"], s["after_label"]) for s in samples]
    detector_maps: dict[int, np.ndarray] = {}
    for seed in DETECTOR_SEEDS:
        folder = SCREENING / f"medium_full_s{seed}"
        ids = np.load(folder / "table.npz")["ids"]
        if list(ids[:count]) != list(dataset.sample_ids[:count]):
            raise ValueError(f"Detector tile order differs from the dataset: seed {seed}")
        detector_maps[seed] = np.load(folder / "probabilities_fp16.npy", mmap_mode="r")[:count]
    detector_latency = json.load(open(SCREENING / "detector_latency_512.json"))["medium_full"]
    frontier_settings: dict[int, dict[float, rs.ScreeningSettings]] = {}
    if args.mode == "frontier":
        grid = rs.settings_grid()
        for seed in DETECTOR_SEEDS:
            table = np.load(SCREENING / f"medium_full_s{seed}" / "table.npz")["table"]
            pooled = rs.pooled_metrics(table.sum(axis=1), float(table.shape[1]), 512 * 512)
            frontier_settings[seed] = {}
            for budget in FRONTIER_BUDGETS:
                recall = np.where(
                    pooled["area_ratio"] <= budget, np.nan_to_num(pooled["pixel_recall"]), -1.0
                )
                frontier_settings[seed][budget] = grid[int(recall.argmax())]
    oracle_maps = np.stack([s["mask"][0].numpy() for s in samples]).astype(np.float32)

    results: dict[str, Any] = {
        "note": "exploratory, val only, pilot checkpoints, fixed ROI policies; see script docstring",
        "checkpoint_set": args.checkpoint_set,
        "tiles": count,
        "detector": "medium_full (SECOND fine-tuned, screening finals, B=3e16, lr 1e-3)",
        "detector_latency_ms_512": detector_latency,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "tests")),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "host_load_avg": list(__import__("os").getloadavg()),
        "models": {},
    }
    for name, relative in CHECKPOINTS[args.checkpoint_set].items():
        run_dir = RUNS / relative
        record = json.load(open(run_dir / "run_record.json"))
        model = load_semantic_checkpoint(run_dir / "best.pt", device)
        entry: dict[str, Any] = {
            "checkpoint": relative,
            "train_flops": record["training_flops"],
            "pilot_best_sek": record["best_sek"],
            "systems": {},
        }

        def evaluate(label: str, scene_fn) -> None:
            histograms, times, crops, areas = [], [], [], []
            for index in range(min(10, count)):  # warm-up, not recorded
                scene_fn(index)
            for index in range(count):
                _sync(device)
                started = time.perf_counter()
                maps, n_crops, area = scene_fn(index)
                maps = (maps[0].cpu(), maps[1].cpu())
                times.append((time.perf_counter() - started) * 1000.0)
                histograms.append(_hist(maps, labels[index]))
                crops.append(n_crops)
                areas.append(area / (512 * 512))
            stack = np.stack(histograms)
            entry["systems"][label] = {
                "metrics": _metrics(stack.sum(0)),
                "scd_pipeline_ms_median": statistics.median(times),
                "scd_pipeline_ms_mean": statistics.fmean(times),
                "crops_per_tile": statistics.fmean(crops),
                "area_ratio": statistics.fmean(areas),
                "_hist": stack,
            }
            print(name, label, entry["systems"][label]["metrics"]["sek"], flush=True)

        evaluate(
            "single", lambda i: (run_scene_single(model, samples[i], device), 1, 512 * 512)
        )
        if args.mode == "frontier":
            for budget in FRONTIER_BUDGETS:
                for seed in DETECTOR_SEEDS:
                    chosen = frontier_settings[seed][budget]
                    entry.setdefault("frontier_settings", {})[f"s{seed}|{budget}"] = (
                        chosen.__dict__
                    )
                    evaluate(
                        f"medium_s{seed}|frontier{budget}",
                        lambda i, s=chosen, sd=seed: run_scene_two_stage(
                            model, samples[i], np.asarray(detector_maps[sd][i], dtype=np.float32),
                            s, device,
                        ),
                    )
        else:
            for policy, settings in POLICIES.items():
                evaluate(
                    f"oracle|{policy}",
                    lambda i, s=settings: run_scene_two_stage(
                        model, samples[i], oracle_maps[i], s, device
                    ),
                )
                for seed in DETECTOR_SEEDS:
                    evaluate(
                        f"medium_s{seed}|{policy}",
                        lambda i, s=settings, sd=seed: run_scene_two_stage(
                            model, samples[i], np.asarray(detector_maps[sd][i], dtype=np.float32),
                            s, device,
                        ),
                    )
            # Self-check: one box covering the whole tile must reproduce the single-stage result.
            whole = rs.ScreeningSettings(0.5, 0, None, 512)
            evaluate(
                "selfcheck_full_tile",
                lambda i: run_scene_two_stage(model, samples[i], np.ones((512, 512), np.float32),
                                              whole, device),
            )
        single_hist = entry["systems"]["single"]["_hist"]
        if args.mode == "fixed":
            check = entry["systems"]["selfcheck_full_tile"]["_hist"]
            entry["selfcheck_full_tile_equals_single"] = bool(np.array_equal(single_hist, check))
        for label, system in entry["systems"].items():
            if label.startswith("medium_s"):
                low, high = _bootstrap_delta(single_hist, system["_hist"], args.resamples)
                system["delta_sek_vs_single_ci95"] = [low, high]
        for system in entry["systems"].values():
            del system["_hist"]
        results["models"][name] = entry
        del model
        torch.cuda.empty_cache()
    suffix = "" if args.mode == "fixed" else "_frontier"
    path = args.out / f"two_stage_vs_single{suffix}_{args.checkpoint_set}.json"
    path.write_text(json.dumps(results, indent=1))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
