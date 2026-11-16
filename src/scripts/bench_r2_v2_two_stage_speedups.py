"""Latency speed-up probes for the exploratory two-stage system on SECOND validation (user request
2026-11-16).

Measurement only: nothing is trained or tuned, the test split is never read, ``src/satchange`` is
not
touched. Same pilot SCD checkpoints, same Medium (BIT) detector probability maps (seed 42) and the
same
ROI policies as ``eval_r2_v2_two_stage_exploratory.py``. Hypotheses tested (the cheap Resized
detector
is deliberately excluded, the detector is fixed to Medium, D-067):

* H1 batching crops: ``batch_exact`` (crops of identical size in one forward) and ``batch_ctx``
  (every
  crop replaced by a square window of a bucket size 128/256/384/512 cut from the tile around the
  crop, so
  that crops batch; only the original box is merged back, the extra pixels are real context).
* H3 launch-overhead removal: bf16 autocast. CUDA graphs were tried first and rejected: capture of
  these
  models fails in the default mode (torch 2.12.1+cu130) and the "relaxed" mode silently replays
  wrong
  outputs (max |graph - eager| = 0.23 on U-Net), so no graph numbers are reported.
* H4 fixed grid: 128x128 / 256x256 cells overlapping any ROI box, one batched forward, no context.
* H5 skip empty tiles: fraction of tiles without any box (no run needed).

Quality (SeK) is computed for every variant on the same tiles; timing = per tile, CUDA-synchronised,
includes CPU box generation, H2D, forward(s), merging and D2H of the two label maps (as in the
earlier
exploratory harness). Run on an idle GPU; load is recorded.

Usage: python src/scripts/bench_r2_v2_two_stage_speedups.py --device cuda:1 --tiles 200 --out <dir>
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_r2_v2_two_stage_exploratory as base  # noqa: E402

from satchange.data.levir_cd import create_dataset_dataloader  # noqa: E402
from satchange.data.normalization import normalize_image  # noqa: E402
from satchange.execution import resolve_device  # noqa: E402
from satchange.models.factory import BINARY_TASK, create_task_model  # noqa: E402
from satchange.semantic.system_eval import load_semantic_checkpoint  # noqa: E402

rs = base.rs
NUM_CLASSES = base.NUM_CLASSES
PAD = base.PAD
BUCKETS = (128, 256, 384, 512)
ROOT = base.ROOT


class Runner:
    """Model call with optional bf16 autocast (CUDA graphs rejected, see module docstring)."""

    def __init__(self, model: torch.nn.Module, *, bf16: bool) -> None:
        self.model, self.bf16 = model, bf16

    def __call__(self, before: torch.Tensor, after: torch.Tensor):
        if self.bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(before, after)
        return self.model(before, after)


def forward_batch(runner: Runner, before: torch.Tensor, after: torch.Tensor):
    """(change probability [B,h,w], per-date log-softmax [B,2,7,h,w]) for raw crops [B,3,h,w]."""
    height, width = before.shape[-2:]
    pad_h, pad_w = (-height) % PAD, (-width) % PAD

    def prepare(image: torch.Tensor) -> torch.Tensor:
        batch = normalize_image(image, base.IMAGENET)
        return F.pad(batch, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else batch

    change, semantic_1, semantic_2 = runner(prepare(before), prepare(after))
    probability = base._change_probability(change)[:, :height, :width]
    semantic = torch.stack([semantic_1.float(), semantic_2.float()], dim=1)
    log_prob = F.log_softmax(semantic, dim=2)[..., :height, :width]
    return probability, log_prob


class Merger:
    """Accumulates per-box outputs into tile-level maps (mean over overlaps, 0 outside)."""

    def __init__(self, device: torch.device) -> None:
        self.change = torch.zeros(512, 512, device=device)
        self.log = torch.zeros(2, NUM_CLASSES, 512, 512, device=device)
        self.covered = torch.zeros(512, 512, device=device)

    def add(self, x0: int, y0: int, x1: int, y1: int, probability, log_prob) -> None:
        self.change[y0:y1, x0:x1] += probability
        self.log[:, :, y0:y1, x0:x1] += log_prob
        self.covered[y0:y1, x0:x1] += 1

    def maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        seen = self.covered.clamp(min=1)
        first, second = base._decode(self.change / seen, self.log / seen)
        inside = self.covered > 0
        return first * inside, second * inside


@torch.inference_mode()
def scene(kind: str, runner: Runner, sample: dict[str, Any], boxes: np.ndarray, device):
    """Run one tile; returns ((T1, T2) maps, forward calls)."""
    merger = Merger(device)
    if len(boxes) == 0:
        return merger.maps(), 0
    before, after = sample["before"].to(device), sample["after"].to(device)
    calls = 0
    if kind == "single":
        probability, log_prob = forward_batch(runner, before[None], after[None])
        return base._decode(probability[0], log_prob[0]), 1
    if kind == "seq":
        for x0, y0, x1, y1 in boxes.tolist():
            probability, log_prob = forward_batch(
                runner, before[None, :, y0:y1, x0:x1], after[None, :, y0:y1, x0:x1]
            )
            merger.add(x0, y0, x1, y1, probability[0], log_prob[0])
            calls += 1
    elif kind == "batch_exact":
        groups: dict[tuple[int, int], list[list[int]]] = {}
        for box in boxes.tolist():
            groups.setdefault((box[3] - box[1], box[2] - box[0]), []).append(box)
        for members in groups.values():
            stack_b = torch.stack([before[:, y0:y1, x0:x1] for x0, y0, x1, y1 in members])
            stack_a = torch.stack([after[:, y0:y1, x0:x1] for x0, y0, x1, y1 in members])
            probability, log_prob = forward_batch(runner, stack_b, stack_a)
            for index, (x0, y0, x1, y1) in enumerate(members):
                merger.add(x0, y0, x1, y1, probability[index], log_prob[index])
            calls += 1
    elif kind == "batch_ctx":
        groups_ctx: dict[int, list[tuple[list[int], int, int]]] = {}
        for box in boxes.tolist():
            x0, y0, x1, y1 = box
            size = max(x1 - x0, y1 - y0)
            bucket = next(b for b in BUCKETS if b >= size)
            wx = min(max((x0 + x1) // 2 - bucket // 2, 0), 512 - bucket)
            wy = min(max((y0 + y1) // 2 - bucket // 2, 0), 512 - bucket)
            groups_ctx.setdefault(bucket, []).append((box, wx, wy))
        for bucket, members in groups_ctx.items():
            window = [(slice(wy, wy + bucket), slice(wx, wx + bucket)) for _, wx, wy in members]
            stack_b = torch.stack([before[:, rows, cols] for rows, cols in window])
            stack_a = torch.stack([after[:, rows, cols] for rows, cols in window])
            probability, log_prob = forward_batch(runner, stack_b, stack_a)
            for index, ((x0, y0, x1, y1), wx, wy) in enumerate(members):
                sl_y, sl_x = slice(y0 - wy, y1 - wy), slice(x0 - wx, x1 - wx)
                merger.add(x0, y0, x1, y1, probability[index][sl_y, sl_x],
                           log_prob[index][:, :, sl_y, sl_x])
            calls += 1
    elif kind.startswith("grid"):
        cell = int(kind[4:])
        cells = [
            (gx, gy)
            for gy in range(0, 512, cell)
            for gx in range(0, 512, cell)
            if bool(((boxes[:, 0] < gx + cell) & (boxes[:, 2] > gx)
                     & (boxes[:, 1] < gy + cell) & (boxes[:, 3] > gy)).any())
        ]
        stack_b = torch.stack([before[:, gy:gy + cell, gx:gx + cell] for gx, gy in cells])
        stack_a = torch.stack([after[:, gy:gy + cell, gx:gx + cell] for gx, gy in cells])
        probability, log_prob = forward_batch(runner, stack_b, stack_a)
        for index, (gx, gy) in enumerate(cells):
            merger.add(gx, gy, gx + cell, gy + cell, probability[index], log_prob[index])
        calls = 1
    else:
        raise ValueError(kind)
    return merger.maps(), calls


def detector_latency(device: torch.device, repeats: int = 200) -> dict[str, Any]:
    """Medium (BIT) forward at 512^2, batch 1: eager / bf16 / CUDA graph (weights irrelevant)."""
    metadata = json.loads((ROOT / "artifacts/final_models/medium_full/metadata.json").read_text())
    model = create_task_model(metadata["model_name"], metadata["model_parameters"], BINARY_TASK)
    model = model.to(device).eval()
    generator = torch.Generator().manual_seed(0)
    before = torch.rand(1, 3, 512, 512, generator=generator).to(device)
    after = torch.rand(1, 3, 512, 512, generator=generator).to(device)
    result: dict[str, Any] = {}
    for label, bf16 in (("fp32", False), ("bf16", True)):
        try:
            runner = Runner(model, bf16=bf16)
            times = []
            with torch.inference_mode():
                for index in range(20 + repeats):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    runner(before, after)
                    torch.cuda.synchronize(device)
                    if index >= 20:
                        times.append((time.perf_counter() - started) * 1000.0)
            result[label] = statistics.median(times)
        except Exception as error:  # noqa: BLE001 - recorded, not hidden
            result[label] = f"failed: {type(error).__name__}: {str(error)[:200]}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tiles", type=int, default=200)
    args = parser.parse_args()
    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)

    dataset, _ = create_dataset_dataloader(base.DATA, "val", None, 1, 0, semantic_labels=True)
    count = min(args.tiles, len(dataset))
    samples = [dataset[i] for i in range(count)]
    labels = [(s["before_label"], s["after_label"]) for s in samples]
    folder = base.SCREENING / "medium_full_s42"
    if list(np.load(folder / "table.npz")["ids"][:count]) != list(dataset.sample_ids[:count]):
        raise ValueError("Detector tile order differs from the dataset")
    detector_map = np.load(folder / "probabilities_fp16.npy", mmap_mode="r")[:count]
    maps = [np.asarray(detector_map[i], dtype=np.float32) for i in range(count)]

    grid = rs.settings_grid()
    table = np.load(folder / "table.npz")["table"]
    pooled = rs.pooled_metrics(table.sum(axis=1), float(table.shape[1]), 512 * 512)
    recall = np.where(pooled["area_ratio"] <= 0.5, np.nan_to_num(pooled["pixel_recall"]), -1.0)
    policies = {"balanced": base.POLICIES["balanced"], "frontier0.5": grid[int(recall.argmax())]}
    boxes_by_policy = {
        name: [base._boxes(maps[i], settings) for i in range(count)]
        for name, settings in policies.items()
    }
    skip = {
        name: float(np.mean([len(b) == 0 for b in boxes]))
        for name, boxes in boxes_by_policy.items()
    }

    results: dict[str, Any] = {
        "note": "speed-up probes, val only, first N tiles, Medium seed 42; see script docstring",
        "tiles": count,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "host_load_avg_start": list(__import__("os").getloadavg()),
        "git_revision": base._git("rev-parse", "HEAD"),
        "policies": {k: v.__dict__ for k, v in policies.items()},
        "empty_tile_fraction_H5": skip,
        "mean_boxes": {k: float(np.mean([len(b) for b in v])) for k, v in boxes_by_policy.items()},
        "detector_ms_512": detector_latency(device),
        "models": {},
    }
    print("detector", results["detector_ms_512"], flush=True)

    def evaluate(model, kind, mode, policy):
        runner = Runner(model, bf16=mode == "bf16")
        boxes_all = boxes_by_policy[policy] if policy else [np.zeros((1, 4), np.int64)] * count
        full = np.array([[0, 0, 512, 512]], dtype=np.int64)

        def run(index: int):
            boxes = full if kind == "single" else base._boxes(maps[index], policies[policy])
            return scene(kind, runner, samples[index], boxes, device)

        for index in range(min(10, count)):
            run(index)
        times, hists, calls = [], [], []
        for index in range(count):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            (first, second), n_calls = run(index)
            first, second = first.cpu(), second.cpu()
            times.append((time.perf_counter() - started) * 1000.0)
            hists.append(base._hist((first, second), labels[index]))
            calls.append(n_calls)
        del boxes_all
        return {
            "metrics": base._metrics(np.stack(hists).sum(0)),
            "scd_stage_ms_median": statistics.median(times),
            "scd_stage_ms_p95": float(np.percentile(times, 95)),
            "forward_calls_per_tile": statistics.fmean(calls),
            "_hist": np.stack(hists),
        }

    for name, relative in base.CHECKPOINTS["best"].items():
        model = load_semantic_checkpoint(base.RUNS / relative / "best.pt", device)
        entry: dict[str, Any] = {"checkpoint": relative, "variants": {}}
        plan = [("single", "fp32", None), ("single", "bf16", None)]
        for policy in policies:
            plan += [("seq", "fp32", policy), ("seq", "bf16", policy),
                     ("batch_exact", "fp32", policy), ("batch_ctx", "fp32", policy),
                     ("batch_ctx", "bf16", policy), ("grid128", "fp32", policy),
                     ("grid128", "bf16", policy), ("grid256", "fp32", policy)]
        for kind, mode, policy in plan:
            label = f"{kind}|{mode}|{policy or '-'}"
            try:
                entry["variants"][label] = evaluate(model, kind, mode, policy)
                row = entry["variants"][label]
                print(name, label, round(row["metrics"]["sek"], 4),
                      round(row["scd_stage_ms_median"], 1), flush=True)
            except Exception as error:  # noqa: BLE001 - recorded, not hidden
                entry["variants"][label] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
                print(name, label, "FAILED", entry["variants"][label]["error"], flush=True)
            torch.cuda.empty_cache()
        for row in entry["variants"].values():
            row.pop("_hist", None)
        results["models"][name] = entry
        del model
        torch.cuda.empty_cache()
    results["host_load_avg_end"] = list(__import__("os").getloadavg())
    path = args.out / "two_stage_speedups.json"
    path.write_text(json.dumps(results, indent=1))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
