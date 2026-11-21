"""End-to-end latency per tile of the 12 systems and their ROI-policy variants (protocol §5b).

Measured on SECOND *validation* tiles (the test split is never read here), batch 1, fp32, TF32 off,
``torch.inference_mode``, one process on one GPU that should be idle (the background load of the
card before and after the run is recorded; a busy card is refused unless ``--allow-busy``).
One tile = host tensors -> device -> [detector forward(s) -> OR (case 4) -> probability map to
host -> ROI boxes] -> SCD forward(s) on the boxes (``batch_ctx``) -> merge -> both class maps back
to the host; the clock stops after a device synchronisation. Case 1 skips the bracketed part.
The tile is moved to the device a second time inside the shared ``scene`` helper (~6 MB, well under
1 ms; it is charged to every two-stage system alike).

Systems per SCD model (seed 42 weights, detector seed 42): ``single`` and ``<case>__<policy>`` for
case2/3/4 and the declared ROI policies. Systems are interleaved per tile so that slow drifts of the
card hit all of them equally; the first ``--warmup`` tiles are run through every system and
discarded (kernel selection, caches). Output: ``v9/latency_val.json`` (per model and system: median,
p95, mean ms, detector-stage median, ratio to ``single`` as median of per-tile ratios with a paired
tile bootstrap CI) and ``v9/latency_val_<model>.npz`` with the per-tile times.

Usage: python src/scripts/latency_r2_v2_systems.py --device cuda:0 [--models ...] [--tiles 444]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import bench_r2_v2_two_stage_speedups as bench
import eval_r2_v2_two_stage_exploratory as base
import numpy as np
import r2_v2_systems_lib as lib
import torch
from torch.nn import functional as F

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.execution import resolve_device
from satchange.semantic.system_eval import load_semantic_checkpoint

scr = lib.scr
SEED = 42
BOOT_SEED = 20261121
BUSY_UTIL_PERCENT = 10
BUSY_MEMORY_MB = 2000


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, cwd=lib.ROOT,
                          check=True).stdout.strip()


def gpu_background(device: torch.device) -> dict[str, float]:
    """Utilisation / memory of the card right now (nvidia-smi), to document the load."""
    index = device.index if device.index is not None else 0
    out = subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return {"utilization_percent": float(out[0]), "memory_used_mb": float(out[1])}


def load_detector(variant: str, device: torch.device):
    group, detector, template, input_size = lib.DETECTOR_RUNS[variant]
    model, _ = scr._load_detector(group, detector, SEED, template.format(seed=SEED))
    return model.to(device).eval(), input_size


@torch.inference_mode()
def detector_map(detector, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Probability map [1,512,512] of one detector (Resized regime: bilinear in and out)."""
    model, input_size = detector
    if input_size is not None:
        before, after = (
            F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
            for x in (before, after)
        )
    probability = model(before, after).float().sigmoid()
    if input_size is not None:
        probability = F.interpolate(probability, size=(lib.TILE, lib.TILE), mode="bilinear",
                                    align_corners=False).clamp(0, 1)
    return probability[0]


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


@torch.inference_mode()
def run_tile(spec, sample, runner, detectors, policies, partner, device):
    """One tile through one system; returns (total ms, detector-stage ms, number of boxes)."""
    _sync(device)
    started = time.perf_counter()
    if spec is None:  # case 1: whole tile
        before, after = sample["before"].to(device), sample["after"].to(device)
        probability, log_prob = bench.forward_batch(runner, before[None], after[None])
        maps = base._decode(probability[0], log_prob[0])
        maps = (maps[0].cpu(), maps[1].cpu())  # D2H is part of the measured path
        _sync(device)
        return (time.perf_counter() - started) * 1000.0, 0.0, 1
    case, policy = spec
    before, after = sample["before"].to(device)[None], sample["after"].to(device)[None]
    if case == "case4":
        fused = torch.maximum(detector_map(detectors["medium_ft"], before, after),
                              detector_map(detectors[partner], before, after))
        probabilities = fused[0].cpu().numpy()
    else:
        variant = lib.CASES[case]
        probabilities = detector_map(detectors[variant], before, after)[0].cpu().numpy()
    boxes = base._boxes(probabilities, policies[case][policy])
    detector_ms = (time.perf_counter() - started) * 1000.0
    maps, _ = bench.scene("batch_ctx", runner, sample, boxes, device)
    maps = (maps[0].cpu(), maps[1].cpu())
    _sync(device)
    return (time.perf_counter() - started) * 1000.0, detector_ms, len(boxes)


def ratio_interval(times: np.ndarray, reference: np.ndarray, rng: np.random.Generator) -> dict:
    """Median of per-tile ratios with a paired tile bootstrap 95 % CI (protocol §9)."""
    ratios = times / reference
    draws = rng.integers(0, ratios.size, (2000, ratios.size))
    medians = np.median(ratios[draws], axis=1)
    return {"median": float(np.median(ratios)),
            "ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--models", nargs="+", default=list(lib.SCD_MODELS))
    parser.add_argument("--tiles", type=int, default=0, help="0 = all validation tiles")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.device)
    before_load = gpu_background(device)
    if not args.allow_busy and (before_load["utilization_percent"] > BUSY_UTIL_PERCENT
                                or before_load["memory_used_mb"] > BUSY_MEMORY_MB):
        raise SystemExit(f"GPU is not idle ({before_load}); latency needs an idle card")
    out_dir = args.out or lib.V9
    partner = lib.partner_name()
    dataset, _ = create_dataset_dataloader(lib.DATA, "val", None, 1, 0, semantic_labels=True)
    count = args.tiles or len(dataset)
    samples = [dataset[i] for i in range(count)]
    detectors = {name: load_detector(name, device) for name in ("medium_ft", "medium_scratch",
                                                                  partner)}
    policies = {case: lib.policies_for(case) for case in lib.CASES}
    systems: dict[str, tuple | None] = {"single": None}
    for case in lib.CASES:
        for policy in policies[case]:
            systems[f"{case}__{policy}"] = (case, policy)
    rng = np.random.default_rng(BOOT_SEED)
    summary_path = out_dir / "latency_val.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    meta = {
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "background_before": before_load, "tiles": count, "warmup_tiles": args.warmup,
        "seed": SEED, "partner": partner, "precision": "fp32 (TF32 off), batch 1",
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "tests")),
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    for model_name in args.models:
        checkpoint = lib.scd_run_dir(model_name, SEED) / "best.pt"
        runner = bench.Runner(load_semantic_checkpoint(checkpoint, device), bf16=False)
        total = {key: [] for key in systems}
        stage = {key: [] for key in systems}
        boxes_count = {key: [] for key in systems}
        for index, sample in enumerate(samples):
            for key, spec in systems.items():
                ms, detector_ms, boxes = run_tile(spec, sample, runner, detectors, policies,
                                                  partner, device)
                if index >= args.warmup:
                    total[key].append(ms)
                    stage[key].append(detector_ms)
                    boxes_count[key].append(boxes)
            if index % 50 == 0:
                print(model_name, f"{index}/{count}", flush=True)
        arrays = {f"ms__{k}": np.array(v) for k, v in total.items()}
        arrays.update({f"detector_ms__{k}": np.array(v) for k, v in stage.items()})
        arrays.update({f"boxes__{k}": np.array(v) for k, v in boxes_count.items()})
        np.savez_compressed(out_dir / f"latency_val_{model_name}.npz", **arrays)
        reference = arrays["ms__single"]
        summary[model_name] = {
            key: {
                "median_ms": float(np.median(arrays[f"ms__{key}"])),
                "p95_ms": float(np.percentile(arrays[f"ms__{key}"], 95)),
                "mean_ms": float(arrays[f"ms__{key}"].mean()),
                "detector_stage_median_ms": float(np.median(arrays[f"detector_ms__{key}"])),
                "ratio_vs_single": ratio_interval(arrays[f"ms__{key}"], reference, rng),
            }
            for key in systems
        }
        del runner
        torch.cuda.empty_cache()
    meta["background_after"] = gpu_background(device)
    meta["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    summary["_meta"] = meta
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print("wrote", summary_path, flush=True)


if __name__ == "__main__":
    main()
