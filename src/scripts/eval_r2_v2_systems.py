"""Evaluate the 12 systems (case 1 single-stage + cases 2-4 two-stage) per SCD model and seed.

One run of ``evaluate_pair`` produces, for one SCD checkpoint (model, seed) and every system
built on
it, per-tile pooled 7x7 histograms (so scene bootstraps can recompute SeK/mIoU/Fscd), ROI statistics
and processed-area numbers: ``single`` plus ``<case>__<policy>`` for the three detector cases and
the
declared ROI policies, plus the diagnostic ``oracle__balanced`` (boxes from the ground-truth mask).
Two-stage systems of SCD seed s use detector seed s (D-075). Inference path = ``batch_ctx`` (D-074).

``--split val`` is free to run. ``--split test`` needs a verified freeze manifest, makes the single
test-read claim and fans out to one worker process per GPU (the marker proves the claim for the
workers); it never selects anything.

Usage:
  python src/scripts/eval_r2_v2_systems.py --split val --devices cuda:0 cuda:1 [--models ...]
  python src/scripts/eval_r2_v2_systems.py --split test --manifest <manifest.json> --devices ...
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import bench_r2_v2_two_stage_speedups as bench
import eval_r2_v2_two_stage_exploratory as base
import numpy as np
import r2_v2_systems_lib as lib
import torch

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.execution import resolve_device
from satchange.semantic.freeze import claim_test_read, sha256_file, verify_freeze_manifest
from satchange.semantic.system_eval import load_semantic_checkpoint

rs = lib.rs
FROZEN_CODE = (
    "src/satchange", "src/scripts/eval_r2_v2_systems.py", "src/scripts/r2_v2_systems_lib.py",
    "src/scripts/bench_r2_v2_two_stage_speedups.py",
    "src/scripts/eval_r2_v2_two_stage_exploratory.py",
    "src/scripts/eval_r2_v2_detector_screening.py", "configs/SCREENING_RULE.json",
)


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, cwd=lib.ROOT,
                          check=True).stdout.strip()


def union_area(boxes: np.ndarray) -> int:
    covered = np.zeros((lib.TILE, lib.TILE), dtype=bool)
    for x0, y0, x1, y1 in boxes.tolist():
        covered[y0:y1, x0:x1] = True
    return int(covered.sum())


def load_split(split: str):
    dataset, _ = create_dataset_dataloader(lib.DATA, split, None, 1, 0, semantic_labels=True)
    samples = [dataset[i] for i in range(len(dataset))]
    return dataset, samples


def ensure_maps(split: str, variant: str, seed: int, device: torch.device) -> None:
    """Detector maps of ``variant``/``seed`` on ``split`` (computed once, atomic write)."""
    path = lib.map_path(split, variant, seed)
    if path.exists():
        return
    probabilities, _, _ = lib.compute_detector_maps(variant, seed, split, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, probabilities)
    temporary.replace(path)  # a concurrent worker may have written the same bytes


def evaluate_pair(
    model_name: str, seed: int, split: str, device: torch.device, out_dir: Path, limit: int = 0
) -> Path:
    path = out_dir / f"{model_name}_s{seed}.npz"
    if path.exists():
        print("exists", path.name, flush=True)
        return path
    partner = lib.partner_name()
    needed = {"medium_ft", "medium_scratch", partner}
    for variant in sorted(needed):
        ensure_maps(split, variant, seed, device)
    dataset, samples = load_split(split)
    count = limit or len(samples)
    samples = samples[:count]
    labels = [(s["before_label"], s["after_label"]) for s in samples]
    ids = np.array(dataset.sample_ids[:count])
    for variant in sorted(needed):  # tile order of the maps must be the dataset order
        stored = np.load(lib.map_path(split, variant, seed), mmap_mode="r")
        if stored.shape[0] < count:
            raise ValueError(f"{variant}: only {stored.shape[0]} maps for {count} tiles")
    oracle = np.stack([s["mask"][0].numpy() for s in samples]).astype(np.float32)
    detector_maps = {case: lib.load_maps(split, variant, seed, partner)[:count]
                     for case, variant in lib.CASES.items()}
    policies = {case: lib.policies_for(case) for case in lib.CASES}
    model = load_semantic_checkpoint(lib.scd_run_dir(model_name, seed) / "best.pt", device)
    runner = bench.Runner(model, bf16=False)
    systems: dict[str, tuple] = {"single": None}
    for case in lib.CASES:
        for policy in policies[case]:
            systems[f"{case}__{policy}"] = (case, policy)
    systems["oracle__balanced"] = ("oracle", "balanced")
    store: dict[str, dict[str, list]] = {k: {"hist": [], "crops": [], "box_area": [],
                                              "window_area": [], "union_area": [], "buckets": []}
                                          for k in systems}
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        for key, spec in systems.items():
            if spec is None:
                probability, log_prob = bench.forward_batch(
                    runner, sample["before"].to(device)[None], sample["after"].to(device)[None])
                maps = base._decode(probability[0], log_prob[0])
                boxes = np.array([[0, 0, lib.TILE, lib.TILE]])
            else:
                case, policy = spec
                if case == "oracle":
                    probabilities = oracle[index]
                    settings = lib.FIXED_POLICY["balanced"]
                else:
                    probabilities = np.asarray(detector_maps[case][index], dtype=np.float32)
                    settings = policies[case][policy]
                boxes = base._boxes(probabilities, settings)
                maps, _ = bench.scene("batch_ctx", runner, sample, boxes, device)
            row = store[key]
            row["hist"].append(base._hist((maps[0].cpu(), maps[1].cpu()), labels[index]))
            row["crops"].append(len(boxes))
            row["box_area"].append(
                int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum()) if len(boxes)
                else 0)
            row["window_area"].append(lib.processed_window_area(boxes) if spec else lib.TILE ** 2)
            row["union_area"].append(union_area(boxes) if len(boxes) else 0)
            row["buckets"].append(
                [sum(1 for box in boxes.tolist() if spec and lib.window_bucket(box) == b)
                 for b in lib.BUCKETS] if spec else [0, 0, 0, 0]
            )
        if index % 50 == 0:
            print(model_name, seed, f"{index}/{count}", f"{time.perf_counter() - started:.0f}s",
                  flush=True)
    arrays = {"ids": ids, "density": np.array([float((s["mask"] > 0.5).float().mean())
                                               for s in samples])}
    for key, row in store.items():
        for field, values in row.items():
            arrays[f"{field}__{key}"] = (
                np.stack(values) if field == "hist" else np.array(values)
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    (out_dir / f"{model_name}_s{seed}.json").write_text(json.dumps({
        "model": model_name, "seed": seed, "split": split, "tiles": count,
        "checkpoint": str(lib.scd_run_dir(model_name, seed) / "best.pt"),
        "checkpoint_sha256": sha256_file(lib.scd_run_dir(model_name, seed) / "best.pt"),
        "partner": partner, "inference_path": "batch_ctx", "seconds": time.perf_counter() - started,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "tests")),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
    }, indent=2))
    del model
    torch.cuda.empty_cache()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=list(lib.SCD_MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(lib.SEEDS))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--worker", action="store_true", help="internal: claim already made")
    parser.add_argument("--pairs", nargs="+", help="internal: model:seed pairs of one worker")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N tiles (val only)")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out_dir = args.out or lib.V9 / "eval" / args.split
    pairs = (
        [(item.split(":")[0], int(item.split(":")[1])) for item in args.pairs]
        if args.pairs else [(m, s) for m in args.models for s in args.seeds]
    )
    if args.split == "test":
        if args.manifest is None or args.limit:
            raise SystemExit("--split test needs --manifest and no --limit")
        verify_freeze_manifest(args.manifest, root=lib.ROOT, code_paths=FROZEN_CODE)
        marker = args.manifest.with_suffix(args.manifest.suffix + ".test_read.json")
        if args.worker:
            claim = json.loads(marker.read_text())
            if claim["manifest_sha256"] != sha256_file(args.manifest):
                raise SystemExit("test-read claim does not belong to this manifest")
        else:
            claim_test_read(args.manifest, system_set="12 systems")
            print("test-read claim recorded:", marker, flush=True)
            procs = []
            for position, device in enumerate(args.devices):
                share = pairs[position::len(args.devices)]
                if share:
                    procs.append(subprocess.Popen([
                        sys.executable, __file__, "--split", "test", "--manifest",
                        str(args.manifest), "--worker", "--devices", device, "--pairs",
                        *[f"{m}:{s}" for m, s in share],
                    ]))
            raise SystemExit(max(p.wait() for p in procs))
    device = resolve_device(args.devices[0])
    for model_name, seed in pairs:
        evaluate_pair(model_name, seed, args.split, device, out_dir, args.limit)


if __name__ == "__main__":
    main()
