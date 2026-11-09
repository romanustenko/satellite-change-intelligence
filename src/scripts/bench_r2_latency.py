"""Per-scene end-to-end latency of frozen systems A and B on one idle GPU (R2-P10-T04, D-026).

Scenes are drawn per density stratum (edges frozen in the manifest) with a fixed RNG from the
scene list/densities saved by ``eval_r2_frozen.py`` (``--results``). Each scene is timed with
``semantic.latency`` (CUDA-synchronized stage timers, warm-up, repetitions; median per scene).
One seed per system is timed (``--seed``): latency does not depend on the trained weights'
quality, and both systems use identical hardware, precision and TF32 state. Refuses a busy GPU
unless ``--allow-busy`` (then the background load is recorded). ``--split test`` needs the test
read already claimed by the evaluation pass (this script reads images only, never labels).

    python src/scripts/bench_r2_latency.py --manifest <m> --results <dir> --split val --gpu 1 \
        --out <json>
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

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.execution import resolve_device
from satchange.run import reconstruct_checkpoint
from satchange.semantic.analysis import assign_strata
from satchange.semantic.freeze import verify_freeze_manifest
from satchange.semantic.latency import benchmark_system_a, benchmark_system_b
from satchange.semantic.roi import ROISettings
from satchange.semantic.system_eval import load_semantic_checkpoint

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "SECOND"


def gpu_state(index: int) -> dict[str, Any]:
    """Utilization/memory of GPU ``index`` and the compute processes on it (nvidia-smi)."""
    def query(*arguments: str) -> str:
        return subprocess.run(["nvidia-smi", *arguments], check=True, capture_output=True,
                              text=True).stdout.strip()

    row = query(f"--id={index}", "--query-gpu=name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits").split(", ")
    processes = query(f"--id={index}", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader,nounits")
    return {"name": row[0], "utilization_percent": int(row[1]), "memory_used_mib": int(row[2]),
            "compute_processes": [p for p in processes.splitlines() if p]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--scenes-per-stratum", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = verify_freeze_manifest(args.manifest, root=ROOT)
    if args.split == "test" and not args.manifest.with_suffix(
        args.manifest.suffix + ".test_read.json"
    ).is_file():
        raise RuntimeError("Run eval_r2_frozen.py --split test first (it claims the test read).")
    before = gpu_state(args.gpu)
    if (before["utilization_percent"] > 10 or before["compute_processes"]) and not args.allow_busy:
        raise RuntimeError(f"GPU {args.gpu} is not idle: {before}")

    config_a = manifest["systems"]["A"]["config"]
    config_b = manifest["systems"]["B"]["config"]
    edges = config_b["analysis"]["density_edges"]
    saved = np.load(args.results / "scene_results.npz")
    ids, density = saved["sample_ids"].tolist(), saved["density"]
    strata = assign_strata(density, edges)
    rng = np.random.default_rng(args.rng_seed)
    chosen: list[int] = []
    for stratum in range(len(edges) + 1):
        members = np.flatnonzero(strata == stratum)
        take = min(args.scenes_per_stratum, members.size)
        chosen.extend(sorted(rng.choice(members, take, replace=False).tolist()))

    device = resolve_device(f"cuda:{args.gpu}")
    a_model = load_semantic_checkpoint(Path(config_a["checkpoint_by_seed"][args.seed]), device)
    detector, _ = reconstruct_checkpoint(Path(config_b["detector_by_seed"][args.seed]), device)
    what = load_semantic_checkpoint(Path(config_b["what_by_seed"][args.seed]), device)
    settings = ROISettings(**config_b["roi_settings"])
    side = int(config_b["detector_input_size"])
    dataset, _ = create_dataset_dataloader(DATA, args.split, None, 1, 0, semantic_labels=True)
    index_of = {s: i for i, s in enumerate(dataset.sample_ids)}

    rows = []
    for position in chosen:
        sample = dataset[index_of[ids[position]]]
        before_image = torch.as_tensor(sample["before"])
        after_image = torch.as_tensor(sample["after"])
        report_a = benchmark_system_a(a_model, before_image, after_image, device,
                                      warmup=args.warmup, repetitions=args.repetitions)
        report_b = benchmark_system_b(
            detector, what, before_image, after_image, device,
            detector_input_size=(side, side), threshold=settings.threshold, settings=settings,
            warmup=args.warmup, repetitions=args.repetitions,
        )
        rows.append({
            "sample_id": ids[position], "density": float(density[position]),
            "a_ms": report_a["total_ms"]["median"], "b_ms": report_b["total_ms"]["median"],
            "b_stages_ms": {k: v["median"] for k, v in report_b["stages_ms"].items()},
            "a_stages_ms": {k: v["median"] for k, v in report_a["stages_ms"].items()},
        })
    after = gpu_state(args.gpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "split": args.split, "seed": args.seed, "sample_ids": [r["sample_id"] for r in rows],
        "latency_a_ms": [r["a_ms"] for r in rows], "latency_b_ms": [r["b_ms"] for r in rows],
        "rows": rows, "median_a_ms": statistics.median(r["a_ms"] for r in rows),
        "median_b_ms": statistics.median(r["b_ms"] for r in rows),
        "gpu_before": before, "gpu_after": after, "warmup": args.warmup,
        "repetitions": args.repetitions, "precision": "fp32",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32, "torch": torch.__version__,
        "manifest": str(args.manifest), "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }, indent=2))
    print(f"saved {args.out}: {len(rows)} scenes", flush=True)


if __name__ == "__main__":
    main()
