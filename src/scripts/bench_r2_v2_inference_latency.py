"""Forward-only inference latency of the three v2 SCD models and the six Research 1 detectors.

Protocol matches the Research 1 forward benchmark (H5): FP32, TF32 off, batch 1, eval mode,
``torch.inference_mode``, warm-up, CUDA-synchronised timing of the model forward only (no I/O,
resize, H2D, sigmoid, thresholding, ROI extraction or reconstruction). Weights are irrelevant to
latency, so every model is built from its architecture parameters (SCD: random init; detectors:
``metadata.json`` of ``artifacts/final_models/<variant>``) and no checkpoint is opened. Detectors
are measured at their Research 1 input size (sanity check against the historical numbers) and at
the SECOND tile size. Run only on an idle GPU; GPU load before and after is recorded.

Usage: python src/scripts/bench_r2_v2_inference_latency.py --device cuda:0 --physical-gpu 2 \
    --output <json>
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from satchange.execution import resolve_device, seed_everything
from satchange.models.factory import BINARY_TASK, SEMANTIC_TASK, create_task_model

ROOT = Path(__file__).resolve().parents[2]
SCD_MODELS = {"siam_unet_resnet18": {"head_layout": "multi_head"}, "scannet": {}, "mamba_scd": {}}
# variant -> input sizes (the first one is the Research 1 native size used for the sanity check).
DETECTORS = {
    "light_full": (1024, 512),
    "medium_full": (1024, 512),
    "heavy_full": (1024, 512),
    "light_resized": (256,),
    "medium_resized": (256,),
    "heavy_resized": (256,),
}


def _gpu_state(index: int) -> str:
    return subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _time_forward(
    model: nn.Module, size: int, device: torch.device, warmup: int, repeats: int
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)
    before = torch.rand(1, 3, size, size, generator=generator).to(device)
    after = torch.rand(1, 3, size, size, generator=generator).to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    times: list[float] = []
    with torch.inference_mode():
        for index in range(warmup + repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(before, after)
            torch.cuda.synchronize(device)
            if index >= warmup:
                times.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(times)
    return {
        "input_size": size,
        "median_ms": statistics.median(times),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--scene-size", type=int, default=512, help="SECOND tile size")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state_before = _gpu_state(args.physical_gpu)
    rows: list[dict[str, Any]] = []

    for name, parameters in SCD_MODELS.items():
        seed_everything(args.seed)
        model = create_task_model(name, parameters, SEMANTIC_TASK).to(device)
        row = {"role": "scd_whole_scene", "model": name, "parameters": _parameter_count(model)}
        row.update(_time_forward(model, args.scene_size, device, args.warmup, args.repeats))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()

    for variant, sizes in DETECTORS.items():
        metadata = json.loads(
            (ROOT / "artifacts" / "final_models" / variant / "metadata.json").read_text()
        )
        seed_everything(args.seed)
        model = create_task_model(
            metadata["model_name"], metadata["model_parameters"], BINARY_TASK
        ).to(device)
        for size in sizes:
            row = {
                "role": "detector",
                "model": variant,
                "architecture": metadata["model_name"],
                "parameters": _parameter_count(model),
            }
            row.update(_time_forward(model, size, device, args.warmup, args.repeats))
            rows.append(row)
            print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()

    record = {
        "task": "R2 inference latency (SCD vs detectors)",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "device": args.device,
        "physical_gpu": args.physical_gpu,
        "gpu_state_before": state_before,
        "gpu_state_after": _gpu_state(args.physical_gpu),
        "precision": "fp32 (TF32 off)",
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "model forward only, weights not loaded (random init / architecture only)",
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip(),
        "git_dirty_paths": subprocess.run(
            ["git", "status", "--short", "--", "src", "tests"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.splitlines(),
        "rows": rows,
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
