"""R2-V3-T03: reference training throughput of the three SCD models on an idle A100.

Synthetic resident batch (512x512, batch 8), the model's own loss, AdamW, fp32 with TF32 off:
median seconds per optimizer step -> achieved training FLOP/s and reference-A100 hours per FLOP.
SCanNet's pseudo-label teacher pass (4-flip TTA) is timed and FLOP-counted separately. Run only
on an idle GPU; GPU load before/after is recorded. No dataset is read.

Usage: python src/scripts/benchmark_r2_v2_throughput.py --device cuda:0 --physical-gpu 1 \
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
from torch.utils.flop_counter import FlopCounterMode

from satchange.execution import resolve_device, seed_everything
from satchange.models.factory import SEMANTIC_TASK, create_task_model
from satchange.semantic.compute import count_training_step_flops
from satchange.semantic.scd_losses import SCanNetPseudoLabeler, create_scd_loss

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"siam_unet_resnet18": {"head_layout": "multi_head"}, "scannet": {}, "mamba_scd": {}}


def _gpu_state(index: int) -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _batch(batch_size: int, size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    changed = torch.rand(batch_size, size, size, generator=generator) < 0.3
    labels = [
        torch.randint(1, 7, (batch_size, size, size), generator=generator) * changed
        for _ in range(2)
    ]
    images = [torch.randn(batch_size, 3, size, size, generator=generator) for _ in range(2)]
    return tuple(t.to(device) for t in (*images, *labels))


def _median_seconds(function: Any, device: torch.device, warmup: int, steps: int) -> list[float]:
    times = []
    for index in range(warmup + steps):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        if index >= warmup:
            times.append(time.perf_counter() - started)
    return times


def _benchmark(name: str, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    model = create_task_model(name, MODELS[name], SEMANTIC_TASK).to(device)
    loss_function = create_scd_loss(name).to(device)
    before, after, label_1, label_2 = _batch(args.batch_size, args.size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss_function(model(before, after), label_1, label_2).backward()
        optimizer.step()

    flops = count_training_step_flops(
        model, [before, after], lambda out: loss_function(out, label_1, label_2)
    )
    times = _median_seconds(step, device, args.warmup, args.steps)
    median = statistics.median(times)
    record: dict[str, Any] = {
        "model": name,
        "step_flops": flops.total,
        "median_step_seconds": median,
        "min_step_seconds": min(times),
        "max_step_seconds": max(times),
        "achieved_tflops": flops.total / median / 1e12,
        "reference_a100_hours_per_1e15_flops": 1e15 / (flops.total / median) / 3600,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    if name == "scannet":
        labeler = SCanNetPseudoLabeler(start_step=1, tta=True)
        labeler.refresh_teacher(model)
        label_0 = torch.zeros_like(label_1)

        def teacher() -> None:
            labeler(before, after, label_0)

        teacher_times = _median_seconds(teacher, device, args.warmup, args.steps)
        with FlopCounterMode(display=False) as counter:
            teacher()
        record["teacher_pass"] = {
            "median_seconds": statistics.median(teacher_times),
            "flops": counter.get_total_flops(),
            "tta": True,
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="torch device, e.g. cuda:0")
    parser.add_argument(
        "--physical-gpu",
        type=int,
        required=True,
        help="physical nvidia-smi index of that device (differs under CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = args.physical_gpu
    before_state = _gpu_state(index)
    results = []
    for name in MODELS:
        results.append(_benchmark(name, device, args))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    record = {
        "task": "R2-V3-T03",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "device": args.device,
        "physical_gpu": index,
        "gpu_state_before": before_state,
        "gpu_state_after": _gpu_state(index),
        "batch_size": args.batch_size,
        "input_size": args.size,
        "precision": "fp32 (TF32 off)",
        "warmup": args.warmup,
        "steps": args.steps,
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip(),
        "models": results,
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
