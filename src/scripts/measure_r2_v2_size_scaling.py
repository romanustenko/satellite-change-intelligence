"""Forward FLOPs and batch-1 latency of the detector and the SCD models versus input side length.

Exploratory check (user request 2026-11-22): does the two-stage/single-stage cost ratio depend on
the image size? Random weights (FLOPs and latency do not depend on them), fp32, no TF32, batch 1.

Usage: python src/scripts/measure_r2_v2_size_scaling.py cuda:N <out.json>
"""

import json
import statistics
import sys
from pathlib import Path

import r2_v2_systems_lib as lib
import torch
from measure_r2_v2_inference_flops import SCD_PARAMETERS, forward_flops

from satchange.models.factory import BINARY_TASK, SEMANTIC_TASK, create_task_model

SIZES = (512, 800, 1024, 2048)
device = torch.device(sys.argv[1])
out = Path(sys.argv[2])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.inference_mode()
def latency_ms(model: torch.nn.Module, size: int, runs: int = 30) -> float:
    before = torch.rand(1, 3, size, size, device=device)
    after = torch.rand(1, 3, size, size, device=device)
    for _ in range(5):
        model(before, after)
    times = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model(before, after)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


meta = json.loads((lib.ROOT / "artifacts/final_models/medium_full/metadata.json").read_text())
models = {"detector_bit": (meta["model_name"], meta["model_parameters"], BINARY_TASK)}
models |= {name: (name, p, SEMANTIC_TASK) for name, p in SCD_PARAMETERS.items()}
result = {}
for key, (name, parameters, task) in models.items():
    model = create_task_model(name, parameters, task).to(device).eval()
    result[key] = {}
    for size in SIZES:
        try:
            row = {"gflops": forward_flops(model, size, device) / 1e9,
                   "latency_ms": latency_ms(model, size)}
        except Exception as error:  # report, do not hide
            row = {"error": f"{type(error).__name__}: {str(error)[:200]}"}
        result[key][str(size)] = row
        print(key, size, row, flush=True)
        torch.cuda.empty_cache()
    del model
out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(device), "result": result}, indent=2))
