"""Forward FLOPs of the SCD models (per context-window size) and the detectors (R2-V9, protocol
§5a).

Convention of ``semantic/compute.py``: matmul/conv/attention FLOPs from ``FlopCounterMode`` plus the
analytic ``selective_scan`` formula; elementwise ops and norms are not counted. Batch 1, eval mode,
architecture only (weights do not change FLOPs). Output ``v9/inference_flops.json``.

Usage: python src/scripts/measure_r2_v2_inference_flops.py --device cuda:0
"""

import argparse
import json

import r2_v2_systems_lib as lib
import torch
from torch.utils.flop_counter import FlopCounterMode

from satchange.execution import resolve_device
from satchange.flop_registry import recording_analytic_flops
from satchange.models.factory import BINARY_TASK, SEMANTIC_TASK, create_task_model

SCD_PARAMETERS = {"siam_unet_resnet18": {"head_layout": "multi_head"}, "scannet": {},
                  "mamba_scd": {}}


@torch.inference_mode()
def forward_flops(model: torch.nn.Module, size: int, device: torch.device) -> int:
    model.eval()
    before = torch.rand(1, 3, size, size, device=device)
    after = torch.rand(1, 3, size, size, device=device)
    with recording_analytic_flops() as analytic, FlopCounterMode(display=False) as counter:
        model(before, after)
    return int(sum(counter.get_flop_counts().get("Global", {}).values()) + analytic.total_forward)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result: dict = {"convention": "FlopCounterMode (matmul/conv/attn) + analytic selective_scan",
                    "scd": {}, "detectors": {}}
    for name, parameters in SCD_PARAMETERS.items():
        model = create_task_model(name, parameters, SEMANTIC_TASK).to(device)
        result["scd"][name] = {str(size): forward_flops(model, size, device)
                               for size in lib.BUCKETS}
        print(name, result["scd"][name], flush=True)
        del model
        torch.cuda.empty_cache()
    for variant, (_, detector, _, input_size) in lib.DETECTOR_RUNS.items():
        metadata = json.loads((lib.ROOT / f"artifacts/final_models/{detector}/metadata.json")
                              .read_text())
        model = create_task_model(metadata["model_name"], metadata["model_parameters"],
                                  BINARY_TASK).to(device)
        size = input_size or lib.TILE
        result["detectors"][variant] = {"input_size": size,
                                        "forward_flops": forward_flops(model, size, device)}
        print(variant, result["detectors"][variant], flush=True)
        del model
    path = lib.V9 / "inference_flops.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
