"""End-to-end latency harness for systems A and B (R2-P10-T04, D-026).

One scene (an already-loaded native RGB pair on the CPU) goes through every stage that a
deployment would run; each stage is bracketed by a device synchronization so asynchronous CUDA
work is attributed to the right stage. Stage names are fixed, so A and B tables are comparable:

* system B: ``h2d_resize`` (detector input), ``detector`` (forward + sigmoid + alignment),
  ``roi_extraction`` (threshold, connected components, crops), ``what_forward`` (all crops),
  ``reconstruction`` (merge into native class maps), ``d2h`` (maps to host).
* system A: ``h2d``, ``forward`` (one dense pass), ``argmax_d2h``.

Model loading and file I/O are excluded (as in ``final_benchmark._hardware_benchmark``).
"""

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from satchange.execution import precision_context
from satchange.semantic.roi import (
    DEFAULT_ROI_SETTINGS,
    ROISettings,
    align_whole_image_probabilities,
    extract_rois,
)
from satchange.semantic.scd_training import predict_maps
from satchange.semantic.what_stage import predict_roi_logits, reconstruct_scene

STAGES_B = ("h2d_resize", "detector", "roi_extraction", "what_forward", "reconstruction", "d2h")
STAGES_A = ("h2d", "forward", "argmax_d2h")


class StageTimer:
    """Accumulate per-stage wall-clock milliseconds for one pass; syncs CUDA around each stage."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.milliseconds: dict[str, float] = {}

    def run(self, name: str, function: Callable[[], Any]) -> Any:
        if name in self.milliseconds:
            raise ValueError(f"Stage timed twice in one pass: {name}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = function()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.milliseconds[name] = (time.perf_counter() - started) * 1000.0
        return result


def run_passes(
    one_pass: Callable[[StageTimer], Any],
    device: torch.device,
    stages: Sequence[str],
    *,
    warmup: int = 5,
    repetitions: int = 30,
) -> dict[str, Any]:
    """Time ``one_pass`` and return per-stage and total statistics (ms)."""
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be >= 0 and repetitions >= 1.")
    for _ in range(warmup):
        one_pass(StageTimer(device))
    totals: list[float] = []
    per_stage: dict[str, list[float]] = {name: [] for name in stages}
    for _ in range(repetitions):
        timer = StageTimer(device)
        one_pass(timer)
        if set(timer.milliseconds) != set(stages):
            raise RuntimeError(f"Stages {sorted(timer.milliseconds)} differ from {list(stages)}.")
        totals.append(sum(timer.milliseconds.values()))
        for name in stages:
            per_stage[name].append(timer.milliseconds[name])
    return {
        "total_ms": {"mean": statistics.mean(totals), "median": statistics.median(totals),
                     "samples": totals},
        "stages_ms": {n: {"mean": statistics.mean(v), "median": statistics.median(v)}
                      for n, v in per_stage.items()},
        "device": str(device), "warmup": warmup, "repetitions": repetitions,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


@torch.inference_mode()
def benchmark_system_a(
    model: nn.Module, before: torch.Tensor, after: torch.Tensor, device: torch.device,
    *, precision: str = "fp32", warmup: int = 5, repetitions: int = 30,
) -> dict[str, Any]:
    """Full-scene dense semantic pass (system A)."""
    model.eval().to(device)

    def one_pass(timer: StageTimer) -> Any:
        inputs = timer.run("h2d", lambda: (before.to(device)[None], after.to(device)[None]))

        def forward() -> torch.Tensor:
            with precision_context(device, precision):
                logits: torch.Tensor = model(*inputs).float()
                return logits

        logits = timer.run("forward", forward)
        return timer.run("argmax_d2h", lambda: [m.cpu() for m in predict_maps(logits)])

    return run_passes(one_pass, device, STAGES_A, warmup=warmup, repetitions=repetitions)


@torch.inference_mode()
def benchmark_system_b(
    detector: nn.Module, what_model: nn.Module, before: torch.Tensor, after: torch.Tensor,
    device: torch.device, *, detector_input_size: tuple[int, int], threshold: float = 0.5,
    settings: ROISettings = DEFAULT_ROI_SETTINGS, precision: str = "fp32",
    warmup: int = 5, repetitions: int = 30,
) -> dict[str, Any]:
    """WHERE detector -> ROI crops -> WHAT model -> reconstruction (system B)."""
    detector.eval().to(device)
    what_model.eval().to(device)
    height, width = before.shape[-2:]
    settings = ROISettings(**{**settings.__dict__, "threshold": threshold})

    def one_pass(timer: StageTimer) -> Any:
        def prepare() -> tuple[torch.Tensor, torch.Tensor]:
            return tuple(  # type: ignore[return-value]
                F.interpolate(v.to(device)[None], size=detector_input_size, mode="bilinear",
                              align_corners=False)
                for v in (before, after)
            )

        inputs = timer.run("h2d_resize", prepare)

        def detect() -> torch.Tensor:
            with precision_context(device, precision):
                logits = detector(*inputs)
            probabilities = logits.float().sigmoid()[0]
            return align_whole_image_probabilities(
                probabilities, (height, width), detector_input_size
            )

        probabilities = timer.run("detector", detect)
        rois = timer.run(
            "roi_extraction",
            lambda: extract_rois(before.to(device), after.to(device), probabilities, settings),
        )

        def classify() -> list[torch.Tensor]:
            with precision_context(device, precision):
                return [predict_roi_logits(what_model, roi, device) for roi in rois]

        logits = timer.run("what_forward", classify)

        def merge() -> tuple[torch.Tensor, torch.Tensor]:
            if not logits:
                empty = torch.zeros(height, width, dtype=torch.long, device=device)
                return empty, empty.clone()
            return reconstruct_scene(rois, logits, (height, width))

        maps = timer.run("reconstruction", merge)
        return timer.run("d2h", lambda: [m.cpu() for m in maps])

    return run_passes(one_pass, device, STAGES_B, warmup=warmup, repetitions=repetitions)
