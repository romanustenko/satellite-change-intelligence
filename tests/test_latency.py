"""End-to-end latency harness (CPU): stage bookkeeping, not timing values."""

import pytest
import torch

from satchange.models import FCSiamDiff
from satchange.semantic.latency import (
    STAGES_A,
    STAGES_B,
    StageTimer,
    benchmark_system_a,
    benchmark_system_b,
)

CPU = torch.device("cpu")


def _pair(size: int = 160) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    first = torch.rand(3, size, size, generator=generator)
    return first, torch.rand(3, size, size, generator=generator)


def test_stage_timer_rejects_duplicate_stage() -> None:
    timer = StageTimer(CPU)
    timer.run("x", lambda: 1)
    with pytest.raises(ValueError):
        timer.run("x", lambda: 2)


def test_system_a_reports_all_stages_and_total_is_their_sum() -> None:
    model = FCSiamDiff(in_channels=3, out_channels=14, base_channels=8)
    result = benchmark_system_a(model, *_pair(), CPU, warmup=1, repetitions=2)
    assert set(result["stages_ms"]) == set(STAGES_A)
    assert len(result["total_ms"]["samples"]) == 2 and result["total_ms"]["mean"] > 0


def test_system_b_reports_six_stages_even_without_rois() -> None:
    detector = FCSiamDiff(in_channels=3, out_channels=1, base_channels=8)
    with torch.no_grad():
        detector.output_layer.bias.fill_(-20.0)  # no component above the threshold
    what_model = FCSiamDiff(in_channels=3, out_channels=14, base_channels=8)
    result = benchmark_system_b(
        detector, what_model, *_pair(), CPU, detector_input_size=(64, 64), warmup=0, repetitions=2
    )
    assert set(result["stages_ms"]) == set(STAGES_B) and len(STAGES_B) == 6


def test_system_b_with_rois_runs_what_model() -> None:
    detector = FCSiamDiff(in_channels=3, out_channels=1, base_channels=8)
    with torch.no_grad():
        detector.output_layer.bias.fill_(20.0)  # whole scene is one component
    what_model = FCSiamDiff(in_channels=3, out_channels=14, base_channels=8)
    result = benchmark_system_b(
        detector, what_model, *_pair(), CPU, detector_input_size=(64, 64), warmup=0, repetitions=1
    )
    assert result["stages_ms"]["what_forward"]["mean"] > 0
