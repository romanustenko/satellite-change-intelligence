"""Training-step FLOP accounting (R2-V3-T01): hand-computed conv/linear and analytic kernels."""

import pytest
import torch
from torch import nn

from satchange.flop_registry import (
    ANALYTIC_FLOP_FORMULAS,
    record_analytic_flops,
    recording_analytic_flops,
)
from satchange.models.selective_scan import selective_scan
from satchange.semantic.compute import count_training_step_flops, steps_for_budget


def _sum_loss(output: torch.Tensor) -> torch.Tensor:
    return output.sum()


def test_two_linear_layers_match_hand_count():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
    flops = count_training_step_flops(model, (torch.randn(4, 5),), _sum_loss)
    forward = 2 * 4 * 5 * 3 + 2 * 4 * 3 * 2
    # layer 2: grad input + grad weight; layer 1: grad weight only (input needs no grad)
    backward = 2 * (2 * 4 * 3 * 2) + 2 * 4 * 5 * 3
    assert (flops.forward, flops.backward, flops.total) == (forward, backward, forward + backward)
    assert flops.forward_analytic == flops.backward_analytic == 0
    assert flops.per_sample == pytest.approx((forward + backward) / 4)


def test_convolution_matches_hand_count_and_model_is_untouched():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(2, 4, kernel_size=3, padding=1), nn.BatchNorm2d(4))
    running_mean = model[1].running_mean.clone()
    flops = count_training_step_flops(model, (torch.randn(2, 2, 6, 6),), _sum_loss)
    conv = 2 * 2 * 4 * 6 * 6 * 2 * 3 * 3  # 2 * B * Cout * H * W * Cin * k * k
    assert flops.forward == conv and flops.backward == conv  # grad weight only
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(model[1].running_mean, running_mean)
    record = flops.to_record()
    assert record["total"] == 2 * conv and record["batch_size"] == 2


class _ScanModel(nn.Module):
    def __init__(self, channels: int = 4, state: int = 3) -> None:
        super().__init__()
        self.A = nn.Parameter(-torch.ones(channels, state))
        self.D = nn.Parameter(torch.ones(channels))

    def forward(self, u: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        return selective_scan(u, u, self.A, bc, bc, self.D, delta_softplus=True)


def test_selective_scan_adds_registered_analytic_flops():
    batch, channels, state, length = 2, 4, 3, 5
    flops = count_training_step_flops(
        _ScanModel(channels, state),
        (torch.randn(batch, channels, length), torch.randn(batch, 2, state, length)),
        _sum_loss,
    )
    forward = 9 * batch * length * channels * state + batch * channels * length
    assert flops.forward_analytic == forward
    assert flops.backward_analytic == 2 * forward
    assert flops.analytic_by_op == {
        "forward:selective_scan": forward,
        "backward:selective_scan": 2 * forward,
    }


def test_recorder_is_noop_outside_and_rejects_unknown_ops():
    record_analytic_flops("selective_scan", needs_backward=True, batch=1, length=1, channels=1,
                          state=1, with_d=False)
    with recording_analytic_flops() as recorder:
        with torch.no_grad():
            selective_scan(torch.ones(1, 2, 3), torch.ones(1, 2, 3), -torch.ones(2, 1),
                           torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3))
        assert recorder.total_forward == 9 * 3 * 2 and recorder.total_backward == 0
        with pytest.raises(KeyError, match="No analytic FLOP formula"):
            record_analytic_flops("mystery", needs_backward=False)
    assert "selective_scan" in ANALYTIC_FLOP_FORMULAS


def test_input_validation_and_step_budget():
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="at least one"):
        count_training_step_flops(model, (), _sum_loss)
    with pytest.raises(ValueError, match="batch dimension"):
        count_training_step_flops(model, (torch.randn(2, 2), torch.randn(3, 2)), _sum_loss)
    with pytest.raises(ValueError, match="scalar tensor"):
        count_training_step_flops(model, (torch.randn(2, 2),), lambda out: out)
    assert steps_for_budget(100.0, 30) == 4 and steps_for_budget(90, 30) == 3
    with pytest.raises(ValueError, match="positive"):
        steps_for_budget(0, 10)
