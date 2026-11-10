"""Training-compute accounting in FLOPs (D-048, R2-V3-T01).

One training step = forward + loss + backward of one batch. ATen matmul/convolution/attention
FLOPs are counted by ``torch.utils.flop_counter.FlopCounterMode`` (forward and backward in
separate passes); custom kernels add their registered analytic FLOPs (``satchange.flop_registry``).
Counting runs on a deep copy of the model, so the caller's weights, gradients and BatchNorm
statistics are untouched. Elementwise ops, normalizations and the optimizer update are not counted
(same convention for every model). Step FLOPs are linear in batch size for these models; the step
budget is ``ceil(flop_budget / flops_per_step)`` optimizer steps.
"""

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from satchange.flop_registry import recording_analytic_flops


@dataclass(frozen=True)
class StepFlops:
    """FLOPs of one training step of one batch."""

    batch_size: int
    forward_counted: int
    backward_counted: int
    forward_analytic: int
    backward_analytic: int
    counted_by_op: dict[str, int]
    analytic_by_op: dict[str, int]

    @property
    def forward(self) -> int:
        return self.forward_counted + self.forward_analytic

    @property
    def backward(self) -> int:
        return self.backward_counted + self.backward_analytic

    @property
    def total(self) -> int:
        return self.forward + self.backward

    @property
    def per_sample(self) -> float:
        return self.total / self.batch_size

    def to_record(self) -> dict[str, Any]:
        """JSON-serializable summary for run records and the compute ledger."""
        record = asdict(self)
        record.update(forward=self.forward, backward=self.backward, total=self.total)
        return record


def _op_counts(counter: FlopCounterMode) -> dict[str, int]:
    return {str(op): int(n) for op, n in counter.get_flop_counts().get("Global", {}).items()}


def count_training_step_flops(
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    loss_fn: Callable[[Any], torch.Tensor],
) -> StepFlops:
    """FLOPs of ``loss_fn(model(*inputs)).backward()`` on a train-mode copy of ``model``."""
    if not inputs:
        raise ValueError("inputs must contain at least one tensor.")
    batch_size = int(inputs[0].shape[0])
    if batch_size <= 0 or any(int(x.shape[0]) != batch_size for x in inputs):
        raise ValueError("All inputs must share a positive batch dimension.")
    replica = copy.deepcopy(model).train()
    with recording_analytic_flops() as analytic:
        with FlopCounterMode(display=False) as forward_counter:
            loss = loss_fn(replica(*inputs))
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
            raise ValueError("loss_fn must return a scalar tensor that requires grad.")
        with FlopCounterMode(display=False) as backward_counter:
            loss.backward()
    forward_ops = _op_counts(forward_counter)
    backward_ops = _op_counts(backward_counter)
    counted = {f"forward:{k}": v for k, v in forward_ops.items()}
    counted.update({f"backward:{k}": v for k, v in backward_ops.items()})
    analytic_ops = {f"forward:{k}": v for k, v in analytic.forward.items()}
    analytic_ops.update({f"backward:{k}": v for k, v in analytic.backward.items()})
    return StepFlops(
        batch_size=batch_size,
        forward_counted=sum(forward_ops.values()),
        backward_counted=sum(backward_ops.values()),
        forward_analytic=analytic.total_forward,
        backward_analytic=analytic.total_backward,
        counted_by_op=counted,
        analytic_by_op=analytic_ops,
    )


def steps_for_budget(flop_budget: float, flops_per_step: int) -> int:
    """Optimizer steps that spend ``flop_budget`` (rounded up; the last step completes)."""
    if flop_budget <= 0 or flops_per_step <= 0:
        raise ValueError("flop_budget and flops_per_step must be positive.")
    return math.ceil(flop_budget / flops_per_step)


__all__ = ["StepFlops", "count_training_step_flops", "steps_for_budget"]
