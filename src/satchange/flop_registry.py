"""Analytic FLOP formulas for operations ``torch.utils.flop_counter`` cannot see (D-048).

Custom CUDA kernels (MambaSCD ``selective_scan``) run outside the ATen ops that
``FlopCounterMode`` counts. Such an operation calls :func:`record_analytic_flops` with its
dimensions; while an :class:`AnalyticFlopRecorder` is active the registered formula is evaluated
and stored, otherwise the call is a no-op. Kept free of model/semantic imports so that model code
can use it without import cycles.

Conventions (same for every model): FLOPs count multiply and add separately, as
``FlopCounterMode`` does for matmul/convolution; elementwise ops, normalizations and the optimizer
update are not counted. A registered backward is a declared multiple of the forward.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnalyticFormula:
    """Forward FLOPs as a function of named dimensions; backward = factor x forward."""

    forward: Callable[..., int]
    backward_factor: float
    source: str


def _selective_scan_forward(
    *, batch: int, length: int, channels: int, state: int, with_d: bool
) -> int:
    flops = 9 * batch * length * channels * state
    return flops + (batch * channels * length if with_d else 0)


ANALYTIC_FLOP_FORMULAS: dict[str, AnalyticFormula] = {
    "selective_scan": AnalyticFormula(
        forward=_selective_scan_forward,
        backward_factor=2.0,
        source=(
            "forward 9*B*L*D*N (+B*D*L with D): ChangeMamba vmamba.flops_selective_scan_fn, "
            "state-spaces/mamba issue #110; backward = 2 x forward (convention, as for matmul)"
        ),
    ),
}


@dataclass
class AnalyticFlopRecorder:
    """Accumulates analytic FLOPs per operation name while active."""

    forward: dict[str, int] = field(default_factory=dict)
    backward: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, *, needs_backward: bool, **dimensions: int | bool) -> None:
        """Evaluate the registered formula for one call and accumulate it."""
        formula = ANALYTIC_FLOP_FORMULAS.get(name)
        if formula is None:
            raise KeyError(f"No analytic FLOP formula registered for {name!r}.")
        forward = int(formula.forward(**dimensions))
        self.forward[name] = self.forward.get(name, 0) + forward
        self.calls[name] = self.calls.get(name, 0) + 1
        if needs_backward:
            backward = int(round(forward * formula.backward_factor))
            self.backward[name] = self.backward.get(name, 0) + backward

    @property
    def total_forward(self) -> int:
        return sum(self.forward.values())

    @property
    def total_backward(self) -> int:
        return sum(self.backward.values())


_active: list[AnalyticFlopRecorder] = []


@contextmanager
def recording_analytic_flops() -> Iterator[AnalyticFlopRecorder]:
    """Activate a fresh recorder for the duration of the block (nestable)."""
    recorder = AnalyticFlopRecorder()
    _active.append(recorder)
    try:
        yield recorder
    finally:
        _active.pop()


def record_analytic_flops(name: str, *, needs_backward: bool, **dimensions: int | bool) -> None:
    """Record one call of a custom operation in every active recorder (no-op if none)."""
    for recorder in _active:
        recorder.add(name, needs_backward=needs_backward, **dimensions)


__all__ = [
    "ANALYTIC_FLOP_FORMULAS",
    "AnalyticFlopRecorder",
    "AnalyticFormula",
    "record_analytic_flops",
    "recording_analytic_flops",
]
