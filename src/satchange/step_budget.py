"""Optimizer-step budget for compute-fair training (D-046/D-048, R2-V3-T02).

Training stops after a pre-computed number of optimizer steps, never on a timer. The counter is
shared by the binary and the semantic epoch runners and is stored in checkpoints so that an exact
resume continues the same budget.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class StepBudget:
    """Optimizer steps and samples consumed against an optional step limit."""

    max_optimizer_steps: int | None = None
    optimizer_steps: int = 0
    samples_seen: int = 0

    def __post_init__(self) -> None:
        limit = self.max_optimizer_steps
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_optimizer_steps must be None or a positive integer.")
        for name in ("optimizer_steps", "samples_seen"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if limit is not None and self.optimizer_steps > limit:
            raise ValueError("optimizer_steps already exceeds max_optimizer_steps.")

    @property
    def exhausted(self) -> bool:
        return self.max_optimizer_steps is not None and (
            self.optimizer_steps >= self.max_optimizer_steps
        )

    @property
    def remaining(self) -> int | None:
        if self.max_optimizer_steps is None:
            return None
        return self.max_optimizer_steps - self.optimizer_steps

    def take(self, batch_size: int) -> None:
        """Account for one completed optimizer step on ``batch_size`` samples."""
        if self.exhausted:
            raise RuntimeError("Step budget is exhausted; no further optimizer step is allowed.")
        self.optimizer_steps += 1
        self.samples_seen += int(batch_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_optimizer_steps": self.max_optimizer_steps,
            "optimizer_steps": self.optimizer_steps,
            "samples_seen": self.samples_seen,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "StepBudget":
        unknown = set(state) - {"max_optimizer_steps", "optimizer_steps", "samples_seen"}
        if unknown:
            raise ValueError(f"Unknown step-budget keys: {sorted(unknown)}")
        return cls(**state)


__all__ = ["StepBudget"]
