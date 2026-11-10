"""Per-optimizer-step learning-rate schedule of the Research 2 recipe (D-057).

A pure function of the step counter, so exact resume needs no scheduler state: linear warm-up over
5 % of the step budget (at least one step), then cosine decay of the multiplier towards 0 at the
end of the budget. The peak learning rate itself is tuned per model by HPO (D-049).
"""

import math

WARMUP_FRACTION = 0.05


def warmup_cosine_factor(
    step: int, *, total_steps: int, warmup_fraction: float = WARMUP_FRACTION
) -> float:
    """LR multiplier for the optimizer step with 0-based index ``step`` of ``total_steps``."""
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError(f"total_steps must be a positive integer, got {total_steps!r}.")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError(f"warmup_fraction must be in [0, 1), got {warmup_fraction}.")
    if type(step) is not int or not 0 <= step < total_steps:
        raise ValueError(f"step must be an integer in [0, {total_steps}), got {step!r}.")
    warmup_steps = max(1, math.ceil(warmup_fraction * total_steps)) if warmup_fraction else 0
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


__all__ = ["WARMUP_FRACTION", "warmup_cosine_factor"]
