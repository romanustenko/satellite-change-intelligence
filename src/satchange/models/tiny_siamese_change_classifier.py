"""Compatibility shim only — the TinySiamese ROI classifier was removed (D-006, R2-P07-T01).

`satchange/semantic/roi.py` imports `validate_unit_tensor` from this path. That file must stay
byte-identical because the Research 1 notebooks 08-11 verify its SHA-256 against recorded H08-H11
manifests. Do not add model code here; move the helper only together with an explicit, documented
update of those hash checks.
"""

import torch


def validate_unit_tensor(value: torch.Tensor, name: str) -> None:
    """Reject non-floating, nonfinite, or out-of-range normalized inputs."""
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point, normalized to [0, 1].")
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} must be finite and in [0, 1]; logits are not probabilities.")
