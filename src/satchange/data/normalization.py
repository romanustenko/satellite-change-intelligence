"""Explicit, per-model input normalization for image pairs (Research 2 v2, R2-V2-T01).

Loaders return RGB in [0, 1] by default; a model with an ImageNet-pretrained encoder opts in by
name. Applied to ``before``/``after`` after the geometric training transform (per-channel affine
maps commute with flips and 90-degree rotations), never to labels.
"""

from typing import Any

import torch

INPUT_NORMALIZATIONS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # torchvision ImageNet-1k statistics for RGB in [0, 1]
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
IMAGE_KEYS = ("before", "after")


def validate_input_normalization(name: str | None) -> str | None:
    """Return ``name`` if it is ``None`` or a known normalization, else raise."""
    if name is not None and name not in INPUT_NORMALIZATIONS:
        raise ValueError(
            f"Unknown input normalization {name!r}; expected None or one of "
            f"{sorted(INPUT_NORMALIZATIONS)}"
        )
    return name


def normalize_image(image: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize one ``[3,H,W]`` (or ``[B,3,H,W]``) RGB tensor in [0, 1]."""
    validate_input_normalization(name)
    if image.dim() not in (3, 4) or image.shape[-3] != 3:
        raise ValueError(f"Expected an RGB tensor [3,H,W] or [B,3,H,W], got {tuple(image.shape)}")
    mean, std = INPUT_NORMALIZATIONS[name]
    shape = (3, 1, 1)
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(shape)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(shape)
    return (image - mean_tensor) / std_tensor


def normalize_sample(sample: dict[str, Any], name: str | None) -> dict[str, Any]:
    """Return ``sample`` with normalized ``before``/``after`` (unchanged when ``name`` is None)."""
    if validate_input_normalization(name) is None:
        return sample
    assert name is not None
    return {
        key: normalize_image(value, name) if key in IMAGE_KEYS else value
        for key, value in sample.items()
    }


__all__ = [
    "INPUT_NORMALIZATIONS",
    "normalize_image",
    "normalize_sample",
    "validate_input_normalization",
]
