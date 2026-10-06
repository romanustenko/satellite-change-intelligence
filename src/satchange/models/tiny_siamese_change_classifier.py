"""A small shared CNN for semantic classification of native-scale changed ROIs."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def validate_classes(classes: Sequence[str]) -> tuple[str, ...]:
    """Validate the ordered, persisted taxonomy; no dataset classes are inferred."""
    if isinstance(classes, str) or len(classes) < 2:
        raise ValueError("classes must contain at least two ordered class names.")
    if any(not isinstance(name, str) or not name.strip() for name in classes):
        raise ValueError("Every class name must be a nonempty string.")
    if len(set(classes)) != len(classes):
        raise ValueError("Class names must be unique.")
    return tuple(classes)


def validate_unit_tensor(value: torch.Tensor, name: str) -> None:
    """Reject non-floating, nonfinite, or out-of-range normalized inputs."""
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point, normalized to [0, 1].")
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} must be finite and in [0, 1]; logits are not probabilities.")


def masked_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Area-reduce [B,1,H,W] weights and pool [B,D,h,w] features in FP32.

    Area reduction retains even one positive source pixel on odd-sized grids.
    Empty support is an error, rather than an implicit global-pooling fallback.
    """
    if features.ndim != 4 or mask.ndim != 4 or mask.shape[:2] != (features.shape[0], 1):
        raise ValueError("Pooling expects features [B,D,h,w] and aligned mask [B,1,H,W].")
    if features.device != mask.device:
        raise ValueError("Features and mask must use the same device.")
    validate_unit_tensor(mask, "mask")
    weights = F.interpolate(mask.float(), size=features.shape[-2:], mode="area")
    denominator = weights.sum(dim=(-2, -1))
    if (denominator <= 0).any():
        raise ValueError("Every ROI requires a nonempty effective mask.")
    return (features.float() * weights).sum(dim=(-2, -1)) / (denominator + 1e-8)


class TinySiameseChangeClassifier(nn.Module):
    """Shared RGB backbone and head, with limited local receptive-field context.

    forward(before, after, mask): images [B,3,H,W], mask [B,1,H,W],
    raw before/after logits [B,C]. RGB remains unmasked and at native scale.
    Two pooling stages produce floor(H/4) x floor(W/4) feature grids.
    Both dates share one BatchNorm update by concatenating on the batch axis.
    """

    def __init__(self, classes: Sequence[str], dropout: float = 0.1) -> None:
        super().__init__()
        self.classes = validate_classes(classes)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.dropout = float(dropout)
        layers: list[nn.Module] = []
        for index, (incoming, outgoing) in enumerate(((3, 32), (32, 64), (64, 128))):
            layers.extend(
                [
                    nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.ReLU(),
                ]
            )
            if index < 2:
                layers.append(nn.MaxPool2d(2))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, len(self.classes))
        )

    def forward(
        self, before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return two semantic-logit tensors using exactly the same parameters."""
        if before.ndim != 4 or before.shape[1] != 3 or before.shape != after.shape:
            raise ValueError("Images must be aligned [B,3,H,W] tensors.")
        if before.shape[0] == 0 or min(before.shape[-2:]) < 4:
            raise ValueError("Images require a nonempty batch and height/width of at least 4.")
        if mask.shape != (before.shape[0], 1, *before.shape[-2:]):
            raise ValueError("mask must be aligned [B,1,H,W].")
        if before.device != after.device or before.device != mask.device:
            raise ValueError("Images and mask must use the same device.")
        if before.dtype != torch.float32 or after.dtype != torch.float32:
            raise ValueError("RGB inputs must be float32 normalized to [0, 1].")
        for name, value in (("before", before), ("after", after), ("mask", mask)):
            validate_unit_tensor(value, name)
        if (mask.float().sum(dim=(-2, -1)) <= 0).any():
            raise ValueError("Every ROI requires a nonempty effective mask.")
        features = self.backbone(torch.cat((before, after), dim=0))
        vectors = masked_pool(features, torch.cat((mask, mask), dim=0))
        before_logits, after_logits = self.head(vectors).chunk(2, dim=0)
        return {"before_logits": before_logits, "after_logits": after_logits}
