"""Convert model-specific SCD heads to the common T1/T2 7-class maps (R2-V2-T02).

Common evaluation format (``scd_metrics.SemanticChangeMetrics``): two int64 maps ``[B,H,W]`` with
ids 0..6, 0 = no change. Two head layouts exist:

* **joint** — ``[B,14,H,W]`` T1+T2 class logits (``scd_training`` layout); use
  ``scd_training.predict_maps``.
* **change + semantic** (MambaSCD, SCanNet, Siamese U-Net) — a change head (1 channel = sigmoid
  logit, or 2 channels = softmax logits [no change, change]) and per-date semantic logits with 7
  channels (channel 0 = no change, usually ignored in training) or 6 channels (land cover only).
  A pixel is changed when ``sigmoid > change_threshold`` (1 channel) or ``argmax == 1`` (2 channels;
  equals softmax > 0.5). Changed pixels take the per-date semantic argmax; unchanged pixels are 0.

``semantic_argmax`` must be given explicitly: ``"all"`` takes the argmax over all 7 channels as
the ChangeMamba and SCanNet inference code do (a changed pixel can then still receive class 0);
``"land_cover"`` restricts the argmax to classes 1..6. Which rule the protocol uses is a protocol
decision (open, see ``docs/research2/DECISION_LOG.md`` D-049).
"""

from typing import Literal

import torch

NUM_CLASSES = 7
LAND_COVER_CLASSES = NUM_CLASSES - 1
SemanticArgmax = Literal["all", "land_cover"]


def change_mask_from_logits(
    change_logits: torch.Tensor, *, change_threshold: float = 0.5
) -> torch.Tensor:
    """Boolean ``[B,H,W]`` change mask from a 1-channel (sigmoid) or 2-channel (softmax) head."""
    if change_logits.ndim != 4 or change_logits.shape[1] not in (1, 2):
        raise ValueError(
            f"Change logits must be [B,1,H,W] or [B,2,H,W], got {tuple(change_logits.shape)}."
        )
    if change_logits.shape[1] == 2:
        return change_logits.float().argmax(dim=1) == 1
    if not 0.0 < change_threshold < 1.0:
        raise ValueError(f"change_threshold must be in (0, 1), got {change_threshold}.")
    return torch.sigmoid(change_logits.float()[:, 0]) > change_threshold


def semantic_class_map(
    semantic_logits: torch.Tensor, *, semantic_argmax: SemanticArgmax
) -> torch.Tensor:
    """Per-date class ids ``[B,H,W]`` in 0..6 from 7- or 6-channel semantic logits."""
    if semantic_argmax not in ("all", "land_cover"):
        raise ValueError(f"semantic_argmax must be 'all' or 'land_cover', got {semantic_argmax!r}.")
    if semantic_logits.ndim != 4 or semantic_logits.shape[1] not in (
        NUM_CLASSES,
        LAND_COVER_CLASSES,
    ):
        raise ValueError(
            f"Semantic logits must be [B,{NUM_CLASSES},H,W] or [B,{LAND_COVER_CLASSES},H,W], "
            f"got {tuple(semantic_logits.shape)}."
        )
    logits = semantic_logits.float()
    if logits.shape[1] == LAND_COVER_CLASSES:
        return logits.argmax(dim=1) + 1
    if semantic_argmax == "all":
        return logits.argmax(dim=1)
    return logits[:, 1:].argmax(dim=1) + 1


def maps_from_change_and_semantic(
    change_logits: torch.Tensor,
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    *,
    semantic_argmax: SemanticArgmax,
    change_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Common T1/T2 maps (int64 ``[B,H,W]``, 0 = no change) from change + per-date heads."""
    change = change_mask_from_logits(change_logits, change_threshold=change_threshold)
    maps = []
    for logits in (before_logits, after_logits):
        if logits.shape[0] != change.shape[0] or logits.shape[2:] != change.shape[1:]:
            raise ValueError(
                f"Semantic logits {tuple(logits.shape)} do not match the change head "
                f"{tuple(change_logits.shape)}."
            )
        classes = semantic_class_map(logits, semantic_argmax=semantic_argmax)
        maps.append(torch.where(change, classes, torch.zeros_like(classes)).long())
    return maps[0], maps[1]


__all__ = [
    "LAND_COVER_CLASSES",
    "NUM_CLASSES",
    "change_mask_from_logits",
    "maps_from_change_and_semantic",
    "semantic_class_map",
]
