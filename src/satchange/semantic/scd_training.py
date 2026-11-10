"""Dense semantic change detection adapters (loss, epoch runner); orchestration: training.py.

Logit layout: ``[B, 2 * K, H, W]`` with K = 7 classes (0 = no change). Channels ``0..K-1`` are the
T1 map, ``K..2K-1`` the T2 map. Selection score = SeK on the validation split (D-014).
"""

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from satchange.execution import precision_context
from satchange.semantic.scd_losses import (
    MambaSCDLoss,
    SCanNetLoss,
    SCanNetPseudoLabeler,
    SiamUNetLoss,
)
from satchange.semantic.scd_metrics import SemanticChangeMetrics
from satchange.semantic.scd_outputs import SemanticArgmax, maps_from_change_and_semantic
from satchange.step_budget import StepBudget

TASK = "semantic_change_detection"
NUM_CLASSES = 7
SELECTION_KEY = "sek"
SELECTION_RULE = "SeK on the pooled T1+T2 confusion matrix (validation); earlier epoch wins ties"


class SemanticSCDLoss(nn.Module):
    """Sum of the T1 and T2 pixel-wise (optionally class-weighted) cross-entropy terms."""

    def __init__(self, class_weights: list[float] | None = None) -> None:
        super().__init__()
        if class_weights is not None and (
            len(class_weights) != NUM_CLASSES or any(w <= 0 for w in class_weights)
        ):
            raise ValueError(f"class_weights needs {NUM_CLASSES} positive values.")
        self.register_buffer(
            "weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
        )

    def forward(
        self, logits: torch.Tensor, before_label: torch.Tensor, after_label: torch.Tensor
    ) -> torch.Tensor:
        first, second = split_logits(logits)
        weights = self.weights
        assert weights is None or isinstance(weights, torch.Tensor)
        return _cross_entropy(first, before_label, weights) + _cross_entropy(
            second, after_label, weights
        )


def _cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None
) -> torch.Tensor:
    """Weighted-mean pixel CE from deterministic ops (``nll_loss2d`` is nondeterministic on CUDA).

    Equals ``F.cross_entropy(logits, labels, weight=weights)`` with ``reduction="mean"``.
    """
    log_probability = F.log_softmax(logits.float(), dim=1)
    one_hot = F.one_hot(labels, NUM_CLASSES).permute(0, 3, 1, 2).to(log_probability.dtype)
    negative = -(log_probability * one_hot).sum(dim=1)
    if weights is None:
        return negative.mean()
    pixel_weights = weights.to(log_probability.dtype)[labels]
    return (negative * pixel_weights).sum() / pixel_weights.sum()


def split_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the T1 and T2 class logits, each ``[B,K,H,W]``."""
    if logits.ndim != 4 or logits.shape[1] != 2 * NUM_CLASSES:
        raise ValueError(
            f"Semantic logits must be [B,{2 * NUM_CLASSES},H,W], got {tuple(logits.shape)}."
        )
    return logits[:, :NUM_CLASSES], logits[:, NUM_CLASSES:]


def predict_maps(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax T1/T2 class maps ``[B,H,W]`` (int64)."""
    first, second = split_logits(logits)
    return first.argmax(dim=1), second.argmax(dim=1)


MULTI_HEAD_LOSSES = (MambaSCDLoss, SCanNetLoss, SiamUNetLoss)


def is_multi_head_loss(loss_function: nn.Module) -> bool:
    """True for the per-model losses of the multi-head SCD models (``semantic.scd_losses``)."""
    return isinstance(loss_function, MULTI_HEAD_LOSSES)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    *,
    precision: str = "fp32",
    optimizer: torch.optim.Optimizer | None = None,
    budget: StepBudget | None = None,
    semantic_argmax: str | None = None,
    pseudo_labeler: SCanNetPseudoLabeler | None = None,
) -> dict[str, Any]:
    """Fit (``optimizer`` given) or evaluate one split; metrics come from pooled counts.

    With a ``budget`` the fit stops as soon as its optimizer-step limit is reached.

    Two model contracts: the joint 14-channel layout with ``SemanticSCDLoss`` (argmax decoding), or
    a multi-head model returning ``(change, semantic_t1, semantic_t2)`` with its per-model loss
    (``semantic.scd_losses``); then ``semantic_argmax`` (``"land_cover"`` per D-049) is required
    and decides how changed pixels are decoded. ``pseudo_labeler`` (SCanNet ``L_psd``, training
    only, active once its teacher exists) supplies pseudo labels to ``SCanNetLoss``.
    """
    training = optimizer is not None
    if budget is not None and not training:
        raise ValueError("A step budget applies only when fitting (optimizer given).")
    multi_head = is_multi_head_loss(loss_function)
    if multi_head and semantic_argmax is None:
        raise ValueError("Multi-head SCD models need an explicit semantic_argmax (D-049).")
    if not multi_head and (semantic_argmax is not None or pseudo_labeler is not None):
        raise ValueError("semantic_argmax and pseudo_labeler apply only to multi-head models.")
    if pseudo_labeler is not None and not isinstance(loss_function, SCanNetLoss):
        raise ValueError("Pseudo labels are part of the SCanNet loss only.")
    if pseudo_labeler is not None and training and budget is None:
        raise ValueError("Pseudo labels start at an optimizer step; a step budget is required.")
    model.train(training)
    metrics = SemanticChangeMetrics(NUM_CLASSES)
    total_loss, total_samples = 0.0, 0
    for batch in loader:
        if budget is not None and budget.exhausted:
            break
        before = batch["before"].to(device, non_blocking=True)
        after = batch["after"].to(device, non_blocking=True)
        label_1 = batch["before_label"].to(device, non_blocking=True)
        label_2 = batch["after_label"].to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        pseudo: dict[str, torch.Tensor] = {}
        if training and pseudo_labeler is not None:
            assert budget is not None
            pseudo_labeler.before_step(model, budget.optimizer_steps)
        if training and pseudo_labeler is not None and pseudo_labeler.active:
            with precision_context(device, precision):
                pseudo_map = pseudo_labeler(before, after, label_1)
            pseudo = {"pseudo_before": pseudo_map, "pseudo_after": pseudo_map}
        with torch.set_grad_enabled(training), precision_context(device, precision):
            raw_outputs = model(before, after)
        if multi_head:
            outputs = tuple(head.float() for head in raw_outputs)
            with torch.set_grad_enabled(training):
                loss = loss_function(outputs, label_1, label_2, **pseudo)
        else:
            logits = raw_outputs.float()
            with torch.set_grad_enabled(training):
                loss = loss_function(logits, label_1, label_2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Semantic loss is not finite: {loss.item()}.")
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            if budget is not None:
                budget.take(before.shape[0])
        with torch.no_grad():
            if multi_head:
                pred_1, pred_2 = maps_from_change_and_semantic(
                    outputs[0], outputs[1], outputs[2],
                    semantic_argmax=cast(SemanticArgmax, semantic_argmax),
                )
            else:
                pred_1, pred_2 = predict_maps(logits)
            metrics.update(pred_1, label_1)
            metrics.update(pred_2, label_2)
        total_loss += loss.item() * before.shape[0]
        total_samples += before.shape[0]
    if total_samples == 0:
        raise RuntimeError("Semantic data loader is empty.")
    result: dict[str, Any] = {"loss": total_loss / total_samples, **metrics.compute()}
    result["selection_score"] = result[SELECTION_KEY]
    result["selection_rule"] = SELECTION_RULE
    result["samples"] = total_samples
    return result
