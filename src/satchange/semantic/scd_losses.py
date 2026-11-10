"""Per-model training losses of the three SCD models (R2-V4-T07, D-049).

D-049 (USER, 2026-11-10): each model keeps the loss of its method; recipe (data, optimizer,
schedule, selection) is shared. All three models are multi-head: ``forward(before, after)`` returns
``(change, semantic_t1, semantic_t2)`` and the losses take that tuple plus the dense labels
(``before_label`` / ``after_label``: int64 ``[B,H,W]``, ids 0..6, 0 = no change; the change
target is
``before_label > 0``, which the SECOND loader guarantees equal to the binary mask).

* ``MambaSCDLoss`` — ChangeMamba ``tasks/scd.py`` (Apache-2.0, commit ``9ce9cec``; identical in
  the original script of ``b834b2a``): CE + Lovasz-softmax on the 2-channel change head, CE +
  Lovasz on both semantic heads with unchanged pixels ignored, and an MSE similarity term between
  the two softmaxes on unchanged pixels; weights ``CE_cd + 0.5 (CE_1 + CE_2 + 0.5 sim) +
  0.75 (Lov_cd + 0.5 (Lov_1 + Lov_2))``.
* ``SCanNetLoss`` — SCanNet ``train_SCD_psd.py`` (code, no licence, re-implemented from the
  description in the paper and the loss formulas): ``0.5 (CE_1 + CE_2) + weighted BCE(change) +
  L_sc``.
  The paper (Eq. 10) lists ``L_sem + L_psd + L_sc`` and no change-head term; the weights of the
  change-head loss (positive 0.25 / negative 0.75) and the 0.5 factor are the authors' code
  (fidelity choice, paper silent). ``SCanNetPseudoLabeler`` produces the pseudo labels of ``L_psd``
  for unchanged pixels; *when* it is switched on is an engine/protocol decision (open).
* ``SiamUNetLoss`` — no published loss exists for our U-Net; D-049: change CE + semantic CE on
  changed pixels. Unit weights: ``BCE(change) + CE_1 + CE_2`` (DELEGATED choice, D-053).

All terms avoid nondeterministic CUDA kernels (``nll_loss2d``; see ``scd_training._cross_entropy``).
The Lovasz-softmax extension follows Berman, Triki, Blaschko (CVPR 2018, MIT licence) and uses the
same conventions as ChangeMamba's call (batch-level, classes present in the valid labels).
"""

import math
from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

NUM_CLASSES = 7
LAND_COVER_CLASSES = NUM_CLASSES - 1

MultiHeadOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# --------------------------------------------------------------------------- validation helpers


def _check_labels(
    outputs: MultiHeadOutputs, before_label: torch.Tensor, after_label: torch.Tensor
) -> None:
    if not isinstance(outputs, tuple | list) or len(outputs) != 3:
        raise ValueError("Outputs must be a (change, semantic_t1, semantic_t2) tuple.")
    change, first, second = outputs
    if change.ndim != 4 or change.shape[1] not in (1, 2):
        raise ValueError(f"Change logits must be [B,1|2,H,W], got {tuple(change.shape)}.")
    for name, logits in (("semantic_t1", first), ("semantic_t2", second)):
        if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"{name} logits must be [B,{NUM_CLASSES},H,W], got {tuple(logits.shape)}."
            )
        if logits.shape[0] != change.shape[0] or logits.shape[2:] != change.shape[2:]:
            raise ValueError(f"{name} logits {tuple(logits.shape)} do not match the change head.")
    expected = (change.shape[0], *change.shape[2:])
    for name, label in (("before_label", before_label), ("after_label", after_label)):
        if label.dtype != torch.int64 or tuple(label.shape) != expected:
            raise ValueError(
                f"{name} must be int64 {expected}, got {label.dtype} {tuple(label.shape)}."
            )


def _change_target(before_label: torch.Tensor) -> torch.Tensor:
    return (before_label > 0).long()


# --------------------------------------------------------------------------- shared loss terms


def masked_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean pixel CE over ``valid`` pixels (all when ``None``) from deterministic ops.

    Equals ``F.cross_entropy(logits, labels, ignore_index=...)`` on the valid pixels; returns a
    graph-connected zero when no pixel is valid (the reference would give NaN).
    """
    classes = logits.shape[1]
    log_probability = F.log_softmax(logits.float(), dim=1)
    one_hot = F.one_hot(labels, classes).permute(0, 3, 1, 2).to(log_probability.dtype)
    negative = -(log_probability * one_hot).sum(dim=1)
    if valid is None:
        return negative.mean()
    weights = valid.to(negative.dtype)
    return (negative * weights).sum() / weights.sum().clamp_min(1.0)


def lovasz_softmax(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Multi-class Lovasz-softmax of softmax(``logits``), batch level, present classes only.

    ``logits`` ``[B,C,H,W]``, ``labels`` ``[B,H,W]``; pixels where ``valid`` is false are dropped.
    Returns a graph-connected zero when no valid pixel (or no class) remains.
    """
    classes = logits.shape[1]
    probabilities = F.softmax(logits.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, classes)
    flat_labels = labels.reshape(-1)
    if valid is not None:
        keep = valid.reshape(-1)
        probabilities, flat_labels = probabilities[keep], flat_labels[keep]
    if flat_labels.numel() == 0:
        return probabilities.sum() * 0.0
    foreground = F.one_hot(flat_labels, classes).to(probabilities.dtype)  # [P, C]
    errors = (foreground - probabilities).abs()
    errors_sorted, permutation = torch.sort(errors, dim=0, descending=True)
    foreground_sorted = torch.gather(foreground, 0, permutation)
    total = foreground.sum(dim=0, keepdim=True)
    intersection = total - foreground_sorted.cumsum(dim=0)
    union = total + (1.0 - foreground_sorted).cumsum(dim=0)
    jaccard = 1.0 - intersection / union
    gradient = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]], dim=0)
    per_class = (errors_sorted * gradient).sum(dim=0)
    present = total[0] > 0
    return per_class[present].sum() / present.sum()


def similarity_mse(
    first: torch.Tensor, second: torch.Tensor, unchanged: torch.Tensor
) -> torch.Tensor:
    """ChangeMamba similarity term: MSE of the two softmaxes, masked to unchanged pixels."""
    mask = unchanged.to(first.dtype).unsqueeze(1)
    return F.mse_loss(
        F.softmax(first.float(), dim=1) * mask, F.softmax(second.float(), dim=1) * mask
    )


# --------------------------------------------------------------------------- MambaSCD


class MambaSCDLoss(nn.Module):
    """ChangeMamba SCD loss (see module docstring); change head has 2 channels."""

    def terms(
        self, outputs: MultiHeadOutputs, before_label: torch.Tensor, after_label: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        _check_labels(outputs, before_label, after_label)
        change, first, second = outputs
        if change.shape[1] != 2:
            raise ValueError("MambaSCDLoss needs a 2-channel change head.")
        target = _change_target(before_label)
        changed_1, changed_2 = before_label > 0, after_label > 0
        parts = {
            "ce_change": masked_cross_entropy(change, target),
            "lovasz_change": lovasz_softmax(change, target),
            "ce_t1": masked_cross_entropy(first, before_label, changed_1),
            "ce_t2": masked_cross_entropy(second, after_label, changed_2),
            "lovasz_t1": lovasz_softmax(first, before_label, changed_1),
            "lovasz_t2": lovasz_softmax(second, after_label, changed_2),
            "similarity": similarity_mse(first, second, ~changed_1),
        }
        parts["loss"] = (
            parts["ce_change"]
            + 0.5 * (parts["ce_t1"] + parts["ce_t2"] + 0.5 * parts["similarity"])
            + 0.75 * (parts["lovasz_change"] + 0.5 * (parts["lovasz_t1"] + parts["lovasz_t2"]))
        )
        return parts

    def forward(
        self, outputs: MultiHeadOutputs, before_label: torch.Tensor, after_label: torch.Tensor
    ) -> torch.Tensor:
        return self.terms(outputs, before_label, after_label)["loss"]


# --------------------------------------------------------------------------- SCanNet


def weighted_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight_pos: float = 0.25,
    weight_neg: float = 0.75,
) -> torch.Tensor:
    """SCanNet ``weighted_BCE_logits``: per-class mean BCE, positives and negatives weighted."""
    if logits.shape != target.shape:
        raise ValueError(f"Logits {tuple(logits.shape)} and target {tuple(target.shape)} differ.")
    loss = F.binary_cross_entropy_with_logits(
        logits.float().reshape(-1), target.float().reshape(-1), reduction="none"
    )
    flat = target.reshape(-1)
    positive = (flat > 0.5).to(loss.dtype)
    negative = (flat < 0.5).to(loss.dtype)
    positive_count = positive.sum() + 1e-12
    negative_count = negative.sum() + 1e-12
    return (
        weight_pos * positive * loss / positive_count
        + weight_neg * negative * loss / negative_count
    ).sum()


def change_similarity(
    first: torch.Tensor, second: torch.Tensor, changed: torch.Tensor, *, margin: float = 0.1
) -> torch.Tensor:
    """SCanNet ``ChangeSimilarity`` (L_sc): cosine-embedding loss of the two softmax vectors.

    ``first``/``second`` ``[B,C,H,W]`` logits, ``changed`` ``[B,H,W]``. Unchanged pixels pull the
    class distributions together (``1 - cos``), changed pixels push them apart
    (``max(0, cos - margin)``);
    mean over pixels. Same arithmetic as ``torch.nn.CosineEmbeddingLoss``.
    """
    probability_1 = F.softmax(first.float(), dim=1)
    probability_2 = F.softmax(second.float(), dim=1)
    dot = (probability_1 * probability_2).sum(dim=1)
    norm = (
        (probability_1.square().sum(dim=1) + 1e-12) * (probability_2.square().sum(dim=1) + 1e-12)
    ).sqrt()
    cosine = dot / norm
    return torch.where(changed.bool(), (cosine - margin).clamp_min(0.0), 1.0 - cosine).mean()


class SCanNetLoss(nn.Module):
    """``0.5 (CE_1 + CE_2) + weighted BCE(change) + L_sc`` (SCanNet code; see module docstring).

    ``pseudo_before`` / ``pseudo_after`` (optional ``[B,H,W]`` class ids from
    ``SCanNetPseudoLabeler``, 0 elsewhere) fill unchanged pixels of the semantic targets
    (``L_psd``).
    """

    def terms(
        self,
        outputs: MultiHeadOutputs,
        before_label: torch.Tensor,
        after_label: torch.Tensor,
        *,
        pseudo_before: torch.Tensor | None = None,
        pseudo_after: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _check_labels(outputs, before_label, after_label)
        change, first, second = outputs
        if change.shape[1] != 1:
            raise ValueError("SCanNetLoss needs a 1-channel change head.")
        targets = []
        for label, pseudo in ((before_label, pseudo_before), (after_label, pseudo_after)):
            if pseudo is not None:
                if pseudo.shape != label.shape or pseudo.dtype != torch.int64:
                    raise ValueError("Pseudo labels must be int64 and match the label shape.")
                label = label + pseudo * (label == 0)
            targets.append(label)
        change_target = _change_target(before_label)
        parts = {
            "ce_t1": masked_cross_entropy(first, targets[0], targets[0] > 0),
            "ce_t2": masked_cross_entropy(second, targets[1], targets[1] > 0),
            "bce_change": weighted_bce_logits(change[:, 0], change_target.float()),
            "similarity": change_similarity(first[:, 1:], second[:, 1:], change_target),
        }
        parts["loss"] = (
            0.5 * (parts["ce_t1"] + parts["ce_t2"]) + parts["bce_change"] + parts["similarity"]
        )
        return parts

    def forward(
        self,
        outputs: MultiHeadOutputs,
        before_label: torch.Tensor,
        after_label: torch.Tensor,
        *,
        pseudo_before: torch.Tensor | None = None,
        pseudo_after: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.terms(
            outputs,
            before_label,
            after_label,
            pseudo_before=pseudo_before,
            pseudo_after=pseudo_after,
        )["loss"]


PSEUDO_LABEL_START_FRACTION = 0.5  # D-055 (USER): L_psd starts at half of the step budget


def pseudo_label_start_step(max_optimizer_steps: int) -> int:
    """Optimizer step at which SCanNet's ``L_psd`` starts (``ceil`` of half the step budget)."""
    if type(max_optimizer_steps) is not int or max_optimizer_steps < 1:
        raise ValueError(
            f"max_optimizer_steps must be a positive integer, got {max_optimizer_steps!r}."
        )
    return max(1, math.ceil(PSEUDO_LABEL_START_FRACTION * max_optimizer_steps))


def pseudo_label_budget_steps(
    flop_budget: float, *, step_flops: float, teacher_step_flops: float
) -> int:
    """Smallest step count whose FLOPs (incl. teacher passes after the start step) reach the budget.

    D-056 (USER): the teacher's inference FLOPs are charged. With ``N`` steps, ``L_psd`` starts at
    ``pseudo_label_start_step(N)``, so the cost is ``N * step + (N - start) * teacher_step``.
    """
    if not (flop_budget > 0 and step_flops > 0 and teacher_step_flops >= 0):
        raise ValueError("flop_budget and step_flops must be positive, teacher_step_flops >= 0.")

    def cost(steps: int) -> float:
        return steps * step_flops + (steps - pseudo_label_start_step(steps)) * teacher_step_flops

    steps = max(1, math.floor(flop_budget / (step_flops + 0.5 * teacher_step_flops)))
    while steps > 1 and cost(steps - 1) >= flop_budget:
        steps -= 1
    while cost(steps) < flop_budget:
        steps += 1
    return steps


def _percentile(values: torch.Tensor, percent: float) -> float:
    """Linear-interpolation percentile (NumPy default) of a 1-D tensor."""
    ordered = torch.sort(values.double()).values
    position = (ordered.numel() - 1) * percent / 100.0
    low = int(position)
    high = min(low + 1, ordered.numel() - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


class _RunningThresholds:
    """Count-weighted running mean of per-class confidence thresholds (SCanNet ``AverageThred``)."""

    def __init__(self, classes: int, initial: float) -> None:
        self.count = [1.0] * classes
        self.total = [initial] * classes

    def update(self, thresholds: list[float], counts: list[float]) -> None:
        for index, (threshold, count) in enumerate(zip(thresholds, counts, strict=True)):
            self.count[index] += count
            self.total[index] += threshold * count

    def value(self) -> list[float]:
        return [min(max(t / c, 0.5), 0.9) for t, c in zip(self.total, self.count, strict=True)]


class SCanNetPseudoLabeler:
    """Pseudo labels for unchanged pixels (SCanNet ``L_psd``; ``train_SCD_psd.py`` semantics).

    A frozen teacher (a copy of the model) predicts both dates, optionally averaged over the four
    flips (TTA); a pixel gets the teacher class when it is unchanged in the ground truth, both
    dates are confident (class-wise running percentile threshold, clipped to [0.5, 0.9]) and the
    two dates agree. The shared threshold state is updated by date 1 then date 2, as in the code.

    Schedule (D-055, USER): the original switches ``L_psd`` on once validation Fscd > 0.6 and
    re-snapshots the teacher at every Fscd improvement; that does not fit a fixed step budget, so
    here ``L_psd`` starts at the fixed optimizer step ``start_step`` (``before_step`` snapshots the
    model then) and the teacher is re-snapshotted after every validation whose Fscd beats the best
    so far (``after_validation``). ``refresh_teacher`` resets the thresholds.
    """

    def __init__(
        self,
        *,
        start_step: int,
        percentile: float = 0.6,
        tta: bool = True,
        classes: int = NUM_CLASSES,
        flops_per_sample: float | None = None,
    ) -> None:
        if type(start_step) is not int or start_step < 1:
            raise ValueError(f"start_step must be a positive integer, got {start_step!r}.")
        if not 0.0 < percentile < 1.0:
            raise ValueError(f"percentile must be in (0, 1), got {percentile}.")
        self.start_step, self.percentile, self.tta, self.classes = (
            start_step,
            percentile,
            tta,
            classes,
        )
        self.teacher: nn.Module | None = None
        self.thresholds = _RunningThresholds(classes, percentile)
        self.best_fscd = float("-inf")
        # Teacher inference is training compute (D-056): FLOPs per sample of one pseudo-label call.
        if flops_per_sample is not None and not flops_per_sample > 0:
            raise ValueError("flops_per_sample must be positive.")
        self.flops_per_sample = flops_per_sample
        self.teacher_samples = 0

    @property
    def teacher_flops(self) -> float:
        """FLOPs spent so far on teacher passes (needs ``flops_per_sample``)."""
        if self.flops_per_sample is None:
            raise RuntimeError("flops_per_sample was not set; measure it with count_flops.")
        return self.flops_per_sample * self.teacher_samples

    def count_flops(self, model: nn.Module, before: torch.Tensor, after: torch.Tensor) -> float:
        """FLOPs per sample of one pseudo-label call (all TTA passes), set as ``flops_per_sample``.

        Measured with ``FlopCounterMode`` on a throw-away teacher copy of ``model``; the labeler's
        own teacher, thresholds and counters are left untouched.
        """
        from torch.utils.flop_counter import FlopCounterMode

        probe = SCanNetPseudoLabeler(
            start_step=self.start_step,
            percentile=self.percentile,
            tta=self.tta,
            classes=self.classes,
        )
        probe.refresh_teacher(model)
        with FlopCounterMode(display=False) as counter:
            probe(before, after, torch.zeros_like(before[:, 0], dtype=torch.int64))
        self.flops_per_sample = counter.get_total_flops() / before.shape[0]
        return self.flops_per_sample

    def before_step(self, model: nn.Module, optimizer_steps: int) -> None:
        """Snapshot the teacher at the start step (no-op afterwards)."""
        if self.teacher is None and optimizer_steps >= self.start_step:
            self.refresh_teacher(model)

    def after_validation(self, model: nn.Module, fscd: float, optimizer_steps: int) -> None:
        """Track the best validation Fscd; refresh the active teacher when it improved."""
        improved = fscd > self.best_fscd
        self.best_fscd = max(self.best_fscd, fscd)
        if improved and optimizer_steps >= self.start_step:
            self.refresh_teacher(model)

    @property
    def active(self) -> bool:
        return self.teacher is not None

    def refresh_teacher(self, model: nn.Module) -> None:
        import copy

        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher
        self.thresholds = _RunningThresholds(self.classes, self.percentile)

    def _probabilities(
        self, before: torch.Tensor, after: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.teacher is not None
        flips: list[list[int]] = [[]] + ([[2], [3], [2, 3]] if self.tta else [])
        total_1 = total_2 = None
        for dims in flips:
            x1 = torch.flip(before, dims) if dims else before
            x2 = torch.flip(after, dims) if dims else after
            _, out_1, out_2 = cast(MultiHeadOutputs, self.teacher(x1, x2))
            p1, p2 = F.softmax(out_1.float(), dim=1), F.softmax(out_2.float(), dim=1)
            if dims:
                p1, p2 = torch.flip(p1, dims), torch.flip(p2, dims)
            total_1 = p1 if total_1 is None else total_1 + p1
            total_2 = p2 if total_2 is None else total_2 + p2
        assert total_1 is not None and total_2 is not None
        return total_1 / len(flips), total_2 / len(flips)

    def _confident(self, probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        confidence, index = probability.max(dim=1)
        thresholds, counts = [], []
        for cls in range(self.classes):
            selected = confidence[index == cls]
            if selected.numel() > 0:
                thresholds.append(_percentile(selected, 100.0 * self.percentile))
                counts.append(float(selected.numel()))
            else:
                thresholds.append(self.percentile)
                counts.append(0.0)
        self.thresholds.update(thresholds, counts)
        table = torch.tensor(
            self.thresholds.value(), dtype=confidence.dtype, device=confidence.device
        )
        return confidence >= table[index], index

    @torch.no_grad()
    def __call__(
        self, before: torch.Tensor, after: torch.Tensor, before_label: torch.Tensor
    ) -> torch.Tensor:
        """Pseudo class ids ``[B,H,W]`` (int64; 0 = no pseudo label) for unchanged pixels."""
        if self.teacher is None:
            raise RuntimeError("Call refresh_teacher(model) before generating pseudo labels.")
        probability_1, probability_2 = self._probabilities(before, after)
        self.teacher_samples += before.shape[0]
        confident_1, index_1 = self._confident(probability_1)
        confident_2, index_2 = self._confident(probability_2)
        selected = confident_1 & confident_2 & (index_1 == index_2) & (before_label == 0)
        return (index_1 * selected).long()


# --------------------------------------------------------------------------- Siamese U-Net


class SiamUNetLoss(nn.Module):
    """``BCE(change) + CE_1 + CE_2`` with semantic CE on changed pixels only (D-049, D-053)."""

    def terms(
        self, outputs: MultiHeadOutputs, before_label: torch.Tensor, after_label: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        _check_labels(outputs, before_label, after_label)
        change, first, second = outputs
        if change.shape[1] != 1:
            raise ValueError("SiamUNetLoss needs a 1-channel change head.")
        parts = {
            "bce_change": F.binary_cross_entropy_with_logits(
                change[:, 0].float(), (before_label > 0).float()
            ),
            "ce_t1": masked_cross_entropy(first, before_label, before_label > 0),
            "ce_t2": masked_cross_entropy(second, after_label, after_label > 0),
        }
        parts["loss"] = parts["bce_change"] + parts["ce_t1"] + parts["ce_t2"]
        return parts

    def forward(
        self, outputs: MultiHeadOutputs, before_label: torch.Tensor, after_label: torch.Tensor
    ) -> torch.Tensor:
        return self.terms(outputs, before_label, after_label)["loss"]


SCD_LOSS_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "mamba_scd": MambaSCDLoss,
    "scannet": SCanNetLoss,
    "siam_unet_resnet18": SiamUNetLoss,
}


def create_scd_loss(model_name: str) -> nn.Module:
    """The method-specific loss of a Research 2 SCD model."""
    try:
        return SCD_LOSS_REGISTRY[model_name]()
    except KeyError:
        raise ValueError(
            f"Unknown SCD model {model_name!r}; expected one of {sorted(SCD_LOSS_REGISTRY)}."
        ) from None


__all__ = [
    "PSEUDO_LABEL_START_FRACTION",
    "SCD_LOSS_REGISTRY",
    "MambaSCDLoss",
    "SCanNetLoss",
    "SCanNetPseudoLabeler",
    "SiamUNetLoss",
    "change_similarity",
    "create_scd_loss",
    "lovasz_softmax",
    "masked_cross_entropy",
    "pseudo_label_budget_steps",
    "pseudo_label_start_step",
    "similarity_mse",
    "weighted_bce_logits",
]
