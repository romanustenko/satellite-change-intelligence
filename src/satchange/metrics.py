"""Reusable binary change-detection metrics."""

import torch


def binary_metrics_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict[str, float | int]:
    """Compute binary pixel metrics from globally accumulated counts."""

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    total_pixels = true_positive + false_positive + false_negative + true_negative

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": (
            true_positive / precision_denominator if precision_denominator > 0 else 0.0
        ),
        "recall": true_positive / recall_denominator if recall_denominator > 0 else 0.0,
        "f1": 2 * true_positive / f1_denominator if f1_denominator > 0 else 0.0,
        "iou": true_positive / iou_denominator if iou_denominator > 0 else 0.0,
        "accuracy": (
            (true_positive + true_negative) / total_pixels if total_pixels > 0 else 0.0
        ),
    }


class BinaryChangeMetrics:
    """Accumulate global confusion counts and compute binary pixel metrics."""

    def __init__(
        self,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the metric accumulator."""

        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated confusion counts."""

        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Convert logits to probabilities and add one batch of counts."""

        self.update_probabilities(torch.sigmoid(logits), targets)

    @torch.no_grad()
    def update_probabilities(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Add one batch of counts from precomputed probabilities."""

        if probabilities.shape != targets.shape:
            raise ValueError(
                "Probabilities and targets must have identical shapes. "
                f"Received {probabilities.shape} and {targets.shape}."
            )

        predictions = probabilities >= self.threshold
        target_mask = targets >= 0.5

        self.true_positive += int(torch.logical_and(predictions, target_mask).sum().item())
        self.false_positive += int(torch.logical_and(predictions, ~target_mask).sum().item())
        self.false_negative += int(torch.logical_and(~predictions, target_mask).sum().item())
        self.true_negative += int(torch.logical_and(~predictions, ~target_mask).sum().item())

    def compute(self) -> dict[str, float | int]:
        """Compute metrics from all accumulated confusion counts."""

        return binary_metrics_from_counts(
            true_positive=self.true_positive,
            false_positive=self.false_positive,
            false_negative=self.false_negative,
            true_negative=self.true_negative,
        )
