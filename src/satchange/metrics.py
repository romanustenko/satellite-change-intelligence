"""Segmentation, calibration and selective prediction metrics."""

# Recall: Of all real changed pixels, how many did the model find?

# Precision: Of all pixels that were predicted as changed, how many really changed?

# IoU (Intersection over Union): How large is the predicted as changed region compared with the complete region
# marked as changed? It compares not only the size of the region, but also the coordinates of pixels.abs
# INTERSECTION over UNION: two regions were united and then intersected by the region marked as changed.abs


# F1 depends on Recall and Precision. When Recall: 1 and Precision: 0 -> F:0.
# Investigation allows us to conclude whose is this problem.

# Accuracy has to be considered only in the pair with F1 / IoU because the model can detect 0 changed pixels,
# because there are only 5 changed pixels, and have high accuracy, but low F1 or IoU.abs


import torch

class BinaryChangeMetrics:
    """Accumulate confusion counts and compute binary pixel metrics."""

    def __init__(
        self,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the metric accumulator."""

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )


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
        """Add confusion conunts from one batch."""

        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have identical shapes. "
                f"Received {logits.shape} and {targets.shape}."
            )

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= self.threshold
        )

        target_mask = targets >= 0.5

        true_positive = torch.logical_and(
            predictions,
            target_mask,
        ).sum()

        false_positive = torch.logical_and(
            predictions,
            ~target_mask,
        ).sum()

        true_negative = torch.logical_and(
            ~predictions,
            ~target_mask,
        ).sum()

        false_negative = torch.logical_and(
            ~predictions,
            target_mask,
        ).sum()

        self.true_positive += int(
            true_positive.item()
        )
        self.false_positive += int(
            false_positive.item()
        )
        self.false_negative += int(
            false_negative.item()
        )
        self.true_negative += int(
            true_negative.item()
        )

    def compute(
        self,
    ) -> dict[str, float | int]:
        """Compute metrics from all accumulated confusion counts."""

        tp = self.true_positive
        fp = self.false_positive
        tn = self.true_negative
        fn = self.false_negative

        precision_denominator = tp + fp
        recall_denominator = tp + fn
        f1_denominator = 2 * tp + fp + fn
        iou_denominator = tp + fp + fn
        total_pixels = tp + fp + fn + tn

        precision = (
            tp / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            tp / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1 = (
            2 * tp / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        iou = (
            tp / iou_denominator
            if iou_denominator > 0
            else 0.0
        )

        accuracy = (
            (tp + tn) / total_pixels
            if total_pixels > 0
            else 0.0
        )

        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
        }