"""Semantic change detection metrics for SECOND (Sek, mIoU, Fscd), reference-compatible.

Reference: ChangeMamba ``changedetection/evaluation/scd.py`` (``SCDD_eval_all``, Apache-2.0) as
used by the SECOND literature. The T1 and T2 class maps are accumulated into ONE ``K x K``
confusion matrix (rows = prediction, columns = label, class 0 = no change), exactly like treating
the two time phases as two images. Counts are exact integers (int64); the metrics are computed in
float64 from the pooled matrix, so they are never batch-averaged. ``oa`` is the pooled pixel
accuracy (the reference averages per-image accuracies).
"""

import math

import numpy as np
import torch


def cohen_kappa(hist: np.ndarray) -> float:
    total = hist.sum()
    if total == 0:
        return 0.0
    po = np.diag(hist).sum() / total
    pe = float(np.matmul(hist.sum(1), hist.sum(0).T)) / total**2
    return 0.0 if pe == 1 else float((po - pe) / (1 - pe))


class SemanticChangeMetrics:
    """Pooled confusion-matrix accumulator for dense T1/T2 semantic maps (0 = no change)."""

    def __init__(self, num_classes: int = 7) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2 (no-change plus one class).")
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction: torch.Tensor, label: torch.Tensor) -> None:
        """Add integer maps of identical shape, e.g. [B,H,W] (call once per time phase)."""
        if prediction.shape != label.shape:
            raise ValueError(f"Shape mismatch: {tuple(prediction.shape)} vs {tuple(label.shape)}")
        for name, value in (("prediction", prediction), ("label", label)):
            if value.is_floating_point() or value.is_complex():
                raise ValueError(f"{name} must be an integer class map.")
            if value.numel() and (int(value.min()) < 0 or int(value.max()) >= self.num_classes):
                raise ValueError(f"{name} contains ids outside 0..{self.num_classes - 1}.")
        k = self.num_classes
        index = prediction.reshape(-1).long() * k + label.reshape(-1).long()
        counts = torch.bincount(index, minlength=k * k).reshape(k, k).cpu().numpy()
        self.hist += counts.astype(np.int64)

    def compute(self) -> dict[str, float]:
        hist = self.hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            raise ValueError("No pixels were accumulated.")
        hist_fg = hist[1:, 1:]
        c2 = np.zeros((2, 2))
        c2[0][0] = hist[0][0]
        c2[0][1] = hist.sum(1)[0] - hist[0][0]
        c2[1][0] = hist.sum(0)[0] - hist[0][0]
        c2[1][1] = hist_fg.sum()
        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cohen_kappa(hist_n0)
        iu = np.diag(c2) / (c2.sum(1) + c2.sum(0) - np.diag(c2) + 1e-10)
        change_pred = total - hist.sum(1)[0]
        change_label = total - hist.sum(0)[0]
        sc_tp = np.diag(hist_fg).sum()
        precision = sc_tp / (change_pred + 1e-10)
        recall = sc_tp / (change_label + 1e-10)
        both = precision > 0 and recall > 0
        fscd = 2 * precision * recall / (precision + recall) if both else 0.0
        return {
            "kappa_n0": kappa_n0,
            "sek": float(kappa_n0 * math.exp(iu[1]) / math.e),
            "miou": float((iu[0] + iu[1]) / 2),
            "iou_change": float(iu[1]),
            "iou_no_change": float(iu[0]),
            "fscd": float(fscd),
            "oa": float(np.diag(hist).sum() / total),
        }
