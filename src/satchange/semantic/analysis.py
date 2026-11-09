"""Scene-level statistics for Research 2 comparisons (R2-P12-T01).

SeK/mIoU/Fscd are pooled-confusion metrics, so a bootstrap must resample *scenes* and recompute
the metric from summed per-scene histograms (never average per-scene scores). Everything is
seeded and deterministic.
"""

from collections.abc import Sequence

import numpy as np
import torch

from satchange.semantic.scd_metrics import SemanticChangeMetrics

NUM_CLASSES = 7


def scene_histogram(
    prediction_1: torch.Tensor, prediction_2: torch.Tensor,
    label_1: torch.Tensor, label_2: torch.Tensor, num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Pooled T1+T2 confusion matrix of one scene (rows = prediction, columns = label)."""
    metrics = SemanticChangeMetrics(num_classes)
    metrics.update(prediction_1, label_1)
    metrics.update(prediction_2, label_2)
    return metrics.hist.copy()


def metrics_from_histogram(hist: np.ndarray) -> dict[str, float]:
    """SeK/mIoU/Fscd/... from a pooled histogram."""
    metrics = SemanticChangeMetrics(hist.shape[0])
    metrics.hist = hist.astype(np.int64)
    return metrics.compute()


def paired_bootstrap(
    histograms_a: np.ndarray, histograms_b: np.ndarray, *, metric: str = "sek",
    resamples: int = 2000, seed: int = 0, confidence: float = 0.95,
) -> dict[str, float]:
    """Paired scene bootstrap of ``metric(A) - metric(B)``; histograms are ``[scenes,K,K]``."""
    if histograms_a.shape != histograms_b.shape or histograms_a.ndim != 3:
        raise ValueError("Paired histograms must share the shape [scenes,K,K].")
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be in (0,1) and resamples >= 1.")
    scenes = histograms_a.shape[0]
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.integers(0, scenes, scenes)
        differences[index] = (
            metrics_from_histogram(histograms_a[chosen].sum(0))[metric]
            - metrics_from_histogram(histograms_b[chosen].sum(0))[metric]
        )
    tail = (1 - confidence) / 2 * 100
    observed = (
        metrics_from_histogram(histograms_a.sum(0))[metric]
        - metrics_from_histogram(histograms_b.sum(0))[metric]
    )
    return {
        "difference": float(observed),
        "ci_low": float(np.percentile(differences, tail)),
        "ci_high": float(np.percentile(differences, 100 - tail)),
        "resamples": resamples, "seed": seed, "metric": metric, "confidence": confidence,
    }


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points for (cost to minimize, quality to maximize)."""
    front = []
    for i, (cost_i, quality_i) in enumerate(points):
        dominated = any(
            cost_j <= cost_i and quality_j >= quality_i
            and (cost_j < cost_i or quality_j > quality_i)
            for j, (cost_j, quality_j) in enumerate(points) if j != i
        )
        if not dominated:
            front.append(i)
    return front
