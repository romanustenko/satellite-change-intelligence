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
) -> dict[str, float | int | str]:
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


def quantile_edges(density: np.ndarray, strata: int) -> list[float]:
    """Interior stratum edges (``strata - 1`` values) at equal-count quantiles of ``density``.

    Compute on the *validation* split and freeze before the test pass (protocol §8).
    """
    if strata < 2:
        raise ValueError("At least two strata are required.")
    levels = [100 * k / strata for k in range(1, strata)]
    edges = [float(v) for v in np.percentile(density, levels)]
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError("Quantile edges are not strictly increasing; use fewer strata.")
    return edges


def assign_strata(density: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Stratum index per scene: 0 for ``rho <= edges[0]`` ... ``len(edges)`` above the last edge."""
    return np.searchsorted(np.asarray(edges), density, side="left")


def _seed_mean_metric(hists: np.ndarray, chosen: np.ndarray, metric: str) -> float:
    return float(np.mean([metrics_from_histogram(h[chosen].sum(0))[metric] for h in hists]))


def paired_bootstrap_seeds(
    candidate: np.ndarray, reference: np.ndarray, *, metric: str = "sek", resamples: int = 2000,
    seed: int = 0, confidence: float = 0.95, subset: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Paired scene bootstrap of ``mean_over_seeds[metric(candidate) - metric(reference)]``.

    Histograms are ``[seeds,scenes,K,K]`` with the same scene order; each resample draws one
    scene index set shared by all seeds and both systems (scene-level pairing nested with
    training seeds). ``subset`` (scene indices) restricts to one density stratum.
    """
    if candidate.shape != reference.shape or candidate.ndim != 4:
        raise ValueError("Histograms must share the shape [seeds,scenes,K,K].")
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be in (0,1) and resamples >= 1.")
    scenes = np.arange(candidate.shape[1]) if subset is None else np.asarray(subset)
    if scenes.size == 0:
        raise ValueError("The scene subset is empty.")
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for index in range(resamples):
        chosen = scenes[rng.integers(0, scenes.size, scenes.size)]
        differences[index] = (_seed_mean_metric(candidate, chosen, metric)
                              - _seed_mean_metric(reference, chosen, metric))
    tail = (1 - confidence) / 2 * 100
    observed = (_seed_mean_metric(candidate, scenes, metric)
                - _seed_mean_metric(reference, scenes, metric))
    return {
        "difference": float(observed), "ci_low": float(np.percentile(differences, tail)),
        "ci_high": float(np.percentile(differences, 100 - tail)), "scenes": int(scenes.size),
        "seeds": int(candidate.shape[0]), "resamples": resamples, "seed": seed,
        "metric": metric, "confidence": confidence,
    }


def non_inferior(result: dict[str, float | int | str], margin: float) -> bool:
    """Non-inferiority of the candidate: lower CI bound of (candidate - reference) > -margin."""
    if margin <= 0:
        raise ValueError("The non-inferiority margin must be positive.")
    return bool(float(result["ci_low"]) > -margin)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="stable")
    ranks = np.empty(values.size)
    ranks[order] = np.arange(values.size)
    for value in np.unique(values):  # average ranks of ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _ranks(np.asarray(x, dtype=float)), _ranks(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def latency_ratio_by_stratum(
    latency_b: np.ndarray, latency_a: np.ndarray, density: np.ndarray, edges: Sequence[float],
    *, resamples: int = 2000, seed: int = 0, confidence: float = 0.95,
) -> dict[str, object]:
    """Per-stratum ratio of mean latencies B/A with a bootstrap CI, plus the density trend.

    ``trend`` is the Spearman correlation between scene density and the per-scene difference
    ``latency_b - latency_a`` with a scene-bootstrap CI (H-B: positive trend, ratio above 1 at
    high density). Latencies are per-scene medians (ms) from the same scene on the same GPU.
    """
    latency_a, latency_b, density = (np.asarray(v, dtype=float) for v in (latency_a, latency_b,
                                                                           density))
    if not latency_a.shape == latency_b.shape == density.shape:
        raise ValueError("latency_a, latency_b and density must have the same shape.")
    strata = assign_strata(density, edges)
    rng = np.random.default_rng(seed)
    tail = (1 - confidence) / 2 * 100
    rows = []
    for stratum in range(len(edges) + 1):
        members = np.flatnonzero(strata == stratum)
        if members.size == 0:
            raise ValueError(f"Stratum {stratum} has no scenes.")
        ratios = np.empty(resamples)
        for index in range(resamples):
            chosen = members[rng.integers(0, members.size, members.size)]
            ratios[index] = latency_b[chosen].mean() / latency_a[chosen].mean()
        rows.append({
            "stratum": stratum, "scenes": int(members.size),
            "ratio": float(latency_b[members].mean() / latency_a[members].mean()),
            "ci_low": float(np.percentile(ratios, tail)),
            "ci_high": float(np.percentile(ratios, 100 - tail)),
            "mean_ms_a": float(latency_a[members].mean()),
            "mean_ms_b": float(latency_b[members].mean()),
        })
    gap = latency_b - latency_a
    boot = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.integers(0, density.size, density.size)
        boot[index] = spearman(density[chosen], gap[chosen])
    return {
        "strata": rows, "edges": [float(e) for e in edges],
        "trend": {
            "spearman": spearman(density, gap), "ci_low": float(np.percentile(boot, tail)),
            "ci_high": float(np.percentile(boot, 100 - tail)),
        },
    }


def error_decomposition(inside: np.ndarray, outside: np.ndarray) -> dict[str, float]:
    """H-C decomposition of system B errors from pooled inside/outside-ROI histograms.

    Counts are over both time phases. ``missed_*`` = changed label pixels (class > 0) predicted as
    something else; outside the ROI union every prediction is class 0, so all changed pixels there
    are misses (stage-1 failures), while inside they are WHAT errors (wrong class or no change).
    ``false_change_*`` = unchanged label pixels predicted as a change class.
    """
    def parts(hist: np.ndarray) -> tuple[int, int, int]:
        changed = int(hist[:, 1:].sum())
        correct = int(np.trace(hist[1:, 1:]))
        return changed, changed - correct, int(hist[1:, 0].sum())

    changed_in, missed_in, false_in = parts(inside)
    changed_out, missed_out, false_out = parts(outside)
    total_missed = missed_in + missed_out
    return {
        "changed_pixels": float(changed_in + changed_out),
        "missed_inside_roi": float(missed_in), "missed_outside_roi": float(missed_out),
        "false_change_inside_roi": float(false_in), "false_change_outside_roi": float(false_out),
        "outside_share_of_missed": missed_out / total_missed if total_missed else 0.0,
    }
