"""Scene bootstrap and Pareto helpers (CPU, synthetic)."""

import numpy as np
import pytest
import torch

from satchange.semantic.analysis import (
    metrics_from_histogram,
    paired_bootstrap,
    pareto_front,
    scene_histogram,
)


def _scenes(accuracy: float, count: int = 12, seed: int = 0) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        label = torch.randint(0, 7, (32, 32), generator=generator)
        noise = torch.randint(0, 7, (32, 32), generator=generator)
        keep = torch.rand(32, 32, generator=generator) < accuracy
        prediction = torch.where(keep, label, noise)
        out.append(scene_histogram(prediction, prediction, label, label))
    return np.stack(out)


def test_scene_histograms_sum_to_pooled_metrics() -> None:
    scenes = _scenes(0.9)
    pooled = metrics_from_histogram(scenes.sum(0))
    assert pooled["oa"] > 0.8 and scenes.sum() == 12 * 2 * 32 * 32


def test_bootstrap_detects_better_system_and_is_seeded() -> None:
    better, worse = _scenes(0.95), _scenes(0.6, seed=1)
    first = paired_bootstrap(better, worse, resamples=200, seed=3)
    assert first["ci_low"] > 0 and first == paired_bootstrap(better, worse, resamples=200, seed=3)
    same = paired_bootstrap(better, better, resamples=100)
    assert same["difference"] == 0 and same["ci_low"] == same["ci_high"] == 0
    with pytest.raises(ValueError):
        paired_bootstrap(better, worse[:5])


def test_pareto_front() -> None:
    points = [(1.0, 0.5), (2.0, 0.9), (2.0, 0.4), (3.0, 0.9)]
    assert pareto_front(points) == [0, 1]
