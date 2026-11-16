"""CPU tests of the pure logic in the V9 scripts (OR fusion, window buckets, queue rules, Holm)."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_r2_v2_systems as analysis  # noqa: E402
import r2_v2_queue as queue  # noqa: E402
import r2_v2_systems_lib as lib  # noqa: E402


def test_or_map_is_pixelwise_max_and_equals_binary_or_at_any_threshold() -> None:
    first = np.array([[0.1, 0.9], [0.4, 0.2]], dtype=np.float32)
    second = np.array([[0.3, 0.1], [0.6, 0.2]], dtype=np.float32)
    fused = lib.or_map(first, second)
    assert np.array_equal(fused, np.array([[0.3, 0.9], [0.6, 0.2]], dtype=np.float32))
    for threshold in (0.25, 0.5, 0.7):
        assert np.array_equal(fused >= threshold, (first >= threshold) | (second >= threshold))
    with pytest.raises(ValueError):
        lib.or_map(first, second[:1])


def test_window_bucket_and_processed_area() -> None:
    assert lib.window_bucket([0, 0, 100, 40]) == 128
    assert lib.window_bucket([0, 0, 129, 10]) == 256
    assert lib.window_bucket([10, 10, 400, 20]) == 512
    boxes = np.array([[0, 0, 64, 64], [0, 0, 200, 100]])
    assert lib.processed_window_area(boxes) == 128 ** 2 + 256 ** 2
    assert lib.processed_window_area(np.zeros((0, 4), dtype=int)) == 0


def test_holm_adjustment_is_monotone_and_capped() -> None:
    adjusted = analysis.holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert adjusted["a"] == pytest.approx(0.003)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.04)
    assert max(analysis.holm({"x": 0.6, "y": 0.9}).values()) <= 1.0


def test_queue_hpo_trials_match_the_search_runner_and_final_jobs_wait_for_all_trials() -> None:
    jobs = queue.hpo_jobs("scannet")
    assert len(jobs) == queue.HPO_TRIALS
    assert [j.name for j in jobs] == [
        queue.hpo_name("scannet", i, t) for i, t in enumerate(queue.hpo.draw_trials(6))
    ]
    assert all(j.budget == queue.HPO_BUDGET and j.seed == 42 for j in jobs)
    # no run_record.json exists for a model name that never ran -> no finals are generated
    assert queue.winner.__name__ == "winner"


def test_queue_place_respects_memory_rules() -> None:
    gpus = {
        0: {"used": 30.0, "own": 25.0, "foreign": 5.0},
        1: {"used": 20.0, "own": 0.0, "foreign": 20.0},  # busy foreign GPU: own cap 40 GB
        2: {"used": 0.0, "own": 0.0, "foreign": 0.0},
    }
    reserved = dict.fromkeys(gpus, 0.0)
    mamba = queue.Job("m", "mamba_scd", "hpo", 1e-4, 0.01, 42, 8e16, "hpo")
    unet = queue.Job("u", "siam_unet_resnet18", "hpo", 1e-4, 0.01, 42, 8e16, "hpo")
    assert queue.place(mamba, {2: gpus[2]}, reserved) == 2
    assert queue.place(mamba, {0: gpus[0]}, reserved) is None  # 25 + 56 + 5 > 76
    assert queue.place(mamba, {1: gpus[1]}, reserved) is None  # busy GPU: own > 40 GB
    assert queue.place(unet, {1: gpus[1]}, reserved) == 1
