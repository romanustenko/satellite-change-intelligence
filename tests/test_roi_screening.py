"""CPU tests of the detector-screening ROI generator and coverage metrics (plan VS)."""

import numpy as np
import pytest

from satchange.semantic import roi_screening as rs


def _mask(boxes: list[tuple[int, int, int, int]], size: int = 128) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = True
    return mask


def test_grid_has_declared_size_and_validates() -> None:
    assert len(rs.settings_grid()) == 9 * 3 * 6 * 2
    with pytest.raises(ValueError):
        rs.ScreeningSettings(0.0, 0, 0, 64)
    with pytest.raises(ValueError):
        rs.ScreeningSettings(0.5, -1, 0, 64)


def test_merge_gap_controls_number_of_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [14, 0, 24, 10], [100, 100, 110, 110]])
    assert len(rs.merge_boxes(boxes, 0)) == 3
    assert len(rs.merge_boxes(boxes, 4)) == 2  # gap of exactly 4 merges
    one = rs.merge_boxes(boxes, None)
    assert one.tolist() == [[0, 0, 110, 110]]
    chained = np.array([[0, 0, 10, 10], [12, 0, 22, 10], [24, 0, 34, 10]])
    assert rs.merge_boxes(chained, 2).tolist() == [[0, 0, 34, 10]]  # transitive merging


def test_generate_boxes_margin_min_crop_and_clipping() -> None:
    probabilities = np.zeros((128, 128))
    probabilities[60:64, 60:64] = 0.9
    detections = rs.detections_by_threshold(probabilities)
    box = rs.generate_boxes(detections, rs.ScreeningSettings(0.5, 0, 0, 32), 128, 128)
    assert box.tolist() == [[46, 46, 78, 78]]  # 4x4 grown around its centre to 32
    corner = np.zeros((128, 128))
    corner[0:2, 0:2] = 0.9
    box = rs.generate_boxes(
        rs.detections_by_threshold(corner), rs.ScreeningSettings(0.5, 8, 0, 32), 128, 128
    )
    assert box.tolist() == [[0, 0, 32, 32]]  # clipped to the tile, still 32 wide
    empty = rs.generate_boxes(
        rs.detections_by_threshold(np.zeros((128, 128))),
        rs.ScreeningSettings(0.5, 0, 0, 32),
        128,
        128,
    )
    assert empty.shape == (0, 4)


def test_scene_counts_perfect_partial_and_missed() -> None:
    gt = rs.build_scene_gt(_mask([(10, 10, 30, 30), (80, 80, 90, 90)]))  # 400 px and 100 px
    perfect = rs.scene_counts(gt, np.array([[10, 10, 30, 30], [80, 80, 90, 90]]))
    assert perfect[rs.FIELD_INDEX["covered_pixels"]] == 500
    assert (
        perfect[rs.FIELD_INDEX["all_single95"]] == 2 and perfect[rs.FIELD_INDEX["all_missed"]] == 0
    )
    assert perfect[rs.FIELD_INDEX["false_crops"]] == 0
    # first component half covered by one crop, second not at all
    partial = rs.scene_counts(gt, np.array([[10, 10, 20, 30]]))
    assert partial[rs.FIELD_INDEX["covered_pixels"]] == 200
    assert partial[rs.FIELD_INDEX["all_single95"]] == 0
    assert partial[rs.FIELD_INDEX["all_missed"]] == 1
    assert partial[rs.FIELD_INDEX["small_n"]] == 2  # both components are < 1024 px
    # two crops that together cover component 1 (union 95 % but no single crop) -> fragmented
    fragmented = rs.scene_counts(gt, np.array([[10, 10, 20, 30], [20, 10, 30, 30]]))
    assert fragmented[rs.FIELD_INDEX["all_union95"]] == 1
    assert fragmented[rs.FIELD_INDEX["all_single95"]] == 0
    assert fragmented[rs.FIELD_INDEX["all_fragmented"]] == 1
    # a crop with no change inside is a false crop
    false = rs.scene_counts(gt, np.array([[40, 40, 60, 60]]))
    assert false[rs.FIELD_INDEX["false_crops"]] == 1
    assert false[rs.FIELD_INDEX["false_area"]] == 400


def test_pooled_metrics_and_strata() -> None:
    big = rs.build_scene_gt(_mask([(0, 0, 100, 100)]))  # 10 000 px -> large
    sums = rs.scene_counts(big, np.array([[0, 0, 100, 100]]))
    metrics = rs.pooled_metrics(sums, 1.0, 128 * 128)
    assert metrics["pixel_recall"] == 1.0
    assert metrics["large_single95"] == 1.0 and np.isnan(metrics["small_single95"])
    assert metrics["area_ratio"] == pytest.approx(100 * 100 / (128 * 128))


def test_bootstrap_and_frontier() -> None:
    rng = np.random.default_rng(0)
    table = rng.integers(0, 5, size=(3, 20, len(rs.FIELDS))).astype(float)
    sums = rs.bootstrap_sums(table, 50, seed=1)
    assert sums.shape == (50, 3, len(rs.FIELDS))
    assert np.allclose(sums.sum(axis=0) / 50, table.sum(axis=1), rtol=0.5)  # same scale
    assert np.array_equal(sums, rs.bootstrap_sums(table, 50, seed=1))  # deterministic
    recall = np.array([0.5, 0.9, 0.95])
    area = np.array([0.1, 0.4, 0.9])
    front = rs.best_under_budget(recall, area, (0.2, 0.5, 1.0))
    assert front.tolist() == [0.5, 0.9, 0.95]
    assert rs.frontier_mean_recall(recall, area, (0.2, 0.5, 1.0)) == pytest.approx(
        (0.5 + 0.9 + 0.95) / 3
    )
    assert rs.min_cost_at_recall(recall, area, 0.9) == pytest.approx(0.4)
    assert np.isinf(rs.min_cost_at_recall(recall, area, 0.99))


def test_scene_table_one_box_extreme_covers_everything() -> None:
    mask = _mask([(5, 5, 15, 15), (100, 100, 120, 120)])
    probabilities = mask.astype(float) * 0.9
    gt = rs.build_scene_gt(mask)
    settings = [rs.ScreeningSettings(0.5, 0, None, 64), rs.ScreeningSettings(0.5, 0, 0, 64)]
    table = rs.scene_table(gt, probabilities, settings)
    assert (
        table[0, rs.FIELD_INDEX["crops"]] == 1
        and table[0, rs.FIELD_INDEX["covered_pixels"]] == mask.sum()
    )
    assert table[1, rs.FIELD_INDEX["crops"]] == 2


def test_detector_diagnostics() -> None:
    mask = _mask([(10, 10, 30, 30), (80, 80, 90, 90)])
    probabilities = np.where(mask, 0.9, 0.05)
    positive = np.zeros(rs.HISTOGRAM_BINS)
    negative = np.zeros(rs.HISTOGRAM_BINS)
    rs.update_histograms(positive, negative, probabilities, mask)
    assert rs.confusion_at_threshold(positive, negative, 0.5)["f1"] == 1.0
    assert rs.average_precision(positive, negative) == pytest.approx(1.0)
    gt = rs.build_scene_gt(mask)
    assert rs.strict_component_matches(gt, mask) == (2, 2)
    assert rs.strict_component_matches(gt, _mask([(10, 10, 30, 30)])) == (1, 2)
    assert rs.strict_component_matches(gt, np.zeros_like(mask)) == (0, 2)
    # a prediction covering one component with IoU < 0.5 does not match
    assert rs.strict_component_matches(gt, _mask([(10, 10, 30, 30), (60, 60, 120, 120)]))[0] == 1
