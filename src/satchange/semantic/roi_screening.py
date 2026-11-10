"""ROI generator family and ROI-coverage metrics for the SECOND detector screening (plan VS).

The generator is one declared family (threshold, margin, merge gap, minimum crop side) applied
identically to every detector; boxes of nearby detections are merged so that many tiny crops do not
arise, and a merge gap of ``None`` merges everything into one box. Metrics are computed per tile
as additive count vectors, so pooled metrics and tile bootstraps are sums over tiles
(``docs/research2/DETECTOR_SCREENING_PLAN.md`` §3). ``semantic/roi.py`` is not touched.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]

THRESHOLDS = (0.2, 0.35, 0.5, 0.65, 0.8)
MARGINS = (0, 16, 32)
MERGE_GAPS: tuple[int | None, ...] = (0, 16, 32, 64, 128, None)
MIN_CROPS = (64, 128)
SMALL_AREA = 1024  # components with fewer pixels are "small" (< 32 x 32)
LARGE_AREA = 8192
SINGLE_CROP_COVERAGE = 0.95
STRATA = ("small", "medium", "large", "all")
COMPONENT_COUNTS = ("n", "single95", "union95", "missed", "fragmented")
FIELDS = (
    "gt_pixels",
    "covered_pixels",
    "crops",
    "crop_area",
    "union_area",
    "false_crops",
    "false_area",
    *(f"{stratum}_{count}" for stratum in STRATA for count in COMPONENT_COUNTS),
)
FIELD_INDEX = {name: index for index, name in enumerate(FIELDS)}
BUDGETS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class ScreeningSettings:
    """One member of the common ROI generator family (all distances in native pixels)."""

    threshold: float
    margin: int
    merge_gap: int | None
    min_crop: int

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1].")
        if self.margin < 0 or self.min_crop < 1:
            raise ValueError("margin must be >= 0 and min_crop >= 1.")
        if self.merge_gap is not None and self.merge_gap < 0:
            raise ValueError("merge_gap must be >= 0 or None (one box).")


def settings_grid() -> list[ScreeningSettings]:
    """The 180 declared settings in a fixed order."""
    return [
        ScreeningSettings(t, m, g, c)
        for t, m, g, c in itertools.product(THRESHOLDS, MARGINS, MERGE_GAPS, MIN_CROPS)
    ]


def detections_by_threshold(
    probabilities: np.ndarray, thresholds: Sequence[float] = THRESHOLDS
) -> dict[float, np.ndarray]:
    """8-connected component boxes ``[n,4]`` (x0,y0,x1,y1; upper bounds exclusive) per threshold."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [H, W].")
    result = {}
    for threshold in thresholds:
        binary = (probabilities >= threshold).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        x, y, w, h = (stats[1:, i].astype(np.int64) for i in (0, 1, 2, 3))
        result[threshold] = np.stack([x, y, x + w, y + h], axis=1).reshape(-1, 4)
    return result


def merge_boxes(boxes: np.ndarray, gap: int | None) -> np.ndarray:
    """Repeatedly merge boxes whose axis gaps are both <= ``gap`` (overlap = negative gap)."""
    if len(boxes) <= 1:
        return boxes
    if gap is None:
        return np.array(
            [[boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()]],
            dtype=np.int64,
        )
    while True:
        gap_x = np.maximum(
            boxes[:, None, 0] - boxes[None, :, 2], boxes[None, :, 0] - boxes[:, None, 2]
        )
        gap_y = np.maximum(
            boxes[:, None, 1] - boxes[None, :, 3], boxes[None, :, 1] - boxes[:, None, 3]
        )
        count, labels = connected_components((gap_x <= gap) & (gap_y <= gap), directed=False)
        if count == len(boxes):
            return boxes
        merged = np.empty((count, 4), dtype=np.int64)
        merged[:, :2] = np.iinfo(np.int64).max
        merged[:, 2:] = np.iinfo(np.int64).min
        np.minimum.at(merged[:, 0], labels, boxes[:, 0])
        np.minimum.at(merged[:, 1], labels, boxes[:, 1])
        np.maximum.at(merged[:, 2], labels, boxes[:, 2])
        np.maximum.at(merged[:, 3], labels, boxes[:, 3])
        boxes = merged


def _grow_axis(
    low: np.ndarray, high: np.ndarray, extent: int, minimum: int
) -> tuple[np.ndarray, np.ndarray]:
    length = np.minimum(extent, np.maximum(high - low, minimum))
    start = np.maximum(0, np.minimum((low + high - length) // 2, extent - length))
    return start, start + length


def generate_boxes(
    detections: dict[float, np.ndarray], settings: ScreeningSettings, height: int, width: int
) -> np.ndarray:
    """Crop boxes ``[n,4]`` of one tile for one setting (margin, merge, minimum side, clip)."""
    boxes = detections[settings.threshold]
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    boxes = boxes + np.array([-settings.margin, -settings.margin, settings.margin, settings.margin])
    boxes = np.stack(
        [
            np.maximum(boxes[:, 0], 0),
            np.maximum(boxes[:, 1], 0),
            np.minimum(boxes[:, 2], width),
            np.minimum(boxes[:, 3], height),
        ],
        axis=1,
    )
    boxes = merge_boxes(boxes, settings.merge_gap)
    x0, x1 = _grow_axis(boxes[:, 0], boxes[:, 2], width, settings.min_crop)
    y0, y1 = _grow_axis(boxes[:, 1], boxes[:, 3], height, settings.min_crop)
    return np.stack([x0, y0, x1, y1], axis=1)


@dataclass
class SceneGT:
    """Ground-truth changed mask of one tile with its 8-connected components."""

    mask: np.ndarray
    labels: np.ndarray
    areas: np.ndarray
    boxes: np.ndarray
    strata: np.ndarray
    integral: np.ndarray


def build_scene_gt(mask: np.ndarray) -> SceneGT:
    """Components, sizes (0 small / 1 medium / 2 large) and an integral image of a bool mask."""
    mask = mask.astype(bool)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    boxes = np.stack(
        [
            stats[1:, cv2.CC_STAT_LEFT],
            stats[1:, cv2.CC_STAT_TOP],
            stats[1:, cv2.CC_STAT_LEFT] + stats[1:, cv2.CC_STAT_WIDTH],
            stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT],
        ],
        axis=1,
    ).astype(np.int64)
    strata = np.where(areas < SMALL_AREA, 0, np.where(areas < LARGE_AREA, 1, 2))
    integral = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    return SceneGT(mask, labels, areas, boxes.reshape(-1, 4), strata, integral)


def scene_counts(gt: SceneGT, boxes: np.ndarray) -> np.ndarray:
    """Additive count vector (``FIELDS``) of one tile and one box set."""
    height, width = gt.mask.shape
    out = np.zeros(len(FIELDS), dtype=np.float64)
    union = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        union[y0:y1, x0:x1] = True
    out[FIELD_INDEX["gt_pixels"]] = gt.mask.sum()
    out[FIELD_INDEX["covered_pixels"]] = (union & gt.mask).sum()
    out[FIELD_INDEX["union_area"]] = union.sum()
    out[FIELD_INDEX["crops"]] = len(boxes)
    if len(boxes):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ii = gt.integral
        inside = (
            ii[boxes[:, 3], boxes[:, 2]]
            - ii[boxes[:, 1], boxes[:, 2]]
            - ii[boxes[:, 3], boxes[:, 0]]
            + ii[boxes[:, 1], boxes[:, 0]]
        )
        out[FIELD_INDEX["crop_area"]] = areas.sum()
        out[FIELD_INDEX["false_crops"]] = (inside == 0).sum()
        out[FIELD_INDEX["false_area"]] = areas[inside == 0].sum()
    components = len(gt.areas)
    if components == 0:
        return out
    in_union = np.bincount(gt.labels[union], minlength=components + 1)[1:]
    union_fraction = in_union / gt.areas
    single_fraction = np.zeros(components)
    if len(boxes):
        gx0, gy0, gx1, gy1 = (gt.boxes[:, i, None] for i in range(4))
        cx0, cy0, cx1, cy1 = (boxes[None, :, i] for i in range(4))
        contains = (cx0 <= gx0) & (cy0 <= gy0) & (cx1 >= gx1) & (cy1 >= gy1)
        overlaps = (np.minimum(cx1, gx1) > np.maximum(cx0, gx0)) & (
            np.minimum(cy1, gy1) > np.maximum(cy0, gy0)
        )
        single_fraction[contains.any(axis=1)] = 1.0
        for component in np.flatnonzero(~contains.any(axis=1) & overlaps.any(axis=1)):
            best = 0
            for index in np.flatnonzero(overlaps[component]):
                x0, y0, x1, y1 = boxes[index]
                best = max(best, int((gt.labels[y0:y1, x0:x1] == component + 1).sum()))
            single_fraction[component] = best / gt.areas[component]
    single = single_fraction >= SINGLE_CROP_COVERAGE
    covered = union_fraction >= SINGLE_CROP_COVERAGE
    flags = {
        "n": np.ones(components, dtype=bool),
        "single95": single,
        "union95": covered,
        "missed": in_union == 0,
        "fragmented": covered & ~single,
    }
    for stratum_index, stratum in enumerate(STRATA):
        selected = (
            np.ones(components, dtype=bool) if stratum == "all" else gt.strata == stratum_index
        )
        for name in COMPONENT_COUNTS:
            out[FIELD_INDEX[f"{stratum}_{name}"]] = (flags[name] & selected).sum()
    return out


def scene_table(
    gt: SceneGT, probabilities: np.ndarray, settings: Sequence[ScreeningSettings]
) -> np.ndarray:
    """``[len(settings), len(FIELDS)]`` counts of one tile for the whole generator grid."""
    height, width = probabilities.shape
    detections = detections_by_threshold(probabilities, sorted({s.threshold for s in settings}))
    return np.stack(
        [scene_counts(gt, generate_boxes(detections, s, height, width)) for s in settings]
    )


def pooled_metrics(
    sums: np.ndarray, tile_weight: np.ndarray | float, tile_pixels: int
) -> dict[str, np.ndarray]:
    """Pooled metrics from summed count vectors ``[..., len(FIELDS)]`` of ``tile_weight`` tiles."""

    def field(name: str) -> np.ndarray:
        return np.asarray(sums[..., FIELD_INDEX[name]])

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.divide(
                numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0
            )
        )

    result = {
        "pixel_recall": ratio(field("covered_pixels"), field("gt_pixels")),
        "crops_per_tile": field("crops") / tile_weight,
        "area_ratio": field("crop_area") / (tile_weight * tile_pixels),
        "unique_area_fraction": field("union_area") / (tile_weight * tile_pixels),
        "false_crop_fraction": ratio(field("false_crops"), field("crops")),
        "false_area_fraction": ratio(field("false_area"), field("crop_area")),
    }
    for stratum in STRATA:
        total = field(f"{stratum}_n")
        for name in COMPONENT_COUNTS[1:]:
            result[f"{stratum}_{name}"] = ratio(field(f"{stratum}_{name}"), total)
    return result


def bootstrap_sums(table: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    """Tile-bootstrap sums ``[B, S, F]`` of ``table`` ``[S, tiles, F]`` (S settings, F fields)."""
    settings, tiles, fields = table.shape
    counts = np.random.default_rng(seed).multinomial(
        tiles, np.full(tiles, 1.0 / tiles), size=resamples
    )
    flat = table.transpose(1, 0, 2).reshape(tiles, settings * fields)
    return np.asarray(counts @ flat).reshape(resamples, settings, fields)


def best_under_budget(
    values: np.ndarray, area: np.ndarray, budgets: Sequence[float] = BUDGETS
) -> np.ndarray:
    """Frontier: best ``values[..., S]`` among settings with processed area ratio <= budget."""
    budget = np.asarray(budgets)[:, None]
    masked = np.where(
        area[..., None, :] <= budget, np.nan_to_num(values[..., None, :], nan=0.0), 0.0
    )
    return np.asarray(masked.max(axis=-1))


def frontier_mean_recall(
    recall: np.ndarray, area: np.ndarray, budgets: Sequence[float] = BUDGETS
) -> np.ndarray:
    """FMR: mean over the budget grid of the best ROI pixel recall within the budget."""
    return np.asarray(best_under_budget(recall, area, budgets).mean(axis=-1))


def min_cost_at_recall(recall: np.ndarray, cost: np.ndarray, target: float) -> np.ndarray:
    """Smallest cost among settings reaching ``recall >= target`` (``inf`` if none)."""
    return np.asarray(np.where(recall >= target, cost, np.inf).min(axis=-1))


# --- detector diagnostics (plan metric M8) -------------------------------------------------

HISTOGRAM_BINS = 1000


def update_histograms(
    positive: np.ndarray, negative: np.ndarray, probabilities: np.ndarray, mask: np.ndarray
) -> None:
    """Accumulate probability histograms of changed / unchanged pixels (for AP and F1 curves)."""
    bins = np.minimum((probabilities * HISTOGRAM_BINS).astype(np.int64), HISTOGRAM_BINS - 1)
    positive += np.bincount(bins[mask], minlength=HISTOGRAM_BINS)
    negative += np.bincount(bins[~mask], minlength=HISTOGRAM_BINS)


def confusion_at_threshold(
    positive: np.ndarray, negative: np.ndarray, threshold: float
) -> dict[str, float]:
    """Pixel precision/recall/F1/IoU for ``probability >= threshold`` (bin resolution 1e-3)."""
    cut = int(round(threshold * HISTOGRAM_BINS))
    tp, fp = positive[cut:].sum(), negative[cut:].sum()
    fn = positive[:cut].sum()
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "iou": float(tp / max(tp + fp + fn, 1)),
    }


def average_precision(positive: np.ndarray, negative: np.ndarray) -> float:
    """Average precision from the two histograms (threshold-free, bin resolution 1e-3)."""
    tp = np.cumsum(positive[::-1])
    fp = np.cumsum(negative[::-1])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(positive.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def strict_component_matches(gt: SceneGT, prediction: np.ndarray) -> tuple[int, int]:
    """(matched GT components, GT components) at IoU >= 0.5 (unique by construction)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        prediction.astype(np.uint8), connectivity=8
    )
    components = len(gt.areas)
    if components == 0 or count == 1:
        return 0, components
    both = (gt.labels > 0) & (labels > 0)
    pair = gt.labels[both].astype(np.int64) * count + labels[both]
    overlaps = np.bincount(pair, minlength=(components + 1) * count).reshape(components + 1, count)
    predicted_areas = stats[:, cv2.CC_STAT_AREA]
    intersection = overlaps[1:, 1:]
    union = gt.areas[:, None] + predicted_areas[None, 1:] - intersection
    return int((intersection / np.maximum(union, 1) >= 0.5).any(axis=1).sum()), components
