"""ROI policy selection on validation and ROI-coverage diagnostics (R2-P11-T04).

Selection never sees the test split: the caller passes a validation ``evaluate`` function. Ties
go to the earlier grid entry (thresholds ascending, then context, then minimum area).
"""

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import torch

from satchange.semantic.roi import DEFAULT_ROI_SETTINGS, ROISettings, extract_rois


def select_roi_policy(
    evaluate: Callable[[ROISettings], dict[str, float]],
    *,
    thresholds: Sequence[float],
    context_pixels: Sequence[int],
    min_component_areas: Sequence[int],
    key: str = "sek",
    base: ROISettings = DEFAULT_ROI_SETTINGS,
) -> dict[str, Any]:
    """Grid-search ROI settings by the validation ``key`` metric; returns best and every row."""
    rows = []
    for threshold, context, area in itertools.product(
        thresholds, context_pixels, min_component_areas
    ):
        settings = ROISettings(**{
            **base.__dict__, "threshold": threshold, "context_pixels": context,
            "min_component_area": area,
        })
        metrics = evaluate(settings)
        rows.append({"settings": dict(settings.__dict__), "metrics": metrics})
    if not rows:
        raise ValueError("The ROI policy grid is empty.")
    best = max(range(len(rows)), key=lambda i: (rows[i]["metrics"][key], -i))
    return {"selection_key": key, "best_index": best, "best": rows[best], "rows": rows}


def coverage_diagnostics(
    probabilities: torch.Tensor, change_mask: torch.Tensor,
    settings: ROISettings = DEFAULT_ROI_SETTINGS,
) -> dict[str, float]:
    """Stage-1 failure diagnostics for one scene: change pixels outside ROI crops, crop count."""
    scene = torch.zeros(3, *probabilities.shape[-2:])
    rois = extract_rois(scene, scene, probabilities, settings)
    covered = torch.zeros(probabilities.shape[-2:], dtype=torch.bool)
    for roi in rois:
        left, top, right, bottom = roi.crop_bbox
        covered[top:bottom, left:right] = True
    change = change_mask[0] > 0.5
    total = int(change.sum())
    missed = int((change & ~covered).sum())
    return {
        "crops": float(len(rois)), "change_pixels": float(total),
        "missed_change_pixels": float(missed),
        "missed_fraction": missed / total if total else 0.0,
        "roi_union_fraction": float(covered.float().mean()),
        "false_crops": float(sum(
            1 for roi in rois
            if not bool((change_mask[:, roi.crop_bbox[1]:roi.crop_bbox[3],
                                     roi.crop_bbox[0]:roi.crop_bbox[2]] > 0.5).any())
        )),
    }
