"""Per-scene evaluation of systems A and B (R2-P11-T06 core, shared by validation and test passes).

Every function returns per-scene pooled T1+T2 confusion matrices (``[N,K,K]``), so the paired
scene bootstrap in ``analysis`` can recompute SeK/mIoU/Fscd on resamples. System B also returns
the matrices split into pixels inside and outside the ROI union (H-C error decomposition) and
per-scene stage-1 coverage diagnostics. Nothing here selects anything: the caller passes frozen
checkpoints and a frozen ROI policy.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from satchange.models.factory import SEMANTIC_TASK, create_task_model
from satchange.semantic.roi import DEFAULT_ROI_SETTINGS, ROISettings, extract_rois
from satchange.semantic.roi_policy import coverage_diagnostics
from satchange.semantic.scd_metrics import SemanticChangeMetrics
from satchange.semantic.scd_training import NUM_CLASSES, predict_maps
from satchange.semantic.what_stage import predict_roi_logits, reconstruct_scene


def load_semantic_checkpoint(path: Path, device: torch.device) -> nn.Module:
    """Rebuild a semantic model from a ``best.pt`` (best-epoch weights) on ``device``."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = create_task_model(
        state["model_name"], state["config"]["model_parameters"], SEMANTIC_TASK
    )
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval()


def _histogram(
    maps: tuple[torch.Tensor, torch.Tensor], labels: tuple[torch.Tensor, torch.Tensor],
    region: torch.Tensor | None = None,
) -> np.ndarray:
    """Pooled T1+T2 confusion matrix, optionally restricted to the pixels of ``region``."""
    metrics = SemanticChangeMetrics(NUM_CLASSES)
    for prediction, label in zip(maps, labels, strict=True):
        if region is None:
            metrics.update(prediction.cpu(), label)
        else:
            metrics.update(prediction.cpu()[region], label[region])
    return metrics.hist.copy()


def change_density(sample: dict[str, Any]) -> float:
    """Changed-pixel fraction rho of a scene (binary change mask)."""
    return float((sample["mask"] > 0.5).float().mean())


@torch.inference_mode()
def evaluate_system_a(
    model: nn.Module, dataset: Any, device: torch.device, *, limit: int = 0
) -> dict[str, Any]:
    """Whole-scene dense semantic pass for every scene of ``dataset``."""
    count = limit or len(dataset)
    histograms, density = [], []
    for index in range(count):
        sample = dataset[index]
        logits = model(sample["before"][None].to(device), sample["after"][None].to(device)).float()
        first, second = predict_maps(logits)
        labels = (sample["before_label"], sample["after_label"])
        histograms.append(_histogram((first[0], second[0]), labels))
        density.append(change_density(sample))
    return {
        "sample_ids": list(dataset.sample_ids[:count]), "histograms": np.stack(histograms),
        "density": np.array(density),
    }


@torch.inference_mode()
def predict_scene_b(
    what: nn.Module, sample: dict[str, Any], probabilities: torch.Tensor, device: torch.device,
    settings: ROISettings,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Detector probabilities -> ROI crops -> WHAT -> native class maps and the ROI union mask."""
    shape = tuple(sample["before"].shape[-2:])
    rois = extract_rois(sample["before"], sample["after"], probabilities, settings)
    union = torch.zeros(shape, dtype=torch.bool)
    for roi in rois:
        left, top, right, bottom = roi.crop_bbox
        union[top:bottom, left:right] = True
    if not rois:
        empty = torch.zeros(shape, dtype=torch.long)
        return (empty, empty.clone()), union
    logits = [predict_roi_logits(what, roi, device) for roi in rois]
    first, second = reconstruct_scene(rois, logits, shape)  # type: ignore[arg-type]
    return (first.cpu(), second.cpu()), union


@torch.inference_mode()
def evaluate_system_b(
    what: nn.Module, dataset: Any, device: torch.device,
    probability_source: Callable[[dict[str, Any]], torch.Tensor] | None,
    *, settings: ROISettings = DEFAULT_ROI_SETTINGS, limit: int = 0,
) -> dict[str, Any]:
    """System B for every scene; ``probability_source=None`` gives the oracle (GT-ROI) variant."""
    count = limit or len(dataset)
    total, inside, outside, density, coverage = [], [], [], [], []
    for index in range(count):
        sample = dataset[index]
        probabilities = sample["mask"] if probability_source is None else probability_source(sample)
        maps, union = predict_scene_b(what, sample, probabilities, device, settings)
        labels = (sample["before_label"], sample["after_label"])
        total.append(_histogram(maps, labels))
        inside.append(_histogram(maps, labels, union))
        outside.append(_histogram(maps, labels, ~union))
        density.append(change_density(sample))
        coverage.append(coverage_diagnostics(probabilities, sample["mask"], settings))
    return {
        "sample_ids": list(dataset.sample_ids[:count]), "histograms": np.stack(total),
        "histograms_inside_roi": np.stack(inside), "histograms_outside_roi": np.stack(outside),
        "density": np.array(density), "coverage": coverage,
    }
