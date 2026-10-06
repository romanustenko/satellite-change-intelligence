"""Common-checkpoint evaluation and native RGB detector/classifier composition."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from satchange.execution import precision_context, resolve_device
from satchange.models import TinySiameseChangeClassifier
from satchange.models.factory import SEMANTIC_TASK
from satchange.run import reconstruct_checkpoint
from satchange.semantic.data import SemanticROIDataset, collate_rois, data_protocol, file_digest
from satchange.semantic.roi import (
    DEFAULT_ROI_SETTINGS,
    ROISettings,
    align_whole_image_probabilities,
    extract_rois,
)
from satchange.semantic.training import run_epoch, validate_parameters


def load_classifier(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> tuple[TinySiameseChangeClassifier, dict[str, Any]]:
    """Reconstruct a trusted HPO checkpoint via the shared loader."""
    model, state = reconstruct_checkpoint(checkpoint, resolve_device(device))
    if state.get("task") != SEMANTIC_TASK or not isinstance(model, TinySiameseChangeClassifier):
        raise ValueError("Expected a semantic Tiny checkpoint from project HPO.")
    validate_parameters(state["config"])
    if state.get("precision", state["config"]["precision"]) != state["config"]["precision"]:
        raise ValueError("Checkpoint precision metadata is inconsistent.")
    return model.eval(), state


def _numerics(state: dict[str, Any], device: torch.device, override: str | None) -> str:
    precision = state["config"]["precision"] if override is None else override
    precision_context(device, precision)
    torch.backends.cuda.matmul.allow_tf32 = state["config"]["tf32"]
    torch.backends.cudnn.allow_tf32 = state["config"]["tf32"]
    return str(precision)


@torch.inference_mode()
def predict_rois(
    model: TinySiameseChangeClassifier,
    before: torch.Tensor,
    after: torch.Tensor,
    probabilities: torch.Tensor,
    settings: ROISettings = DEFAULT_ROI_SETTINGS,
    *,
    precision: str = "fp32",
) -> list[dict[str, Any]]:
    """Classify native proposals without annotations; confidence is uncalibrated softmax."""
    model.eval()
    device = next(model.parameters()).device
    precision_context(device, precision)
    result = []
    for roi in extract_rois(before, after, probabilities, settings):
        with precision_context(device, precision):
            output = model(*(v.unsqueeze(0).to(device) for v in (roi.before, roi.after, roi.mask)))
        dates = []
        for key in ("before_logits", "after_logits"):
            score, index = output[key].float().softmax(dim=1)[0].max(dim=0)
            dates.append((model.classes[int(index)], float(score)))
        result.append(
            {
                "component_id": roi.component_id,
                "bbox": list(roi.component_bbox),
                "crop_bbox": list(roi.crop_bbox),
                "area": roi.area,
                "minimum_extent_limited": roi.minimum_extent_limited,
                "before_class": dates[0][0],
                "before_score": dates[0][1],
                "after_class": dates[1][0],
                "after_score": dates[1][1],
                "transition": f"{dates[0][0]} -> {dates[1][0]}",
            }
        )
    return result


def predict_checkpoint(
    checkpoint: str | Path,
    before: torch.Tensor,
    after: torch.Tensor,
    probabilities: torch.Tensor,
    device: str | torch.device = "cpu",
    *,
    inference_mask_source: str | None = None,
    detector: dict[str, Any] | None = None,
    settings: ROISettings | None = None,
    precision: str | None = None,
) -> dict[str, Any]:
    """Explicit GT-to-predicted mask shift is supported with the same weights."""
    device = resolve_device(device)
    model, state = load_classifier(checkpoint, device)
    training = data_protocol(state["config"]["semantic_config"])
    source = training["mask_source"] if inference_mask_source is None else inference_mask_source
    if source not in ("ground_truth", "predicted"):
        raise ValueError("Declare ground_truth or predicted inference_mask_source.")
    if source == "predicted" and (not isinstance(detector, dict) or not detector.get("identity")):
        raise ValueError(
            "Predicted inference needs frozen detector identity/preprocessing provenance."
        )
    if source == "ground_truth" and not ((probabilities == 0) | (probabilities == 1)).all():
        raise ValueError("GT support must be binary; explicitly declare predicted probabilities.")
    chosen = settings or ROISettings(**training["roi"])
    actual = _numerics(state, device, precision)
    return {
        "classes": list(model.classes),
        "training_mask_source": training["mask_source"],
        "inference_mask_source": source,
        "training_data_identity": state["config"]["prepared_identity"],
        "detector": detector,
        "proposal_settings": asdict(chosen),
        "normalization": training["normalization"],
        "inference_precision": actual,
        "precision_override": precision,
        "device": str(device),
        "confidence_definition": "uncalibrated per-date softmax scores",
        "predictions": predict_rois(model, before, after, probabilities, chosen, precision=actual),
    }


def evaluate_classifier(
    checkpoint: str | Path,
    prepared: str | Path,
    split: str,
    device: str | torch.device = "cpu",
    *,
    evaluation_config: dict[str, Any] | None = None,
    precision: str | None = None,
    evaluation_batch_size: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Evaluate eligible val/test ROIs, recording both identities and explicit mask-source shift.

    Alternate prepared sets require explicit evaluation_config. Shared split assignments cannot
    change; evaluation groups/pairs cannot overlap fitting data. This is ROI-only accuracy.
    """
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or untouched test.")
    device = resolve_device(device)
    model, state = load_classifier(checkpoint, device)
    parameters = state["config"]
    config = parameters["semantic_config"] if evaluation_config is None else evaluation_config
    training, evaluation = data_protocol(parameters["semantic_config"]), data_protocol(config)
    for key in ("classes", "class_ids", "ignore_ids", "normalization", "label_policy"):
        if training[key] != evaluation[key]:
            raise ValueError(f"Evaluation {key} differs from saved semantic protocol.")
    identity = file_digest(Path(prepared))
    if evaluation_config is None and identity != parameters["prepared_identity"]:
        raise ValueError("Alternate prepared identity needs explicit evaluation_config.")
    dataset = SemanticROIDataset(prepared, split, config)
    dataset.verify_sources()
    original = parameters.get("training_data", {}).get("scenes", [])
    if not original:
        raise ValueError("Checkpoint lacks scene/split provenance for semantic evaluation.")
    known = {scene["sample_id"]: scene for scene in original}
    train_groups = {s["group_id"] for s in original if s["split"] == "train"}
    train_pairs = {
        tuple(sorted(s["rgb_hashes"].values())) for s in original if s["split"] == "train"
    }
    for scene in dataset.document["provenance"]["scenes"]:
        if scene["split"] != split:
            continue
        if scene["sample_id"] in known and known[scene["sample_id"]]["split"] != split:
            raise ValueError("Evaluation changes a saved scene split.")
        if (
            scene["group_id"] in train_groups
            or tuple(sorted(scene["rgb_hashes"].values())) in train_pairs
        ):
            raise ValueError("Evaluation RGB/group leakage from classifier training data.")
    weights = parameters["class_weights"]
    actual = _numerics(state, device, precision)
    result = run_epoch(
        model,
        DataLoader(
            dataset,
            batch_size=evaluation_batch_size or parameters["batch_size"],
            num_workers=num_workers,
            collate_fn=collate_rois,
        ),
        device,
        precision=actual,
        weights=torch.tensor(weights, device=device) if weights is not None else None,
    )
    return {
        **result,
        "split": split,
        "seed": parameters["seed"],
        "training_mask_source": training["mask_source"],
        "evaluation_mask_source": evaluation["mask_source"],
        "training_data_identity": parameters["prepared_identity"],
        "evaluation_data_identity": identity,
        "training_protocol": training,
        "evaluation_protocol": evaluation,
        "domain_shift": training["mask_source"] != evaluation["mask_source"],
        "evaluation_precision": actual,
        "precision_override": precision,
        "device": str(device),
        "checkpoint_identity": file_digest(Path(checkpoint)),
        "coverage": dataset.document["coverage"][split],
        "coverage_definition": dataset.document["coverage_definition"],
        "evaluated_epoch": state["epoch"],
    }


@torch.inference_mode()
def detector_probability_map(
    detector: nn.Module,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    input_size: tuple[int, int],
    output_kind: str,
    geometry: str = "whole_image_resize",
    precision: str = "fp32",
    normalization: str = "rgb_0_1",
) -> torch.Tensor:
    """Frozen detector with declared preprocessing; only RGB/255 whole-image resize is supported.

    Reuse reconstruct_checkpoint for detector loading. Native classifier RGB is untouched.
    Cropped/tiled/padded/custom-normalized inference must provide a known aligned map instead.
    """
    if geometry != "whole_image_resize" or normalization != "rgb_0_1":
        raise ValueError(
            "Unsupported detector preprocessing; supply an explicitly aligned saved map."
        )
    if output_kind not in ("logits", "probabilities"):
        raise ValueError("Declare detector output_kind as logits or probabilities.")
    if before.ndim != 3 or before.shape[0] != 3 or before.shape != after.shape:
        raise ValueError("Detector requires aligned native RGB [3,H,W].")
    for value in (before, after):
        if (
            value.dtype != torch.float32
            or not torch.isfinite(value).all()
            or value.min() < 0
            or value.max() > 1
        ):
            raise ValueError("Detector inputs must be float32 RGB in [0,1].")
    if len(input_size) != 2 or any(type(v) is not int or v < 1 for v in input_size):
        raise ValueError("Declare positive whole-image detector input_size (H,W).")
    detector.eval()
    device = next(detector.parameters()).device
    inputs = [
        F.interpolate(v.unsqueeze(0), size=input_size, mode="bilinear", align_corners=False).to(
            device
        )
        for v in (before, after)
    ]
    with precision_context(device, precision):
        output = detector(*inputs)
    if output.shape != (1, 1, *input_size) or not torch.isfinite(output).all():
        raise ValueError("Detector output must be finite [1,1,input H,input W].")
    probabilities = output.float().sigmoid() if output_kind == "logits" else output.float()
    return align_whole_image_probabilities(
        probabilities[0].cpu(), (before.shape[-2], before.shape[-1]), input_size
    )


def predict_detector_checkpoint(
    classifier_checkpoint: str | Path,
    detector_checkpoint: str | Path,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    input_size: tuple[int, int],
    output_kind: str,
    device: str | torch.device = "cpu",
    detector_precision: str | None = None,
    classifier_precision: str | None = None,
) -> dict[str, Any]:
    """Compose the shared detector loader, aligned map, and native classifier."""
    detector, detector_state = reconstruct_checkpoint(detector_checkpoint, resolve_device(device))
    detector_config = detector_state["config"]
    if detector_precision is None:
        detector_precision = detector_state.get(
            "precision", detector_config.get("precision", "fp32")
        )
    detector_tf32 = bool(detector_config.get("tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = detector_tf32
    torch.backends.cudnn.allow_tf32 = detector_tf32
    probabilities = detector_probability_map(
        detector,
        before,
        after,
        input_size=input_size,
        output_kind=output_kind,
        precision=detector_precision,
    )
    return predict_checkpoint(
        classifier_checkpoint,
        before,
        after,
        probabilities,
        device,
        inference_mask_source="predicted",
        precision=classifier_precision,
        detector={
            "identity": file_digest(Path(detector_checkpoint)),
            "input_size": list(input_size),
            "output_kind": output_kind,
            "precision": detector_precision,
            "tf32": detector_tf32,
            "geometry": "whole_image_resize",
            "normalization": "rgb_0_1",
        },
    )
