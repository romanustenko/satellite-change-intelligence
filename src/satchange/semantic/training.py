"""Semantic batch/loss/metric adapters; epoch and checkpoint orchestration stays in training.py."""

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from satchange.execution import precision_context
from satchange.models import TinySiameseChangeClassifier
from satchange.models.tiny_siamese_change_classifier import validate_classes

TASK = "semantic_roi_classification"
SELECTION_RULE = "mean(before_macro_f1, after_macro_f1) on val; earlier epoch wins ties"


class SemanticCELoss(nn.Module):
    """Loss settings consumed by the shared engine's semantic batch adapter."""

    def __init__(self, class_weights: list[float] | None = None) -> None:
        super().__init__()
        self.register_buffer(
            "weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
        )


def hpo_parameters(config: dict[str, Any], prepared: str | Path) -> dict[str, Any]:
    """Resolve YAML into the existing HPO schema, including content/protocol identity."""
    from satchange.semantic.data import data_protocol, file_digest

    protocol = data_protocol(config)
    training = config["training"]
    return {
        "task": TASK,
        "model_parameters": {"classes": protocol["classes"], "dropout": config.get("dropout", 0.1)},
        "semantic_config": {
            key: deepcopy(config[key])
            for key in (
                "classes",
                "class_ids",
                "ignore_ids",
                "mask_source",
                "roi",
                "label_policy",
                "detector",
            )
            if key in config
        },
        "prepared_identity": file_digest(Path(prepared)),
        "class_weights": training.get("class_weights"),
        "seed": training.get("seed", 42),
        "batch_size": training.get("batch_size", 4),
        "num_workers": training.get("num_workers", 0),
        "learning_rate": training.get("learning_rate", 1e-3),
        "weight_decay": training.get("weight_decay", 1e-4),
        "scheduler": training.get("scheduler", "none"),
        "precision": training.get("precision", "fp32"),
        "tf32": training.get("tf32", False),
    }


def validate_parameters(parameters: dict[str, Any]) -> None:
    """Validate semantic settings before launching GPU workers; reject segmentation-only fields."""
    from satchange.semantic.data import data_protocol

    required = {
        "task",
        "semantic_config",
        "prepared_identity",
        "model_parameters",
        "seed",
        "batch_size",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "class_weights",
        "precision",
        "tf32",
    }
    if required - parameters.keys():
        raise ValueError(f"Missing semantic HPO settings: {sorted(required - parameters.keys())}")
    forbidden = {"image_size", "positive_weight", "bce_weight", "dice_weight", "loss_weights"}
    if forbidden & parameters.keys():
        raise ValueError("Semantic trials cannot use RGB resize, BCE/Dice or positive_weight.")
    protocol = data_protocol(parameters["semantic_config"])
    model_parameters = parameters["model_parameters"]
    if set(model_parameters) != {"classes", "dropout"}:
        raise ValueError("Tiny model_parameters must declare classes and dropout only.")
    classes = validate_classes(model_parameters["classes"])
    if not 0 <= model_parameters["dropout"] < 1:
        raise ValueError("dropout must be in [0,1).")
    if list(classes) != protocol["classes"]:
        raise ValueError("Model class order differs from semantic data taxonomy.")
    for key, lower in (("seed", 0), ("batch_size", 1), ("num_workers", 0)):
        if type(parameters[key]) is not int or parameters[key] < lower:
            raise ValueError(f"{key} must be an integer >= {lower}.")
    if parameters["seed"] >= 2**32:
        raise ValueError("seed must be below 2**32.")
    for key in ("learning_rate", "weight_decay"):
        if not math.isfinite(parameters[key]) or parameters[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if parameters["learning_rate"] == 0 or parameters["scheduler"] not in (
        "none",
        "reduce_on_plateau",
    ):
        raise ValueError("Use a positive LR and a supported project scheduler.")
    if parameters["precision"] not in ("fp32", "bf16") or type(parameters["tf32"]) is not bool:
        raise ValueError("Use fp32/CUDA bf16 and an explicit boolean tf32 setting.")
    weights = parameters["class_weights"]
    if weights is not None and (
        not isinstance(weights, list)
        or len(weights) != len(classes)
        or any(not math.isfinite(w) or w <= 0 for w in weights)
    ):
        raise ValueError(
            "class_weights must be positive, finite, and aligned with the saved classes."
        )


def create_loaders(
    prepared: str | Path, parameters: dict[str, Any]
) -> tuple[DataLoader, DataLoader]:
    """Build native-size list batches before a trial; no ROI generation runs on a worker GPU."""
    from satchange.semantic.data import SemanticROIDataset, collate_rois, file_digest

    path = Path(prepared)
    if path.is_dir():
        path /= "prepared.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared semantic data missing: {path}; run SECOND preparation first."
        )
    if file_digest(path) != parameters["prepared_identity"]:
        raise ValueError("Prepared training identity changed; start a new declared dataset/trial.")
    datasets = [
        SemanticROIDataset(path, split, parameters["semantic_config"], augment=split == "train")
        for split in ("train", "val")
    ]
    datasets[0].verify_sources()
    if "unknown" in parameters["model_parameters"]["classes"]:
        target = parameters["model_parameters"]["classes"].index("unknown")
        if not any(target in (r["before_label"], r["after_label"]) for r in datasets[0].records):
            raise ValueError("A learned unknown class requires labeled training examples.")
    loaders = [
        DataLoader(
            dataset,
            batch_size=parameters["batch_size"],
            num_workers=parameters["num_workers"],
            collate_fn=collate_rois,
            shuffle=index == 0,
            generator=torch.Generator().manual_seed(parameters["seed"] + index),
        )
        for index, dataset in enumerate(datasets)
    ]
    return loaders[0], loaders[1]


class SemanticMetrics:
    """Two global CxC confusion matrices: rows true, columns predicted.

    Undefined precision/recall/F1 are zero; macro F1 includes every declared class,
    including classes with zero support. Joint accuracy requires both dates correct.
    """

    def __init__(self, classes: tuple[str, ...]) -> None:
        self.classes = validate_classes(classes)
        self.matrices = [
            torch.zeros(len(classes), len(classes), dtype=torch.int64) for _ in range(2)
        ]
        self.joint_correct = 0
        self.samples = 0

    def update(
        self,
        logits: dict[str, torch.Tensor],
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> None:
        """Accumulate counts, never per-batch F1 averages."""
        correct = []
        for index, (key, target) in enumerate((("before_logits", before), ("after_logits", after))):
            values = logits[key].detach().float().cpu()
            target = target.detach().cpu()
            if values.shape != (target.numel(), len(self.classes)) or target.ndim != 1:
                raise ValueError("Metrics require logits [B,C] and labels [B].")
            if (
                target.dtype != torch.long
                or (target < 0).any()
                or (target >= len(self.classes)).any()
            ):
                raise ValueError("Metric labels must be int64 indices in the declared class order.")
            if not torch.isfinite(values).all():
                raise ValueError("Semantic logits must be finite.")
            predicted = values.argmax(dim=1)
            count = torch.bincount(
                target * len(self.classes) + predicted, minlength=len(self.classes) ** 2
            )
            self.matrices[index] += count.reshape(len(self.classes), len(self.classes))
            correct.append(predicted == target)
        self.samples += before.numel()
        self.joint_correct += int((correct[0] & correct[1]).sum())

    def compute(self) -> dict[str, Any]:
        """Return counts, class supports, per-date metrics, and the declared selection score."""
        if not self.samples:
            raise ValueError("Cannot evaluate an empty semantic split.")
        result: dict[str, Any] = {
            "classes": list(self.classes),
            "samples": self.samples,
            "joint_accuracy": self.joint_correct / self.samples,
            "zero_denominator_convention": "zero; macro includes all declared classes",
        }
        for date, matrix in zip(("before", "after"), self.matrices, strict=True):
            counts = matrix.double()
            tp, support, predicted = counts.diag(), counts.sum(1), counts.sum(0)
            precision = torch.where(predicted > 0, tp / predicted, 0)
            recall = torch.where(support > 0, tp / support, 0)
            f1 = torch.where(
                precision + recall > 0, 2 * precision * recall / (precision + recall), 0
            )
            result[f"{date}_accuracy"] = float(tp.sum()) / self.samples
            result[f"{date}_macro_f1"] = float(f1.mean())
            result[f"{date}_confusion"] = matrix.tolist()
            result[f"{date}_per_class"] = [
                {
                    "class": name,
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, name in enumerate(self.classes)
            ]
        result["selection_score"] = (result["before_macro_f1"] + result["after_macro_f1"]) / 2
        result["selection_rule"] = SELECTION_RULE
        return result


def logical_forward(
    model: TinySiameseChangeClassifier,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Forward each variable-size pair, then concatenate logits over the logical batch.

    BatchNorm statistics cover each temporal pair, not the full logical minibatch.
    """
    if not samples:
        raise ValueError("Logical batches must not be empty.")
    outputs = [
        model(*(sample[key].unsqueeze(0).to(device) for key in ("before", "after", "mask")))
        for sample in samples
    ]
    logits = {
        key: torch.cat([output[key] for output in outputs])
        for key in ("before_logits", "after_logits")
    }
    labels = [
        torch.tensor([sample[key] for sample in samples], dtype=torch.long, device=device)
        for key in ("before_label", "after_label")
    ]
    return logits, labels[0], labels[1]


def semantic_loss(
    logits: dict[str, torch.Tensor],
    before: torch.Tensor,
    after: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[tuple[float, float]]]:
    """Sum date-wise weighted CE over the logical batch; return exact logging denominators.

    For a singleton logical batch, normalized weighted CE cancels its class weight.
    Split logging sums numerators/denominators separately for each date.
    """
    terms = []
    totals = []
    for key, labels in (("before_logits", before), ("after_logits", after)):
        numerator = F.cross_entropy(logits[key].float(), labels, weight=weights, reduction="sum")
        denominator = (
            weights[labels].sum() if weights is not None else numerator.new_tensor(labels.numel())
        )
        if not torch.isfinite(numerator) or not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError("Semantic CE needs finite logits and a positive loss denominator.")
        terms.append(numerator / denominator)
        totals.append((float(numerator.detach()), float(denominator.detach())))
    return terms[0] + terms[1], totals


def run_epoch(
    model: TinySiameseChangeClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    weights: torch.Tensor | None = None,
    precision: str = "fp32",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Fit or evaluate a split, handling final partial logical batches correctly."""
    model.train(optimizer is not None)
    metrics = SemanticMetrics(model.classes)
    totals = [[0.0, 0.0], [0.0, 0.0]]
    for samples in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None), precision_context(device, precision):
            logits, before, after = logical_forward(model, samples, device)
            loss, batch_totals = semantic_loss(logits, before, after, weights)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        metrics.update(logits, before, after)
        for total, batch_total in zip(totals, batch_totals, strict=True):
            total[0] += batch_total[0]
            total[1] += batch_total[1]
    result = metrics.compute()
    result["loss"] = sum(numerator / denominator for numerator, denominator in totals)
    result["loss_totals"] = totals
    return result
