"""Model construction utilities."""

from typing import Any

from torch import nn

from satchange.models.bit import BIT
from satchange.models.fc_siam_diff import FCSiamDiff
from satchange.models.siam_mit_b3_diff import SiamMiTB3Diff

BINARY_TASK = "binary_change_detection"


def model_class_for_task(model_name: str, task: str = BINARY_TASK) -> type[nn.Module]:
    """Select the model class for a supported task contract (binary change detection only)."""
    if task == BINARY_TASK and model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    raise ValueError(f"Unsupported model/task pair: {model_name}/{task}")


def create_task_model(
    model_name: str,
    parameters: dict[str, Any],
    task: str = BINARY_TASK,
) -> nn.Module:
    """Construct a model with an explicit detector or ROI-classifier contract."""
    return model_class_for_task(model_name, task)(**parameters)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "siam_mit_b3_diff": SiamMiTB3Diff,
    "bit": BIT,
    "fc_siam_diff": FCSiamDiff,
}


def count_model_parameters(
    model: nn.Module,
) -> dict[str, int]:
    """Return total and trainable parameter counts for a model."""

    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def create_change_model(
    model_name: str,
    parameters: dict[str, Any],
) -> nn.Module:
    """Create a registered change-detection model."""

    model_name = model_name.strip().lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {tuple(MODEL_REGISTRY)}."
        )

    model_class = MODEL_REGISTRY[model_name]

    return model_class(**parameters)
