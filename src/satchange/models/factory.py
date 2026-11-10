"""Model construction utilities."""

from typing import Any

from torch import nn

from satchange.models.bit import BIT
from satchange.models.fc_siam_diff import FCSiamDiff
from satchange.models.fc_siam_semantic import FCSiamSemantic
from satchange.models.mamba_scd import MambaSCD
from satchange.models.scannet import SCanNet
from satchange.models.siam_mit_b3_diff import SiamMiTB3Diff
from satchange.models.siam_unet_resnet import SiamUNetResNet18

BINARY_TASK = "binary_change_detection"
SEMANTIC_TASK = "semantic_change_detection"
SEMANTIC_OUTPUT_CHANNELS = 14  # T1 and T2 class logits, 7 classes each (0 = no change)
SEMANTIC_REGISTRY: dict[str, type[nn.Module]] = {
    "fc_siam_diff": FCSiamDiff,  # symmetric diff decoder; kept as the documented negative baseline
    "fc_siam_semantic": FCSiamSemantic,
    # Research 2 v2 multi-head SCD models: forward -> (change, semantic_t1, semantic_t2)
    "mamba_scd": MambaSCD,
    "scannet": SCanNet,
    "siam_unet_resnet18": SiamUNetResNet18,
}
MULTI_HEAD_SEMANTIC_MODELS = ("mamba_scd", "scannet", "siam_unet_resnet18")
SEMANTIC_MODELS = tuple(
    name for name in SEMANTIC_REGISTRY if name not in MULTI_HEAD_SEMANTIC_MODELS
)


def model_class_for_task(model_name: str, task: str = BINARY_TASK) -> type[nn.Module]:
    """Select the model class for a supported task contract."""
    if task == BINARY_TASK and model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    if task == SEMANTIC_TASK and model_name in SEMANTIC_REGISTRY:
        return SEMANTIC_REGISTRY[model_name]
    raise ValueError(f"Unsupported model/task pair: {model_name}/{task}")


def create_task_model(
    model_name: str,
    parameters: dict[str, Any],
    task: str = BINARY_TASK,
) -> nn.Module:
    """Construct a model with an explicit binary-detector or dense-semantic contract.

    Multi-head SCD models (``MULTI_HEAD_SEMANTIC_MODELS``) have their own head parameters and no
    ``out_channels``; everything else keeps the joint 14-channel semantic contract.
    """
    joint = task == SEMANTIC_TASK and model_name not in MULTI_HEAD_SEMANTIC_MODELS
    if joint and parameters.get("out_channels") != SEMANTIC_OUTPUT_CHANNELS:
        raise ValueError(
            f"Semantic models need out_channels={SEMANTIC_OUTPUT_CHANNELS} (T1+T2, 7 classes)."
        )
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
