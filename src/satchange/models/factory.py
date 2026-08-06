"""Model construction utilities."""


from typing import Any

from torch import nn

from satchange.models.fc_siam_diff import FCSiamDiff


MODEL_REGISTRY = {
    "fc_siam_diff": FCSiamDiff,
}


def create_change_model(
    model_name: str,
    parameters: dict[str, Any],
) -> nn.Module:
    """Create a registered change-detection model."""

    model_name = model_name.strip().lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {tuple(MODEL_REGISTRY)}."
        )

    model_class = MODEL_REGISTRY[
        model_name
    ]

    return model_class(
        **parameters
    )