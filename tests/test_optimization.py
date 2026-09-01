"""Tests for pure hyperparameter-search helpers."""

from copy import deepcopy

import pytest

from satchange.losses import BCEDiceLoss
from satchange.models import BIT, create_change_model
from satchange.optimization import _build_trial_parameters, _expand_search_space


def test_build_trial_parameters_maps_architecture_training_and_loss_values() -> None:
    fixed_parameters = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "model_parameters": {
            "embedding_dim": 8,
            "token_length": 2,
            "num_heads": 3,
            "encoder_head_dim": 4,
            "decoder_head_dim": 4,
            "encoder_depth": 1,
            "decoder_depth": 1,
        },
    }
    search_parameters = {
        "model__embedding_dim": 10,
        "model__decoder_head_dim": 7,
        "learning_rate": 3e-4,
        "loss_weights": (0.3, 0.7),
    }
    original_fixed = deepcopy(fixed_parameters)
    original_search = deepcopy(search_parameters)

    parameters = _build_trial_parameters(fixed_parameters, search_parameters)
    model = create_change_model("bit", parameters["model_parameters"])
    loss = BCEDiceLoss(
        bce_weight=parameters["bce_weight"],
        dice_weight=parameters["dice_weight"],
    )

    assert isinstance(model, BIT)
    assert model.embedding_dim == 10
    assert model.decoder_layers[0].cross_attention.head_dim == 7
    assert parameters["learning_rate"] == pytest.approx(3e-4)
    assert parameters["bce_weight"] == pytest.approx(0.3)
    assert parameters["dice_weight"] == pytest.approx(0.7)
    assert "loss_weights" not in parameters
    assert not any(name.startswith("model__") for name in parameters)
    assert loss.bce_weight == pytest.approx(0.3)
    assert loss.dice_weight == pytest.approx(0.7)
    assert fixed_parameters == original_fixed
    assert search_parameters == original_search


def test_build_trial_parameters_deep_copies_nested_values() -> None:
    fixed_parameters = {
        "model_parameters": {"embedding_dim": 8},
        "metadata": {"tags": ["fixed"]},
    }

    parameters = _build_trial_parameters(fixed_parameters, {})
    parameters["model_parameters"]["embedding_dim"] = 16
    parameters["metadata"]["tags"].append("trial")

    assert fixed_parameters == {
        "model_parameters": {"embedding_dim": 8},
        "metadata": {"tags": ["fixed"]},
    }


def test_build_trial_parameters_rejects_invalid_special_values() -> None:
    with pytest.raises(ValueError, match="model__"):
        _build_trial_parameters({}, {"model__": 8})

    with pytest.raises(ValueError, match="loss_weights"):
        _build_trial_parameters({}, {"loss_weights": (0.5,)})


def test_expand_search_space_keeps_model_parameter_names() -> None:
    trials = _expand_search_space(
        {
            "model__embedding_dim": [16, 32],
            "learning_rate": [1e-4],
        }
    )

    assert trials == [
        {"model__embedding_dim": 16, "learning_rate": 1e-4},
        {"model__embedding_dim": 32, "learning_rate": 1e-4},
    ]
