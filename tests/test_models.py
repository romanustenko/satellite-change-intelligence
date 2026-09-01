"""Tests for change detection model interfaces."""

from typing import Any

import pytest
import torch
from torch import nn

from satchange.losses import BCEDiceLoss
from satchange.models import BIT, FCSiamDiff, count_model_parameters, create_change_model


def _small_bit(*, decoder_depth: int = 2) -> BIT:
    return BIT(
        embedding_dim=8,
        token_length=2,
        num_heads=3,
        encoder_head_dim=4,
        decoder_head_dim=4,
        encoder_depth=1,
        decoder_depth=decoder_depth,
    )


def test_bit_construction_factory_and_parameter_count() -> None:
    direct_model = BIT()
    factory_model = create_change_model(
        model_name="bit",
        parameters={
            "embedding_dim": 8,
            "token_length": 2,
            "num_heads": 3,
            "encoder_head_dim": 4,
            "decoder_head_dim": 5,
            "encoder_depth": 1,
            "decoder_depth": 1,
        },
    )

    assert isinstance(direct_model, BIT)
    assert isinstance(factory_model, BIT)
    assert not any(isinstance(module, nn.Sigmoid) for module in direct_model.modules())
    parameter_counts = count_model_parameters(direct_model)
    assert parameter_counts["total_parameters"] > 0
    assert parameter_counts["total_parameters"] == parameter_counts["trainable_parameters"]
    assert parameter_counts["total_parameters"] < 11_280_133 // 2


def test_bit_backbone_and_projection_shapes() -> None:
    model = _small_bit().eval()
    inputs = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        stem_features = model.backbone.stem(inputs)
        layer1_features = model.backbone.layer1(stem_features)
        layer2_features = model.backbone.layer2(layer1_features)
        backbone_features = model.backbone.layer3(layer2_features)
        projected_features = model._encode(inputs)

    assert stem_features.shape == (1, 64, 64, 64)
    assert layer1_features.shape == (1, 64, 64, 64)
    assert layer2_features.shape == (1, 128, 32, 32)
    assert backbone_features.shape == (1, 256, 32, 32)
    assert model.backbone(inputs).shape == (1, 256, 32, 32)
    assert model.feature_projection.kernel_size == (3, 3)
    assert projected_features.shape == (1, 8, 64, 64)


def test_bit_tokenizer_attention_is_spatially_normalized() -> None:
    model = _small_bit()
    features = torch.rand(2, 8, 16, 16)

    attention = torch.softmax(model.token_attention(features).flatten(2), dim=-1)
    tokens = model._tokenize(features)

    assert tokens.shape == (2, 2, 8)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 2))


def test_bit_joint_encoder_and_decoder_shapes() -> None:
    model = _small_bit()
    before_tokens = torch.rand(2, 2, 8)
    after_tokens = torch.rand(2, 2, 8)

    before_context, after_context = model._encode_tokens(before_tokens, after_tokens)
    decoded = model._decode_features(torch.rand(2, 8, 16, 16), before_context)

    assert before_context.shape == (2, 2, 8)
    assert after_context.shape == (2, 2, 8)
    assert decoded.shape == (2, 8, 16, 16)
    assert not hasattr(model, "decoder_norm")


def test_bit_attention_head_dimensions_are_independent() -> None:
    model = BIT(
        embedding_dim=10,
        num_heads=3,
        encoder_head_dim=5,
        decoder_head_dim=7,
        decoder_depth=1,
    )

    assert model.encoder_layers[0].attention.inner_dim == 15
    assert model.decoder_layers[0].cross_attention.inner_dim == 21


@pytest.mark.parametrize("spatial_size", [(256, 256), (128, 128), (128, 160)])
def test_bit_forward_preserves_supported_spatial_size(
    spatial_size: tuple[int, int],
) -> None:
    model = _small_bit(decoder_depth=1).eval()
    before = torch.rand(1, 3, *spatial_size)
    after = torch.rand(1, 3, *spatial_size)
    shared_layer_calls: list[nn.Module] = []
    handle = model.backbone.layer1.register_forward_hook(
        lambda module, inputs, output: shared_layer_calls.append(module)
    )

    with torch.no_grad():
        logits = model(before, after)
    handle.remove()

    assert logits.shape == (1, 1, *spatial_size)
    assert torch.isfinite(logits).all()
    assert shared_layer_calls == [model.backbone.layer1, model.backbone.layer1]


def test_bit_real_training_step_reaches_all_major_components() -> None:
    torch.manual_seed(3)
    model = _small_bit().train()
    loss_function = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7, pos_weight=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = torch.rand(1, 3, 128, 128)
    after = torch.rand(1, 3, 128, 128)
    mask = torch.randint(0, 2, (1, 1, 128, 128), dtype=torch.float32)
    first_backbone_weight = model.backbone.stem[0].weight.detach().clone()

    optimizer.zero_grad()
    logits = model(before, after)
    loss = loss_function(logits, mask)
    loss.backward()

    parameters_with_required_gradients = (
        model.backbone.stem[0].weight,
        model.feature_projection.weight,
        model.token_attention.weight,
        model.positional_embedding,
        model.encoder_layers[0].attention.query_projection.weight,
        model.decoder_layers[0].cross_attention.query_projection.weight,
        model.decoder_layers[-1].cross_attention.query_projection.weight,
        model.prediction_head[0].weight,
        model.prediction_head[-1].weight,
    )
    for parameter in parameters_with_required_gradients:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    optimizer.step()
    assert not torch.equal(first_backbone_weight, model.backbone.stem[0].weight)


@pytest.mark.parametrize(
    "parameters",
    [
        {"token_length": 0},
        {"embedding_dim": 0},
        {"num_heads": 0},
        {"encoder_head_dim": 0},
        {"decoder_head_dim": 0},
        {"encoder_depth": 0},
        {"decoder_depth": 0},
        {"in_channels": 1},
        {"out_channels": 2},
        {"dropout": 1.0},
        {"pretrained_backbone": True},
    ],
)
def test_bit_rejects_invalid_constructor_parameters(
    parameters: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        BIT(**parameters)


@pytest.mark.parametrize(
    ("before_shape", "after_shape"),
    [
        ((3, 128, 128), (3, 128, 128)),
        ((1, 3, 128, 128), (2, 3, 128, 128)),
        ((1, 1, 128, 128), (1, 1, 128, 128)),
        ((1, 3, 100, 128), (1, 3, 100, 128)),
        ((1, 3, 16, 32), (1, 3, 16, 32)),
    ],
)
def test_bit_rejects_invalid_inputs(
    before_shape: tuple[int, ...],
    after_shape: tuple[int, ...],
) -> None:
    model = _small_bit()

    with pytest.raises(ValueError):
        model(torch.rand(before_shape), torch.rand(after_shape))


def test_bit_prediction_head_is_full_resolution_binary_head() -> None:
    model = _small_bit()

    assert isinstance(model.prediction_head[0], nn.Conv2d)
    assert model.prediction_head[0].kernel_size == (3, 3)
    assert model.prediction_head[0].bias is None
    assert isinstance(model.prediction_head[1], nn.BatchNorm2d)
    assert isinstance(model.prediction_head[2], nn.ReLU)
    assert isinstance(model.prediction_head[3], nn.Conv2d)
    assert model.prediction_head[3].kernel_size == (3, 3)
    assert model.prediction_head[3].out_channels == 1


def test_fc_siam_diff_forward_and_backward_regression() -> None:
    model = FCSiamDiff(base_channels=2)
    before = torch.rand(1, 3, 16, 16)
    after = torch.rand(1, 3, 16, 16)

    logits = model(before, after)

    assert logits.shape == (1, 1, 16, 16)
    assert torch.isfinite(logits).all()

    logits.square().mean().backward()
    assert model.output_layer.weight.grad is not None


def test_model_parameter_counts() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    model[1].weight.requires_grad_(False)
    model[1].bias.requires_grad_(False)

    assert count_model_parameters(model) == {
        "total_parameters": 11,
        "trainable_parameters": 8,
    }
