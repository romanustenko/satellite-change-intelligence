"""Offline Heavy conformance, numerical attention, and real common training tests."""

import os
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from satchange.losses import BCEDiceLoss
from satchange.models import SiamMiTB3Diff, count_model_parameters, create_change_model
from satchange.models.siam_mit_b3_diff import SpatialReductionAttention
from satchange.optimization import _build_trial_parameters, _expand_search_space
from satchange.run import reconstruct_checkpoint, resolve_parameters
from satchange.training import evaluate_thresholds, fit_binary_change_model
from satchange.utils import load_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_default_architecture_and_actual_multiscale_flow():
    config = load_config("configs/siam_mit_b3_diff_resized.yaml")
    saved = deepcopy(config)
    model = create_change_model(
        config["model"]["name"], resolve_parameters(config)["model_parameters"]
    )
    model.eval()
    assert config == saved
    assert [len(stage) for stage in model.encoder.stages] == [3, 4, 18, 3]
    blocks = [block for stage in model.encoder.stages for block in stage]
    assert len({id(block) for block in blocks}) == 28
    assert count_model_parameters(model) == {
        "total_parameters": 45_772_961,
        "trainable_parameters": 45_772_961,
    }
    assert sum(p.numel() for p in model.encoder.parameters()) == 44_072_128
    assert [block.drop_path.probability for block in blocks] == pytest.approx(
        torch.linspace(0, 0.1, 28).tolist()
    )
    for stage, channels, heads, reduction in zip(
        model.encoder.stages, (64, 128, 320, 512), (1, 2, 5, 8), (8, 4, 2, 1), strict=True
    ):
        for block in stage:
            assert block.attention.heads == heads
            assert block.attention.head_dim == 64
            assert block.attention.sr_ratio == reduction
            assert block.ffn.depthwise.groups == channels * 4
            assert block.ffn.depthwise.kernel_size == (3, 3)
            assert block.ffn.fc1.out_features == channels * 4
            assert block.norm1.eps == block.norm2.eps == 1e-6
            assert (hasattr(block.attention, "sr")) == (reduction > 1)
            if reduction > 1:
                assert block.attention.sr.groups == 1
                assert block.attention.sr_norm.eps == 1e-5
    features, projections, decoder, refinements, calls, attention = [], [], [], [], [], []
    handles = [
        model.encoder.register_forward_hook(
            lambda module, args, output: (
                (
                    calls.append(id(module)),
                    features.extend([tuple(value.shape) for value in output]),
                )
                and None
            )
        )
    ]
    for module in model.projections:
        handles.append(
            module.register_forward_hook(
                lambda module, args, output: projections.append(tuple(output.shape))
            )
        )
    for module in [model.coarsest, *model.fusions]:
        handles.append(
            module.register_forward_hook(
                lambda module, args, output: decoder.append(tuple(output.shape))
            )
        )
    for module in model.refinements:
        handles.append(
            module.register_forward_hook(
                lambda module, args, output: refinements.append(tuple(output.shape))
            )
        )
    sdpa = F.scaled_dot_product_attention

    def record(q, k, v, **kwargs):
        attention.append((q.shape[-2], k.shape[-2], q.shape[-1]))
        return sdpa(q, k, v, **kwargs)

    with torch.no_grad(), patch.object(F, "scaled_dot_product_attention", side_effect=record):
        output = model(torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256))
    for handle in handles:
        handle.remove()
    assert calls == [id(model.encoder)] * 2
    assert features == [(1, 64, 64, 64), (1, 128, 32, 32), (1, 320, 16, 16), (1, 512, 8, 8)] * 2
    assert projections == [(1, 128, size, size) for size in (64, 32, 16, 8)]
    assert decoder == [(1, 128, size, size) for size in (8, 16, 32, 64)]
    assert refinements == [(1, 64, 128, 128), (1, 32, 256, 256)]
    assert (
        attention
        == ([(4096, 64, 64)] * 3 + [(1024, 64, 64)] * 4 + [(256, 64, 64)] * 18 + [(64, 64, 64)] * 3)
        * 2
    )
    assert output.shape == (1, 1, 256, 256) and output.isfinite().all()


@pytest.mark.parametrize("reduction", [1, 2])
def test_attention_matches_explicit_reference_outputs_and_gradients(reduction):
    torch.manual_seed(4)
    model = SpatialReductionAttention(8, 2, reduction, 0, 0).double()
    reference = deepcopy(model)
    x = torch.randn(2, 16, 8, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    output = model(x, 4, 4)
    q = reference.q(y).reshape(2, 16, 2, 4).transpose(1, 2)
    source = y
    if reduction > 1:
        source = reference.sr(y.transpose(1, 2).reshape(2, 8, 4, 4))
        source = reference.sr_norm(source.flatten(2).transpose(1, 2))
    k, v = reference.kv(source).reshape(2, -1, 2, 2, 4).permute(2, 0, 3, 1, 4)
    attended = torch.softmax(q @ k.transpose(-1, -2) / 2, dim=-1) @ v
    expected = reference.projection(attended.transpose(1, 2).reshape(2, 16, 8))
    torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-10)
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-8, atol=1e-10)
    for actual, target in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, target.grad, rtol=1e-8, atol=1e-10)
    model.attention_dropout = 0.9
    model.eval()
    torch.testing.assert_close(model(x, 4, 4), output)


@pytest.mark.parametrize("size", [(64, 64), (128, 128), (128, 256)])
def test_shapes_symmetry_identical_differences_and_raw_logits(size):
    model = SiamMiTB3Diff(encoder_depths=(1, 1, 1, 1)).eval()  # Isolated depth ablation.
    a, b = torch.rand(1, 3, *size), torch.rand(1, 3, *size)
    with torch.no_grad():
        torch.testing.assert_close(model(a, b), model(b, a))
        differences = []
        hooks = [
            module.register_forward_pre_hook(
                lambda module, args: differences.append(args[0].abs().max().item())
            )
            for module in model.projections
        ]
        model(a, a)
        assert differences == [0] * 4
        for hook in hooks:
            hook.remove()
        model.head.weight.zero_()
        model.head.bias.fill_(2)
        assert torch.equal(model(a, b), torch.full((1, 1, *size), 2.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 1},
        {"out_channels": 2},
        {"encoder_depths": (1, 2)},
        {"encoder_depths": (1, 1, 0, 1)},
        {"encoder_depths": (1, 1, True, 1)},
        {"refinement_channels": (64,)},
        {"decoder_channels": 127},
        {"norm_groups": 0},
        {"dropout": -0.1},
        {"attention_dropout": 1},
        {"drop_path_rate": float("nan")},
        {"use_checkpointing": "true"},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        SiamMiTB3Diff(**kwargs)


def test_pretraining_and_input_validation():
    with pytest.raises(NotImplementedError, match="provenance"):
        SiamMiTB3Diff(pretrained_backbone=True)
    model = SiamMiTB3Diff(encoder_depths=(1, 1, 1, 1))
    for shape in [(3, 64, 64), (1, 1, 64, 64), (0, 3, 64, 64), (1, 3, 16, 32), (1, 3, 64, 65)]:
        with pytest.raises(ValueError):
            model(torch.rand(shape), torch.rand(shape))
    a = torch.rand(1, 3, 64, 64)
    for b in [a.long(), a.double(), a[:, :, :32], torch.empty_like(a, device="meta")]:
        with pytest.raises(ValueError):
            model(a, b)
    with pytest.raises(ValueError):
        model(a.long(), a.long())


@pytest.mark.parametrize("stochastic", [False, True])
def test_checkpointing_preserves_outputs_gradients_and_rng(stochastic):
    torch.manual_seed(13)
    model = SiamMiTB3Diff(
        encoder_depths=(1, 1, 1, 1),
        dropout=0.1 if stochastic else 0,
        drop_path_rate=0.2 if stochastic else 0,
    )
    a = torch.rand(2, 3, 64, 64, requires_grad=True)
    b = torch.rand(2, 3, 64, 64, requires_grad=True)
    torch.manual_seed(33)
    expected = model(a, b)
    expected.mean().backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.encoder.use_checkpointing = True
    torch.manual_seed(33)
    output = model(a.detach(), b.detach())  # Dataset inputs need no requires_grad.
    output.mean().backward()
    torch.testing.assert_close(output, expected)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-4, atol=2e-6)


def test_default_depth_common_fit_gradients_checkpoint_and_resume(tmp_path):
    torch.manual_seed(42)
    config = load_config("configs/siam_mit_b3_diff_resized.yaml")
    config["model"]["parameters"]["drop_path_rate"] = 0
    model = create_change_model(config["model"]["name"], config["model"]["parameters"])
    sample = {
        "before": torch.rand(3, 64, 64),
        "after": torch.rand(3, 64, 64),
        "mask": torch.randint(0, 2, (1, 64, 64)).float(),
        "sample_id": "synthetic",
    }
    loader = DataLoader([sample], batch_size=1)
    loss = BCEDiceLoss(**config["training"]["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    }
    history = fit_binary_change_model(
        model,
        loader,
        loader,
        loss,
        optimizer,
        torch.device("cpu"),
        1,
        [0.3, 0.5, 0.7],
        tmp_path,
        "siam_mit_b3_diff",
        "smoke",
        config=config,
    )
    # Every trainable module participates, including stage 3 block 18 and every decoder block.
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name
        if name.endswith("weight"):
            assert parameter.grad.abs().sum() > 0, name
            assert not torch.equal(parameter, initial[name]), name
    assert history[0]["train_loss"] > 0 and history[0]["best_threshold"] in (0.3, 0.5, 0.7)
    checkpoint = tmp_path / "siam_mit_b3_diff/smoke/last.pt"
    assert checkpoint.with_name("best.pt").is_file()
    del initial, optimizer, model
    fresh, state = reconstruct_checkpoint(checkpoint, torch.device("cpu"))
    evaluation = evaluate_thresholds(fresh, loader, loss, torch.device("cpu"), [0.3, 0.5, 0.7])
    assert evaluation["loss"] == pytest.approx(history[0]["validation_loss"])
    optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-4, weight_decay=0.01)
    resumed = fit_binary_change_model(
        fresh,
        loader,
        loader,
        loss,
        optimizer,
        torch.device("cpu"),
        1,
        [0.3, 0.5, 0.7],
        tmp_path,
        "siam_mit_b3_diff",
        "smoke",
        config=config,
        resume_from=checkpoint,
    )
    assert [row["epoch"] for row in resumed] == [1, 2]
    assert state["config"] == config


def test_heavy_hpo_sequences_and_independent_configs():
    fixed = resolve_parameters(load_config("configs/siam_mit_b3_diff_resized.yaml"))
    saved = deepcopy(fixed)
    candidates = _expand_search_space(
        {
            "model__encoder_depths": [(3, 4, 6, 3), (3, 4, 18, 3)],
            "loss_weights": [(0.3, 0.7)],
            "learning_rate": [3e-4],
        }
    )
    trials = [_build_trial_parameters(fixed, candidate) for candidate in candidates]
    assert len(trials) == 2
    assert trials[0]["model_parameters"]["encoder_depths"] == (3, 4, 6, 3)
    assert trials[1]["model_parameters"]["encoder_depths"] == (3, 4, 18, 3)
    assert trials[0]["learning_rate"] == 3e-4
    assert trials[0]["bce_weight"] == 0.3 and trials[0]["dice_weight"] == 0.7
    assert not any(key.startswith("model__") for key in trials[0])
    trials[0]["model_parameters"]["refinement_channels"][0] = 8
    assert trials[1]["model_parameters"]["refinement_channels"] == [64, 32]
    assert fixed == saved
    import json

    assert json.loads(json.dumps(trials))[1]["model_parameters"]["encoder_depths"] == [3, 4, 18, 3]


@pytest.mark.parametrize("regime", ["resized", "full_resolution"])
def test_explicitly_allocated_cuda_real_step(regime):
    from satchange.run import probe_training_step

    device_name = os.environ.get("SATCHANGE_TEST_CUDA_DEVICE")
    if not device_name or not torch.cuda.is_available():
        pytest.skip("Requires CUDA and explicit SATCHANGE_TEST_CUDA_DEVICE allocation.")
    if regime == "full_resolution" and os.environ.get("SATCHANGE_TEST_NATIVE") != "1":
        pytest.skip("Native 1024 training probe additionally requires SATCHANGE_TEST_NATIVE=1.")
    config = load_config(f"configs/siam_mit_b3_diff_{regime}.yaml")
    config["runtime"]["precision"] = "bf16"
    result = probe_training_step(
        config, torch.device(device_name), enabled=True, allow_native=regime == "full_resolution"
    )
    assert result["status"] == "completed"
    assert result["loss"] > 0 and result["peak_allocated_mib"] > 0
