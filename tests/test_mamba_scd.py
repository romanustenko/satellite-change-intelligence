"""MambaSCD port (R2-V4-T04): shapes, scan geometry, published-weight compatibility."""

import os
from pathlib import Path

import pytest
import torch

from satchange.models.mamba_scd import (
    MambaSCD,
    cross_merge,
    cross_scan,
    load_vmamba_encoder_weights,
)
from satchange.models.selective_scan import selective_scan_kernel_available
from satchange.semantic.compute import count_training_step_flops
from satchange.semantic.scd_outputs import maps_from_change_and_semantic

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "artifacts/research2/v2/reference/changemamba/MambaSCD_Tiny_SECOND_SeK_0.2208.pth"
)
IMAGENET = ROOT / "artifacts/research2/v2/pretrained/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth"
IMAGENET_SHA256 = "09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201"
TINY = {"depths": [1, 1, 1, 1], "embed_dim": 8, "hidden_dim": 8, "drop_path_rate": 0.0,
        "decoder_drop_path": 0.0}


def test_cross_scan_directions_and_merge_inverse():
    x = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).view(2, 3, 2, 4)
    xs = cross_scan(x)
    assert xs.shape == (2, 4, 3, 8)
    assert torch.equal(xs[:, 0], x.flatten(2))
    assert torch.equal(xs[:, 1], x.transpose(2, 3).flatten(2))
    assert torch.equal(xs[:, 2], xs[:, 0].flip(-1)) and torch.equal(xs[:, 3], xs[:, 1].flip(-1))
    assert torch.equal(cross_merge(xs, 2, 4), 4 * x.flatten(2))


def test_tiny_model_shapes_validation_and_common_maps():
    torch.manual_seed(0)
    model = MambaSCD(**TINY).eval()
    before, after = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    change, first, second = model(before, after)
    assert change.shape == (1, 2, 64, 64) and first.shape == second.shape == (1, 7, 64, 64)
    maps = maps_from_change_and_semantic(change, first, second, semantic_argmax="all")
    assert maps[0].shape == (1, 64, 64) and int(maps[0].max()) <= 6
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.rand(1, 3, 48, 40), torch.rand(1, 3, 48, 40))
    with pytest.raises(ValueError, match="depths"):
        MambaSCD(depths=[2, 2, 4])
    with pytest.raises(ValueError, match="scan_backend"):
        MambaSCD(scan_backend="triton")  # type: ignore[arg-type]


def test_training_step_flops_include_the_analytic_scan():
    model = MambaSCD(**TINY)
    flops = count_training_step_flops(
        model, (torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)),
        lambda outputs: sum(o.float().mean() for o in outputs),
    )
    assert flops.forward_analytic > 0 and flops.backward_analytic == 2 * flops.forward_analytic
    assert flops.forward_counted > flops.forward_analytic


def test_default_architecture_has_published_tiny_size():
    model = MambaSCD()
    assert sum(p.numel() for p in model.parameters()) == 37_133_865 - _buffer_numel(model)


def _buffer_numel(model: torch.nn.Module) -> int:
    """Published state dict counts BatchNorm buffers too (running stats, counters)."""
    return sum(b.numel() for b in model.buffers())


@pytest.mark.skipif(not REFERENCE.is_file(), reason="published reference checkpoint not present")
def test_published_checkpoint_loads_strictly():
    state = torch.load(REFERENCE, map_location="cpu", weights_only=True)
    model = MambaSCD()
    model.load_state_dict(state, strict=True)


@pytest.mark.skipif(not IMAGENET.is_file(), reason="VMamba ImageNet weights not present")
def test_imagenet_encoder_initialization_drops_only_declared_keys():
    model = MambaSCD()
    report = load_vmamba_encoder_weights(model, IMAGENET, expected_sha256=IMAGENET_SHA256)
    assert {k.split(".blocks.")[0] for k in report["dropped"] if k.startswith("layers.")} == {
        "layers.2"
    }
    assert all(k.startswith(("classifier.", "layers.2.blocks.4.")) for k in report["dropped"])
    assert report["new_layers"] and all(k.startswith("outnorm") for k in report["new_layers"])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_vmamba_encoder_weights(model, IMAGENET, expected_sha256="0" * 64)


def test_cuda_kernel_path_matches_reference_path():
    device_name = os.environ.get("SATCHANGE_TEST_CUDA_DEVICE")
    if not device_name or not torch.cuda.is_available():
        pytest.skip("Requires CUDA and explicit SATCHANGE_TEST_CUDA_DEVICE allocation.")
    if not selective_scan_kernel_available():
        pytest.fail("SATCHANGE_TEST_CUDA_DEVICE is set but the selective-scan kernel is missing")
    torch.manual_seed(0)
    reference = MambaSCD(**TINY, scan_backend="reference").eval()
    kernel = MambaSCD(**TINY, scan_backend="kernel").eval()
    kernel.load_state_dict(reference.state_dict())
    inputs = (torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    flags = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            expected = reference(*inputs)
            actual = kernel.to(device_name)(*(x.to(device_name) for x in inputs))
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = flags
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.cpu(), e, rtol=1e-4, atol=1e-4)
