"""SCanNet re-implementation (R2-V4-T03): shapes, stripe attention, published-weight loading."""

from pathlib import Path

import pytest
import torch

from satchange.models.scannet import SCanNet, StripeAttention, load_resnet34_encoder_weights
from satchange.semantic.compute import count_training_step_flops

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "artifacts/research2/v2/reference/scannet"
    / "SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth"
)
RESNET34 = ROOT / "artifacts/research2/v2/pretrained/torchvision/resnet34-b627a593.pth"
RESNET34_SHA256 = "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"


def test_stripe_attention_is_local_to_its_stripe():
    torch.manual_seed(0)
    attention = StripeAttention(dim=4, heads=2, stripe=2, direction=0).eval()
    height, width = 4, 8
    q, k, v = (torch.randn(1, height * width, 4) for _ in range(3))
    base = attention(q, k, v, height, width)
    changed = v.clone().view(1, height, width, 4)
    changed[:, :, 6:] += 10.0  # last vertical stripe (columns 6-7)
    out = attention(q, k, changed.view(1, -1, 4), height, width).view(1, height, width, 4)
    base = base.view(1, height, width, 4)
    assert torch.allclose(out[:, :, :5], base[:, :, :5])  # columns 0..4 unaffected (LePE reach 1)
    assert not torch.allclose(out[:, :, 6:], base[:, :, 6:])
    with pytest.raises(ValueError, match="divisible"):
        attention(q[:, :12], k[:, :12], v[:, :12], 4, 3)


def test_forward_shapes_flops_and_validation():
    torch.manual_seed(0)
    model = SCanNet()
    change, first, second = model.eval()(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    assert change.shape == (1, 1, 64, 64) and first.shape == second.shape == (1, 7, 64, 64)
    flops = count_training_step_flops(
        model, (torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)),
        lambda outputs: sum(o.float().mean() for o in outputs),
    )
    assert flops.forward_counted > 0 and flops.forward_analytic == 0
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.rand(1, 3, 40, 40), torch.rand(1, 3, 40, 40))
    with pytest.raises(ValueError, match="heads"):
        SCanNet(heads=3)


def test_parameter_count_matches_published_checkpoint():
    model = SCanNet()
    total = sum(t.numel() for t in model.state_dict().values())
    assert total == 27_919_752


@pytest.mark.skipif(not REFERENCE.is_file(), reason="published reference checkpoint not present")
def test_published_checkpoint_loads_strictly():
    SCanNet().load_state_dict(torch.load(REFERENCE, map_location="cpu", weights_only=True))


@pytest.mark.skipif(not RESNET34.is_file(), reason="torchvision ResNet-34 weights not present")
def test_imagenet_encoder_initialization():
    model = SCanNet()
    report = load_resnet34_encoder_weights(model, RESNET34, expected_sha256=RESNET34_SHA256)
    assert report["new_layers"] and all(k.startswith("head.") for k in report["new_layers"])
    assert model.FCN.layer3[0].conv1.stride == (1, 1)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_resnet34_encoder_weights(model, RESNET34, expected_sha256="0" * 64)
