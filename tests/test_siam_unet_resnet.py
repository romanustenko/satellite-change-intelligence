"""Siamese U-Net + ResNet-18 (R2-V4-T02): both head layouts, Siamese weight sharing, init."""

from pathlib import Path

import pytest
import torch

from satchange.models.siam_unet_resnet import SiamUNetResNet18, load_resnet_encoder_weights
from satchange.semantic.compute import count_training_step_flops
from satchange.semantic.scd_outputs import maps_from_change_and_semantic
from satchange.semantic.scd_training import predict_maps

ROOT = Path(__file__).resolve().parents[1]
RESNET18 = ROOT / "artifacts/research2/v2/pretrained/torchvision/resnet18-f37072fd.pth"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def test_multi_head_layout_outputs_and_date_symmetry():
    torch.manual_seed(0)
    model = SiamUNetResNet18(head_layout="multi_head").eval()
    a, b = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    change, first, second = model(a, b)
    assert change.shape == (1, 1, 64, 64) and first.shape == second.shape == (1, 7, 64, 64)
    _, swapped_first, swapped_second = model(b, a)
    torch.testing.assert_close(swapped_first, second)  # one semantic decoder for both dates
    torch.testing.assert_close(swapped_second, first)
    maps = maps_from_change_and_semantic(change, first, second, semantic_argmax="land_cover")
    assert maps[0].shape == (1, 64, 64)


def test_joint_layout_matches_the_engine_contract_and_flops():
    model = SiamUNetResNet18(head_layout="joint")
    logits = model.eval()(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    assert isinstance(logits, torch.Tensor) and logits.shape == (1, 14, 64, 64)
    assert predict_maps(logits)[0].shape == (1, 64, 64)
    flops = count_training_step_flops(model, (torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)),
                                      lambda out: out.mean())
    assert flops.forward_counted > 0 and flops.forward_analytic == 0


def test_validation():
    with pytest.raises(TypeError):
        SiamUNetResNet18()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="head_layout"):
        SiamUNetResNet18(head_layout="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decoder_channels"):
        SiamUNetResNet18(head_layout="joint", decoder_channels=[8, 8])
    with pytest.raises(ValueError, match="divisible by 32"):
        SiamUNetResNet18(head_layout="joint")(torch.rand(1, 3, 48, 48), torch.rand(1, 3, 48, 48))


@pytest.mark.skipif(not RESNET18.is_file(), reason="torchvision ResNet-18 weights not present")
def test_imagenet_encoder_initialization():
    model = SiamUNetResNet18(head_layout="multi_head")
    report = load_resnet_encoder_weights(model, RESNET18, expected_sha256=RESNET18_SHA256)
    assert report["loaded"] == 100
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_resnet_encoder_weights(model, RESNET18, expected_sha256="0" * 64)
