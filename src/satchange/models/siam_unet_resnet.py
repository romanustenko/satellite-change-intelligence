"""Siamese U-Net with a shared ResNet-18 encoder for semantic change detection (R2-V4-T02).

Own implementation (no published reference model). The encoder (torchvision ResNet-18, ImageNet
initialization via :func:`load_resnet_encoder_weights`) is shared by both dates and yields five
feature levels (strides 2..32). U-Net decoder blocks: x2 bilinear upsampling, concatenation with
the skip feature, two 3x3 conv-BN-ReLU layers; a last block restores full resolution.

The head layout is an open protocol decision (Q-4 / D-049) and must be given explicitly:

* ``"multi_head"`` — a change decoder on the per-level concatenation ``[F1, F2]`` (1 sigmoid
  logit) and one semantic decoder shared by both dates (7 logits each), the same output contract
  as MambaSCD/SCanNet: ``(change [B,1,H,W], before [B,7,H,W], after [B,7,H,W])``.
* ``"joint"`` — one decoder on ``[F1, F2]`` producing the 14 T1+T2 logits ``[B,14,H,W]`` of the
  ``scd_training`` layout.
"""

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
import torchvision  # type: ignore[import-untyped]
from torch import nn

HeadLayout = Literal["multi_head", "joint"]
ENCODER_CHANNELS = (64, 64, 128, 256, 512)  # strides 2, 4, 8, 16, 32


class ResNetEncoder(nn.Module):
    """ResNet-18 feature pyramid ``[stem, layer1, layer2, layer3, layer4]``."""

    def __init__(self) -> None:
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.conv1, self.bn1, self.relu = resnet.conv1, resnet.bn1, resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stem = self.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(self.maxpool(stem))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [stem, f1, f2, f3, self.layer4(f3)]


def _conv_bn_relu(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]


class UNetDecoder(nn.Module):
    """Decoder over a five-level pyramid with ``widen`` x encoder channels per level."""

    def __init__(self, widen: int, channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        skips = [c * widen for c in ENCODER_CHANNELS]
        previous = skips[-1]
        self.blocks = nn.ModuleList()
        for skip, width in zip(reversed(skips[:-1]), channels[:-1], strict=True):
            self.blocks.append(nn.Sequential(*_conv_bn_relu(previous + skip, width),
                                             *_conv_bn_relu(width, width)))
            previous = width
        self.final = nn.Sequential(*_conv_bn_relu(previous, channels[-1]),
                                   *_conv_bn_relu(channels[-1], channels[-1]))
        self.classifier = nn.Conv2d(channels[-1], out_channels, 1)

    def forward(self, features: list[torch.Tensor], size: torch.Size) -> torch.Tensor:
        x = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1]), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return cast(torch.Tensor, self.classifier(self.final(x)))


class SiamUNetResNet18(nn.Module):
    """Siamese U-Net + ResNet-18; ``head_layout`` is required (no default, see module doc)."""

    def __init__(
        self,
        *,
        head_layout: HeadLayout,
        decoder_channels: list[int] | None = None,
        semantic_classes: int = 7,
    ) -> None:
        super().__init__()
        if head_layout not in ("multi_head", "joint"):
            raise ValueError(f"head_layout must be 'multi_head' or 'joint', got {head_layout!r}")
        channels = tuple(decoder_channels or (256, 128, 64, 32, 16))
        if len(channels) != 5 or any(type(c) is not int or c < 1 for c in channels):
            raise ValueError(f"decoder_channels must be five positive integers, got {channels}")
        if semantic_classes != 7:
            raise ValueError("semantic_classes must be 7 (no change + six SECOND classes).")
        self.head_layout = head_layout
        self.encoder = ResNetEncoder()
        if head_layout == "joint":
            self.joint_decoder = UNetDecoder(2, channels, 2 * semantic_classes)
        else:
            self.change_decoder = UNetDecoder(2, channels, 1)
            self.semantic_decoder = UNetDecoder(1, channels, semantic_classes)

    def forward(
        self, before: torch.Tensor, after: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if before.shape != after.shape or before.ndim != 4:
            raise ValueError("before and after must be [B,3,H,W] tensors of the same shape.")
        if before.shape[-2] % 32 or before.shape[-1] % 32:
            raise ValueError("SiamUNetResNet18 needs H and W divisible by 32.")
        first, second = self.encoder(before), self.encoder(after)
        pairs = [torch.cat([a, b], dim=1) for a, b in zip(first, second, strict=True)]
        size = before.shape[-2:]
        if self.head_layout == "joint":
            return cast(torch.Tensor, self.joint_decoder(pairs, size))
        return (
            self.change_decoder(pairs, size),
            self.semantic_decoder(first, size),
            self.semantic_decoder(second, size),
        )


def load_resnet_encoder_weights(
    model: SiamUNetResNet18, path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Initialize the shared encoder from torchvision ResNet-18 ImageNet weights (no ``fc``)."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {digest} != {expected_sha256}")
    source = {k: v for k, v in torch.load(path, map_location="cpu", weights_only=True).items()
              if not k.startswith("fc.")}
    target = model.encoder.state_dict()
    unexpected = sorted(set(source) - set(target))
    missing = sorted(k for k in set(target) - set(source) if not k.endswith("num_batches_tracked"))
    if unexpected or missing:
        raise ValueError(f"ResNet-18 weights do not match: unexpected {unexpected[:5]}, "
                         f"missing {missing[:5]}")
    model.encoder.load_state_dict(source, strict=False)
    return {"path": str(path), "sha256": digest, "loaded": len(source)}


__all__ = ["SiamUNetResNet18", "load_resnet_encoder_weights"]
