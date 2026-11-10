"""SCanNet (Ding et al., "Joint Spatio-Temporal Modeling for Semantic Change Detection in Remote
Sensing Images", IEEE TGRS 62, 2024; arXiv 2212.05245v4), re-implemented for R2-V4-T03.

Independent implementation from the paper (Sec. III-A/B/D): no code was copied from the
unlicensed author repository. Where the paper is silent, the structure follows its cited
components (Siamese ResNet-34 encoders with output stride 8 and a 1x1 128-channel head as in
SSCDl [8]; U-Net-style decoder blocks; CSWin cross-shaped stripe attention with LePE, Dong et al.,
CVPR 2022) and module names/shapes follow the published SECOND checkpoint so that it loads with
``strict=True`` (V4-T05 fidelity reference). Remaining unspecified choices were resolved
empirically against that checkpoint on our validation split (see ``docs/research2/PROGRESS.md``).

Paper settings: SCanFormer with L = 2 blocks, stripe width s = 2, 2K = 4 heads (K per stripe
direction), token depth d = 3 x 128 at H/4 x W/4. ``forward(before, after)`` returns
``(change [B,1,H,W] sigmoid logit, before_semantic [B,7,H,W], after_semantic [B,7,H,W])``.
"""

import hashlib
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
import torchvision  # type: ignore[import-untyped]
from torch import nn


class Rearrange(nn.Module):
    """``[B,C,H,W]`` → ``[B,H*W,C]`` (no parameters; keeps checkpoint indices)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)


class ResidualBlock(nn.Module):
    """ResNet basic block with optional 1x1 projection (torchvision naming)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return cast(torch.Tensor, self.relu(out + identity))


class SiameseEncoder(nn.Module):
    """ResNet-34, output stride 8 (layer3/4 strides set to 1), 1x1 head to 128 channels."""

    def __init__(self, head_channels: int = 128) -> None:
        super().__init__()
        resnet = torchvision.models.resnet34(weights=None)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        for layer in (self.layer3, self.layer4):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Conv2d) and (
                    name.endswith("conv1") or name.endswith("downsample.0")
                ):
                    module.stride = (1, 1)
        self.head = nn.Sequential(
            nn.Conv2d(512, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.layer1(self.maxpool(self.layer0(x)))  # [B,64,H/4,W/4]
        high = self.head(self.layer4(self.layer3(self.layer2(low))))  # [B,128,H/8,W/8]
        return high, low


class DecoderBlock(nn.Module):
    """x2 transposed-conv upsampling, 1x1 lateral transit to 64, 3x3 fusion to 128."""

    def __init__(self, channels: int, low_channels: int, transit_channels: int = 64) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.transit = nn.Sequential(
            nn.Conv2d(low_channels, transit_channels, 1, bias=False),
            nn.BatchNorm2d(transit_channels),
            nn.ReLU(inplace=True),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(channels + transit_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return cast(torch.Tensor, self.decode(torch.cat([x, self.transit(low)], dim=1)))


class StripeAttention(nn.Module):
    """CSWin LePE attention inside vertical (``direction=0``) or horizontal stripes of width s."""

    def __init__(self, dim: int, heads: int, stripe: int, direction: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads.")
        self.heads, self.stripe, self.direction = heads, stripe, direction
        self.scale = (dim // heads) ** -0.5
        self.get_v = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def _window(self, height: int, width: int) -> tuple[int, int]:
        return (height, self.stripe) if self.direction == 0 else (self.stripe, width)

    def _to_windows(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """``[B,N,C]`` → ``[B*windows, C, wh, ww]``."""
        batch, _, channels = x.shape
        wh, ww = self._window(height, width)
        x = x.transpose(1, 2).reshape(batch, channels, height // wh, wh, width // ww, ww)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(-1, channels, wh, ww)

    def _heads(self, windows: torch.Tensor) -> torch.Tensor:
        """``[B',C,wh,ww]`` → ``[B',heads,wh*ww,C/heads]``."""
        count, channels = windows.shape[:2]
        return windows.reshape(count, self.heads, channels // self.heads, -1).transpose(2, 3)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        batch, _, channels = q.shape
        wh, ww = self._window(height, width)
        if height % wh or width % ww:
            raise ValueError(f"Feature size {height}x{width} is not divisible by stripes.")
        v_windows = self._to_windows(v, height, width)
        lepe = self._heads(self.get_v(v_windows))
        q_h = self._heads(self._to_windows(q, height, width)) * self.scale
        k_h = self._heads(self._to_windows(k, height, width))
        v_h = self._heads(v_windows)
        attention = torch.softmax(q_h @ k_h.transpose(-2, -1), dim=-1)
        out = attention @ v_h + lepe  # [B',heads,wh*ww,C/heads]
        out = out.transpose(1, 2).reshape(-1, wh, ww, channels)
        out = out.view(batch, height // wh, width // ww, wh, ww, channels)
        return out.permute(0, 1, 3, 2, 4, 5).reshape(batch, height * width, channels)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc2(self.act(self.fc1(x))))


class SCanFormerBlock(nn.Module):
    """Pre-norm block (Eq. 2): cross-shaped stripe self-attention (Eq. 3-5) and MLP."""

    def __init__(self, dim: int, heads: int, stripe: int, mlp_ratio: float) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.norm1 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.attns = nn.ModuleList(
            [StripeAttention(dim // 2, heads // 2, stripe, direction) for direction in (0, 1)]
        )
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        channels = x.shape[-1]
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        half = channels // 2
        mixed = torch.cat([
            attention(q[..., i * half:(i + 1) * half], k[..., i * half:(i + 1) * half],
                      v[..., i * half:(i + 1) * half], height, width)
            for i, attention in enumerate(self.attns)
        ], dim=-1)
        x = x + self.proj(mixed)
        return x + cast(torch.Tensor, self.mlp(self.norm2(x)))


class SCanFormer(nn.Module):
    """Joint spatio-temporal transformer over concatenated ``[X1, X2, Xc]`` tokens."""

    def __init__(self, dim: int, depth: int, heads: int, stripe: int, mlp_ratio: float) -> None:
        super().__init__()
        self.stage1_conv_embed = nn.Sequential(nn.Conv2d(dim, dim, 1), Rearrange(),
                                               nn.LayerNorm(dim))
        self.stage = nn.ModuleList(
            [SCanFormerBlock(dim, heads, stripe, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = self.stage1_conv_embed(x)
        for block in self.stage:
            tokens = block(tokens, height, width)
        tokens = cast(torch.Tensor, self.norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class SCanNet(nn.Module):
    """TED (triple encoder-decoder) + SCanFormer head; defaults = the paper's SECOND model."""

    def __init__(
        self,
        *,
        semantic_classes: int = 7,
        channels: int = 128,
        transformer_depth: int = 2,
        heads: int = 4,
        stripe: int = 2,
        mlp_ratio: float = 4.0,
        change_blocks: int = 6,
    ) -> None:
        super().__init__()
        for name, value in (("channels", channels), ("transformer_depth", transformer_depth),
                            ("heads", heads), ("stripe", stripe), ("change_blocks", change_blocks)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if semantic_classes not in (6, 7):
            raise ValueError("semantic_classes must be 6 or 7.")
        if heads % 2 or (3 * channels // 2) % (heads // 2):
            raise ValueError("heads must be even and divide the per-direction channels.")
        self.stripe = stripe
        self.FCN = SiameseEncoder(channels)
        self.resCD = nn.Sequential(
            ResidualBlock(2 * channels, channels),
            *[ResidualBlock(channels, channels) for _ in range(change_blocks - 1)],
        )
        self.transformer = SCanFormer(3 * channels, transformer_depth, heads, stripe, mlp_ratio)
        self.DecCD = DecoderBlock(channels, 2 * 64)
        self.Dec1 = DecoderBlock(channels, 64)
        self.Dec2 = DecoderBlock(channels, 64)
        self.classifierA = nn.Conv2d(channels, semantic_classes, 1)
        self.classifierB = nn.Conv2d(channels, semantic_classes, 1)
        self.classifierCD = nn.Sequential(
            nn.Conv2d(channels, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(
        self, before: torch.Tensor, after: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if before.shape != after.shape or before.ndim != 4:
            raise ValueError("before and after must be [B,3,H,W] tensors of the same shape.")
        if before.shape[-2] % (8 * self.stripe) or before.shape[-1] % (8 * self.stripe):
            raise ValueError(f"SCanNet needs H and W divisible by {8 * self.stripe}.")
        x1, low1 = self.FCN(before)
        x2, low2 = self.FCN(after)
        xc = self.resCD(torch.cat([x1, x2], dim=1))
        x1 = self.Dec1(x1, low1)
        x2 = self.Dec2(x2, low2)
        xc = self.DecCD(xc, torch.cat([low1, low2], dim=1))
        fused = self.transformer(torch.cat([x1, x2, xc], dim=1))
        x1, x2, xc = fused.chunk(3, dim=1)
        size = before.shape[-2:]

        def up(x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(x, size=size, mode="bilinear")

        return up(self.classifierCD(xc)), up(self.classifierA(x1)), up(self.classifierB(x2))


def load_resnet34_encoder_weights(
    model: SCanNet, path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Initialize the Siamese ResNet-34 from torchvision ImageNet weights (``fc`` dropped)."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {digest} != {expected_sha256}")
    source = torch.load(path, map_location="cpu", weights_only=True)
    stem = {"conv1": "layer0.0", "bn1": "layer0.1"}
    renamed = {}
    for key, value in source.items():
        prefix, _, rest = key.partition(".")
        if prefix != "fc":
            renamed[f"{stem[prefix]}.{rest}" if prefix in stem else key] = value
    target = model.FCN.state_dict()
    unexpected = sorted(set(renamed) - set(target))
    missing = sorted(
        k for k in set(target) - set(renamed)
        if not k.startswith("head.") and not k.endswith("num_batches_tracked")
    )
    if unexpected or missing:
        raise ValueError(f"ResNet-34 weights do not match: unexpected {unexpected[:5]}, "
                         f"missing {missing[:5]}")
    model.FCN.load_state_dict(renamed, strict=False)
    return {"path": str(path), "sha256": digest, "loaded": len(renamed),
            "new_layers": sorted(k for k in target if k.startswith("head."))}
    # torchvision ImageNet files predate ``num_batches_tracked``; those counters stay at 0.


__all__ = ["SCanNet", "load_resnet34_encoder_weights"]
