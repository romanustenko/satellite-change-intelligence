"""MambaSCD-Tiny (ChangeMamba semantic change detection), ported for Research 2 v2 (R2-V4-T04).

Ported from ChangeMamba (https://github.com/ChenHongruixuan/MambaCD), architecture as of commit
``b834b2a`` (last revision before the 2026-03 refactor, i.e. the code that produced the published
SECOND checkpoint): ``classification/models/vmamba.py`` (VSSM, VSSBlock, SS2D ``v3noz`` path),
``changedetection/models/{Mamba_backbone,ChangeDecoder,SemanticDecoder,ChangeMambaSCD}.py``.
Copyright the ChangeMamba / VMamba authors; licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes: only the configuration used by
MambaSCD-Tiny is kept (``vssm_tiny_224_0229flex``: channel-last LayerNorm, patch embed v2,
downsample v3, forward type ``v3noz``, PyTorch cross-scan as for ``vssm1_0230`` and earlier);
the scan uses ``satchange.models.selective_scan`` (compiled kernel on CUDA, reference on CPU);
cross-scan/merge use differentiable tensor ops instead of hand-written autograd functions; the
decoders' CUDA-only ``torch.empty(...).cuda()`` buffers are device-agnostic; stochastic depth is
re-implemented (timm semantics); module and parameter names are unchanged so the published
checkpoint loads with ``strict=True``. Encoder initialization from the VMamba-Tiny ImageNet file
follows D-052 (explicit list of dropped keys instead of a silent non-strict load).

``forward(before, after)`` returns ``(change [B,2,H,W], before_semantic [B,7,H,W],
after_semantic [B,7,H,W])`` logits; ``semantic.scd_outputs`` converts them to the common maps.
"""

import hashlib
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from satchange.models.selective_scan import Backend, selective_scan


class DropPath(nn.Module):
    """Per-sample stochastic depth (timm ``DropPath`` semantics, scale by keep probability)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Permute(nn.Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc2(self.act(self.fc1(x))))


def cross_scan(x: torch.Tensor) -> torch.Tensor:
    """``[B,C,H,W]`` → ``[B,4,C,H*W]``: row-major, column-major and both reversed."""
    rows = x.flatten(2, 3)
    columns = x.transpose(2, 3).flatten(2, 3)
    forward = torch.stack([rows, columns], dim=1)
    return torch.cat([forward, forward.flip(-1)], dim=1)


def cross_merge(ys: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse-order sum of the four directions: ``[B,4,C,L]`` → ``[B,C,L]`` (row-major)."""
    batch, _, channels, length = ys.shape
    pair = ys[:, 0:2] + ys[:, 2:4].flip(-1)
    columns = pair[:, 1].view(batch, channels, width, height).transpose(2, 3)
    return pair[:, 0] + columns.reshape(batch, channels, length)


class SS2D(nn.Module):
    """2D selective-scan block, VMamba ``forwardv2`` with forward type ``v3noz`` (channel-last)."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        ssm_ratio: float,
        d_conv: int,
        scan_backend: Backend,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16)
        groups = 4
        self.scan_backend = scan_backend
        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(
            d_inner, d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2,
            groups=d_inner, bias=False,
        )
        x_proj = [nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False) for _ in range(groups)]
        self.x_proj_weight = nn.Parameter(torch.stack([p.weight for p in x_proj]))
        std = dt_rank**-0.5
        weight = torch.empty(groups, d_inner, dt_rank).uniform_(-std, std)
        self.dt_projs_weight = nn.Parameter(weight)
        dt = torch.exp(
            torch.rand(groups, d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_projs_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus inverse
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(groups * d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A))
        self.Ds = nn.Parameter(torch.ones(groups * d_inner))
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dt_rank, self.d_state = dt_rank, d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,H,W,C]
        x = self.in_proj(x).permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        batch, channels, height, width = x.shape
        length = height * width
        xs = cross_scan(x)  # [B,4,D,L]
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        ys = selective_scan(
            xs.reshape(batch, -1, length),
            dts.contiguous().view(batch, -1, length),
            -torch.exp(self.A_logs.float()),
            Bs.contiguous(),
            Cs.contiguous(),
            self.Ds.float(),
            self.dt_projs_bias.view(-1).float(),
            delta_softplus=True,
            backend=self.scan_backend,
        ).view(batch, 4, channels, length)
        y = cross_merge(ys, height, width).transpose(1, 2).contiguous()  # [B,L,D]
        y = self.out_norm(y).view(batch, height, width, -1).to(x.dtype)
        return cast(torch.Tensor, self.out_proj(y))


class VSSBlock(nn.Module):
    """Pre-norm SS2D + MLP residual block on channel-last tensors."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        drop_path: float,
        d_state: int,
        ssm_ratio: float,
        d_conv: int,
        mlp_ratio: float,
        scan_backend: Backend,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.op = SS2D(hidden_dim, d_state=d_state, ssm_ratio=ssm_ratio, d_conv=d_conv,
                       scan_backend=scan_backend)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = Mlp(hidden_dim, int(hidden_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.op(self.norm(x)))
        return cast(torch.Tensor, x + self.drop_path(self.mlp(self.norm2(x))))


class VSSMEncoder(nn.Module):
    """VMamba backbone (patch embed v2, downsample v3); four ``[B,C,H,W]`` feature maps."""

    def __init__(
        self,
        *,
        depths: list[int],
        embed_dim: int,
        drop_path_rate: float,
        block_options: dict[str, Any],
    ) -> None:
        super().__init__()
        self.dims = [embed_dim * 2**i for i in range(len(depths))]
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(embed_dim // 2),
            Permute(0, 3, 1, 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(embed_dim),
        )
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        for index, depth in enumerate(depths):
            start = sum(depths[:index])
            blocks = nn.Sequential(*[
                VSSBlock(self.dims[index], drop_path=rates[start + d], **block_options)
                for d in range(depth)
            ])
            downsample: nn.Module = nn.Identity()
            if index < len(depths) - 1:
                downsample = nn.Sequential(
                    Permute(0, 3, 1, 2),
                    nn.Conv2d(self.dims[index], self.dims[index + 1], 3, stride=2, padding=1),
                    Permute(0, 2, 3, 1),
                    nn.LayerNorm(self.dims[index + 1]),
                )
            self.layers.append(nn.Sequential(OrderedDict(blocks=blocks, downsample=downsample)))
        for index, dim in enumerate(self.dims):
            self.add_module(f"outnorm{index}", nn.LayerNorm(dim))
        self.apply(_init_vssm_weights)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        features = []
        for index, layer in enumerate(self.layers):
            stage = cast(nn.Sequential, layer)  # OrderedDict(blocks=..., downsample=...)
            out = stage[0](x)
            x = stage[1](out)
            norm = getattr(self, f"outnorm{index}")
            features.append(norm(out).permute(0, 3, 1, 2).contiguous())
        return features


def _init_vssm_weights(module: nn.Module) -> None:
    """VSSM ``_init_weights``: truncated-normal Linear (std 0.02), unit LayerNorm."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return cast(torch.Tensor, self.relu(out + x))


def _vss_stage(in_channels: int | None, hidden: int, block: dict[str, Any]) -> nn.Sequential:
    """``[conv1x1], Permute, VSSBlock, Permute`` (published module indices)."""
    layers: list[nn.Module] = []
    if in_channels is not None:
        layers.append(nn.Conv2d(in_channels, hidden, kernel_size=1))
    layers += [Permute(0, 2, 3, 1), VSSBlock(hidden, **block), Permute(0, 3, 1, 2)]
    return nn.Sequential(*layers)


def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def _upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=y.shape[-2:], mode="bilinear") + y


class ChangeDecoder(nn.Module):
    """Spatio-temporal change decoder: concat / interleave / side-by-side VSS mixing per stage."""

    def __init__(self, dims: list[int], hidden: int, block: dict[str, Any]) -> None:
        super().__init__()
        for stage, dim in zip((1, 2, 3, 4), dims, strict=True):
            self.add_module(f"st_block_{stage}1", _vss_stage(2 * dim, hidden, block))
            self.add_module(f"st_block_{stage}2", _vss_stage(dim, hidden, block))
            self.add_module(f"st_block_{stage}3", _vss_stage(dim, hidden, block))
            self.add_module(f"fuse_layer_{stage}", _conv_bn_relu(5 * hidden, hidden))
        for stage in (1, 2, 3):
            self.add_module(f"smooth_layer_{stage}", ResBlock(hidden))

    def _stage(self, stage: int, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        width = pre.shape[-1]
        p1 = getattr(self, f"st_block_{stage}1")(torch.cat([pre, post], dim=1))
        interleaved = torch.stack([pre, post], dim=-1).flatten(-2)  # columns pre0, post0, ...
        p2 = getattr(self, f"st_block_{stage}2")(interleaved)
        p3 = getattr(self, f"st_block_{stage}3")(torch.cat([pre, post], dim=-1))
        mixed = [p1, p2[..., ::2], p2[..., 1::2], p3[..., :width], p3[..., width:]]
        fused: torch.Tensor = getattr(self, f"fuse_layer_{stage}")(torch.cat(mixed, dim=1))
        return fused

    def forward(self, pre: list[torch.Tensor], post: list[torch.Tensor]) -> torch.Tensor:
        current = self._stage(4, pre[3], post[3])
        for stage in (3, 2, 1):
            fused = self._stage(stage, pre[stage - 1], post[stage - 1])
            current = getattr(self, f"smooth_layer_{stage}")(_upsample_add(current, fused))
        return current


class SemanticDecoder(nn.Module):
    """Single-date semantic decoder (FPN-style with VSS blocks)."""

    def __init__(self, dims: list[int], hidden: int, block: dict[str, Any]) -> None:
        super().__init__()
        self.st_block_4_semantic = _vss_stage(dims[3], hidden, block)
        for stage in (3, 2, 1):
            self.add_module(f"st_block_{stage}_semantic", _vss_stage(None, hidden, block))
            self.add_module(f"trans_layer_{stage}", _conv_bn_relu(dims[stage - 1], hidden))
        for stage in (3, 2, 1, 0):
            self.add_module(f"smooth_layer_{stage}_semantic", ResBlock(hidden))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        current = self.st_block_4_semantic(features[3])
        for stage in (3, 2, 1):
            lateral = getattr(self, f"trans_layer_{stage}")(features[stage - 1])
            current = _upsample_add(current, lateral)
            current = getattr(self, f"smooth_layer_{stage}_semantic")(current)
            current = getattr(self, f"st_block_{stage}_semantic")(current)
        final = cast(nn.Module, self.smooth_layer_0_semantic)
        return cast(torch.Tensor, final(current))


class MambaSCD(nn.Module):
    """ChangeMamba MambaSCD; defaults = MambaSCD-Tiny (``vssm_tiny_224_0229flex``)."""

    def __init__(
        self,
        *,
        depths: list[int] | None = None,
        embed_dim: int = 96,
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        d_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        decoder_drop_path: float = 0.1,
        hidden_dim: int = 128,
        change_classes: int = 2,
        semantic_classes: int = 7,
        scan_backend: Backend = "auto",
    ) -> None:
        super().__init__()
        depths = [2, 2, 4, 2] if depths is None else list(depths)
        if len(depths) != 4 or any(type(d) is not int or d < 1 for d in depths):
            raise ValueError(f"depths must be four positive integers, got {depths}")
        for name, value in (("embed_dim", embed_dim), ("d_state", d_state), ("d_conv", d_conv),
                            ("hidden_dim", hidden_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if embed_dim % 2:
            raise ValueError("embed_dim must be even (patch embed halves it).")
        if change_classes not in (1, 2) or semantic_classes not in (6, 7):
            raise ValueError("change_classes must be 1 or 2 and semantic_classes 6 or 7.")
        if not (ssm_ratio > 0 and mlp_ratio > 0):
            raise ValueError("ssm_ratio and mlp_ratio must be positive.")
        if scan_backend not in ("auto", "kernel", "reference"):
            raise ValueError(f"Unknown scan_backend {scan_backend!r}")
        options = {"d_state": d_state, "ssm_ratio": ssm_ratio, "d_conv": d_conv,
                   "mlp_ratio": mlp_ratio, "scan_backend": scan_backend}
        self.encoder = VSSMEncoder(depths=depths, embed_dim=embed_dim,
                                   drop_path_rate=drop_path_rate, block_options=options)
        decoder_block = {**options, "drop_path": decoder_drop_path}
        dims = self.encoder.dims
        self.decoder_bcd = ChangeDecoder(dims, hidden_dim, decoder_block)
        self.decoder_T1 = SemanticDecoder(dims, hidden_dim, decoder_block)
        self.decoder_T2 = SemanticDecoder(dims, hidden_dim, decoder_block)
        self.main_clf_cd = nn.Conv2d(hidden_dim, change_classes, kernel_size=1)
        self.aux_clf = nn.Conv2d(hidden_dim, semantic_classes, kernel_size=1)

    def forward(
        self, before: torch.Tensor, after: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if before.shape != after.shape or before.ndim != 4:
            raise ValueError("before and after must be [B,3,H,W] tensors of the same shape.")
        if before.shape[-1] % 32 or before.shape[-2] % 32:
            raise ValueError("MambaSCD needs H and W divisible by 32.")
        pre, post = self.encoder(before), self.encoder(after)
        size = before.shape[-2:]

        def head(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(conv(x), size=size, mode="bilinear")

        return (
            head(self.main_clf_cd, self.decoder_bcd(pre, post)),
            head(self.aux_clf, self.decoder_T1(pre)),
            head(self.aux_clf, self.decoder_T2(post)),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vmamba_encoder_weights(
    model: MambaSCD, path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Initialize ``model.encoder`` from a VMamba ImageNet file (D-052); returns a load report.

    Keys absent from the encoder are allowed only for the ImageNet classifier and for blocks
    beyond the configured stage depths; every encoder parameter except the new ``outnorm*``
    layers must be present. Anything else fails.
    """
    path = Path(path)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    target = model.encoder.state_dict()
    dropped, mismatched = [], []
    for key in source:
        if key in target:
            if tuple(source[key].shape) != tuple(target[key].shape):
                mismatched.append(key)
            continue
        block = re.match(r"layers\.(\d+)\.blocks\.(\d+)\.", key)
        is_extra_block = False
        if block is not None and int(block.group(1)) < len(model.encoder.layers):
            stage = cast(nn.Sequential, model.encoder.layers[int(block.group(1))])
            is_extra_block = int(block.group(2)) >= len(cast(nn.Sequential, stage.blocks))
        if key.startswith("classifier.") or is_extra_block:
            dropped.append(key)
        else:
            mismatched.append(key)
    missing = [k for k in target if k not in source and not k.startswith("outnorm")]
    if mismatched or missing:
        raise ValueError(
            f"VMamba encoder weights do not match: unexpected/mismatched {mismatched[:5]}, "
            f"missing {missing[:5]}"
        )
    model.encoder.load_state_dict({k: v for k, v in source.items() if k in target}, strict=False)
    return {
        "path": str(path),
        "sha256": actual,
        "loaded": sum(1 for k in source if k in target),
        "dropped": sorted(dropped),
        "new_layers": sorted(k for k in target if k not in source),
    }


__all__ = ["MambaSCD", "cross_merge", "cross_scan", "load_vmamba_encoder_weights"]
