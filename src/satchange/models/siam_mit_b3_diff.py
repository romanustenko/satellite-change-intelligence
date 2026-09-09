"""Shared MiT-B3 encoder with project-specific absolute-difference decoding.

Independently implemented from the specified MiT operations; see docs/heavy_model.md.
"""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DropPath(nn.Module):
    """Drop a residual branch independently per sample, with expectation scaling."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return inputs
        keep = 1 - self.probability
        mask = inputs.new_empty((inputs.shape[0],) + (1,) * (inputs.ndim - 1))
        return inputs * mask.bernoulli_(keep) / keep


class SpatialReductionAttention(nn.Module):
    """Full-grid queries attend to learned spatially reduced keys and values."""

    def __init__(
        self, channels: int, heads: int, sr_ratio: int, dropout: float, attention_dropout: float
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, 2 * channels)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(channels, eps=1e-5)
        self.projection = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = attention_dropout

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, channels = tokens.shape
        query = self.q(tokens).reshape(batch, length, self.heads, self.head_dim)
        query = query.transpose(1, 2)
        source = tokens
        if self.sr_ratio > 1:
            source = tokens.transpose(1, 2).reshape(batch, channels, height, width)
            source = self.sr_norm(self.sr(source).flatten(2).transpose(1, 2))
        kv = self.kv(source).reshape(batch, -1, 2, self.heads, self.head_dim)
        key, value = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, channels)
        output: torch.Tensor = self.dropout(self.projection(attended))
        return output


class MixFFN(nn.Module):
    """Expand tokens by four and mix spatial neighbors with a depthwise convolution."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels * 4)
        self.depthwise = nn.Conv2d(channels * 4, channels * 4, 3, padding=1, groups=channels * 4)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(channels * 4, channels)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        spatial = self.fc1(tokens).transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
        tokens = self.depthwise(spatial).flatten(2).transpose(1, 2)
        output: torch.Tensor = self.dropout(self.fc2(self.dropout(self.activation(tokens))))
        return output


class MiTBlock(nn.Module):
    """Two pre-normalized attention/Mix-FFN residual branches."""

    def __init__(
        self,
        channels: int,
        heads: int,
        sr_ratio: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attention = SpatialReductionAttention(
            channels, heads, sr_ratio, dropout, attention_dropout
        )
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.ffn = MixFFN(channels, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        tokens = tokens + self.drop_path(self.attention(self.norm1(tokens), height, width))
        output: torch.Tensor = tokens + self.drop_path(self.ffn(self.norm2(tokens), height, width))
        return output


class MiTEncoder(nn.Module):
    """Four real hierarchical stages, returning normalized spatial feature maps."""

    channels = (64, 128, 320, 512)
    heads = (1, 2, 5, 8)
    reductions = (8, 4, 2, 1)

    def __init__(
        self,
        depths: tuple[int, ...],
        dropout: float,
        attention_dropout: float,
        drop_path_rate: float,
        use_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.depths = depths
        self.use_checkpointing = use_checkpointing
        self.embeddings = nn.ModuleList()
        self.embedding_norms = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.output_norms = nn.ModuleList()
        probabilities = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        for index, (channels, depth, heads, reduction) in enumerate(
            zip(self.channels, depths, self.heads, self.reductions, strict=True)
        ):
            self.embeddings.append(
                nn.Conv2d(
                    3 if index == 0 else self.channels[index - 1],
                    channels,
                    kernel_size=7 if index == 0 else 3,
                    stride=4 if index == 0 else 2,
                    padding=3 if index == 0 else 1,
                )
            )
            self.embedding_norms.append(nn.LayerNorm(channels, eps=1e-5))
            self.stages.append(
                nn.ModuleList(
                    [
                        MiTBlock(
                            channels,
                            heads,
                            reduction,
                            dropout,
                            attention_dropout,
                            probabilities[offset + block],
                        )
                        for block in range(depth)
                    ]
                )
            )
            self.output_norms.append(nn.LayerNorm(channels, eps=1e-6))
            offset += depth

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for embedding, embedding_norm, blocks, output_norm in zip(
            self.embeddings, self.embedding_norms, self.stages, self.output_norms, strict=True
        ):
            spatial = embedding(inputs)
            batch, channels, height, width = spatial.shape
            tokens = embedding_norm(spatial.flatten(2).transpose(1, 2))
            assert isinstance(blocks, nn.ModuleList)
            for block in blocks:
                if self.use_checkpointing and self.training and torch.is_grad_enabled():
                    tokens = checkpoint(
                        block, tokens, height, width, use_reentrant=False, preserve_rng_state=True
                    )
                else:
                    tokens = block(tokens, height, width)
            inputs = output_norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)
            features.append(inputs)
        return features


def _conv_block(in_channels: int, out_channels: int, groups: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


def _positive_sequence(name: str, values: Sequence[int], length: int) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain {length} positive integers.")
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} must contain {length} positive integers.")
    return tuple(values)


class SiamMiTB3Diff(nn.Module):
    """RGB binary change detector; non-default depths are explicit depth ablations."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_depths: Sequence[int] = (3, 4, 18, 3),
        decoder_channels: int = 128,
        refinement_channels: Sequence[int] = (64, 32),
        norm_groups: int = 8,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pretrained_backbone: bool = False,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if type(in_channels) is not int or in_channels != 3:
            raise ValueError("SiamMiTB3Diff requires in_channels=3.")
        if type(out_channels) is not int or out_channels != 1:
            raise ValueError("SiamMiTB3Diff requires out_channels=1.")
        depths = _positive_sequence("encoder_depths", encoder_depths, 4)
        refinements = _positive_sequence("refinement_channels", refinement_channels, 2)
        for name, value in (("decoder_channels", decoder_channels), ("norm_groups", norm_groups)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if any(width % norm_groups for width in (decoder_channels, *refinements)):
            raise ValueError("norm_groups must divide all decoder and refinement widths.")
        for name, probability in (
            ("dropout", dropout),
            ("attention_dropout", attention_dropout),
            ("drop_path_rate", drop_path_rate),
        ):
            if not isinstance(probability, (int, float)) or not 0 <= probability < 1:
                raise ValueError(f"{name} must be in [0, 1).")
        if type(use_checkpointing) is not bool or type(pretrained_backbone) is not bool:
            raise ValueError("use_checkpointing and pretrained_backbone must be booleans.")
        if pretrained_backbone:
            raise NotImplementedError(
                "External pretraining requires explicit compatible checkpoint provenance and "
                "input normalization in a separate experiment. Use pretrained_backbone=False."
            )
        self.encoder = MiTEncoder(
            depths, dropout, attention_dropout, drop_path_rate, use_checkpointing
        )
        self.projections = nn.ModuleList(
            [
                _conv_block(channels, decoder_channels, norm_groups, kernel=1)
                for channels in self.encoder.channels
            ]
        )
        self.coarsest = _conv_block(decoder_channels, decoder_channels, norm_groups)
        self.fusions = nn.ModuleList(
            [
                nn.Sequential(
                    _conv_block(2 * decoder_channels, decoder_channels, norm_groups),
                    _conv_block(decoder_channels, decoder_channels, norm_groups),
                )
                for _ in range(3)
            ]
        )
        self.refinements = nn.ModuleList(
            [
                _conv_block(decoder_channels, refinements[0], norm_groups),
                _conv_block(refinements[0], refinements[1], norm_groups),
            ]
        )
        self.head = nn.Conv2d(refinements[1], 1, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = math.prod(module.kernel_size) * module.out_channels / module.groups
            nn.init.normal_(module.weight, std=math.sqrt(2 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Return [B,1,H,W] raw logits for aligned floating RGB pairs divisible by 32."""
        if before.ndim != 4 or after.ndim != 4 or before.shape != after.shape:
            raise ValueError("Inputs must be four-dimensional tensors with identical shapes.")
        if not before.is_floating_point() or not after.is_floating_point():
            raise ValueError("Inputs must be floating-point tensors.")
        if before.dtype != after.dtype or before.device != after.device:
            raise ValueError("Inputs must have identical dtype and device.")
        batch, channels, height, width = before.shape
        if batch <= 0 or channels != 3 or min(height, width) < 32 or height % 32 or width % 32:
            raise ValueError("Require positive batch, RGB channels, and H/W >=32 divisible by 32.")
        before_features = self.encoder(before)
        after_features = self.encoder(after)
        projected = [
            projection(torch.abs(a - b))
            for projection, a, b in zip(
                self.projections, before_features, after_features, strict=True
            )
        ]
        current = self.coarsest(projected[-1])
        for fusion, skip in zip(self.fusions, reversed(projected[:-1]), strict=True):
            current = F.interpolate(
                current, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            current = fusion(torch.cat((current, skip), dim=1))
        for refinement, size in zip(
            self.refinements, ((height // 2, width // 2), (height, width)), strict=True
        ):
            current = refinement(
                F.interpolate(current, size=size, mode="bilinear", align_corners=False)
            )
        output: torch.Tensor = self.head(current)
        return output
