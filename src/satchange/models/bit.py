"""Bitemporal Image Transformer for binary change detection."""

import torch
from torch import nn
from torch.nn import functional as F


def _conv3x3(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """Create the 3 x 3 convolution used by ResNet18 basic blocks."""

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class _BasicBlock(nn.Module):
    """ResNet18 basic residual block with optional downsampling and dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = _conv3x3(
            in_channels,
            out_channels,
            stride=stride,
            dilation=dilation,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(
            out_channels,
            out_channels,
            dilation=dilation,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply one residual block."""

        identity = inputs
        features = self.relu(self.batch_norm1(self.conv1(inputs)))
        features = self.batch_norm2(self.conv2(features))

        if self.downsample is not None:
            identity = self.downsample(identity)

        output: torch.Tensor = self.relu(features + identity)
        return output


class _ResNet18S4Backbone(nn.Module):
    """ResNet18 stem through a dilated layer3, omitting layer4."""

    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, dilation=2)

        self._initialize_weights()

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        return nn.Sequential(
            _BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                dilation=dilation,
            ),
            _BasicBlock(
                out_channels,
                out_channels,
                dilation=dilation,
            ),
        )

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_with_high_resolution(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return layer1 and dilated layer3 features."""

        features = self.stem(inputs)
        high_resolution_features = self.layer1(features)
        features = self.layer2(high_resolution_features)
        features = self.layer3(features)
        return high_resolution_features, features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the dilated layer3 feature map."""

        _, features = self.forward_with_high_resolution(inputs)
        return features


class _MultiHeadAttention(nn.Module):
    """Attention with an independent dimension for every head."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        self.query_projection = nn.Linear(embedding_dim, self.inner_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, self.inner_dim, bias=False)
        self.value_projection = nn.Linear(embedding_dim, self.inner_dim, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(self.inner_dim, embedding_dim)
        self.output_dropout = nn.Dropout(dropout)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = values.shape
        return values.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from queries to context and return the embedding dimension."""

        query_heads = self._split_heads(self.query_projection(queries))
        key_heads = self._split_heads(self.key_projection(context))
        value_heads = self._split_heads(self.value_projection(context))

        attention = torch.softmax(
            torch.matmul(query_heads, key_heads.transpose(-2, -1)) * self.scale,
            dim=-1,
        )
        attended = torch.matmul(self.attention_dropout(attention), value_heads)
        attended = attended.transpose(1, 2).contiguous().flatten(2)
        output: torch.Tensor = self.output_dropout(self.output_projection(attended))
        return output


def _feed_forward(embedding_dim: int, dropout: float) -> nn.Sequential:
    """Create the pre-normalized Transformer feed-forward branch."""

    return nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(embedding_dim * 2, embedding_dim),
        nn.Dropout(dropout),
    )


class _TransformerEncoderBlock(nn.Module):
    """Pre-normalized residual Transformer encoder block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = _MultiHeadAttention(
            embedding_dim,
            num_heads,
            head_dim,
            dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = _feed_forward(embedding_dim, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and a feed-forward residual update."""

        normalized_tokens = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized_tokens, normalized_tokens)
        feed_forward_output: torch.Tensor = self.feed_forward(
            self.feed_forward_norm(tokens)
        )
        return tokens + feed_forward_output


class _CrossAttentionDecoderBlock(nn.Module):
    """Refine spatial queries with pre-normalized token cross-attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.query_norm = nn.LayerNorm(embedding_dim)
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = _MultiHeadAttention(
            embedding_dim,
            num_heads,
            head_dim,
            dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = _feed_forward(embedding_dim, dropout)

    def forward(
        self,
        spatial_queries: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from every spatial feature to the contextual tokens."""

        attended_features = self.cross_attention(
            self.query_norm(spatial_queries),
            self.token_norm(tokens),
        )
        spatial_queries = spatial_queries + attended_features
        feed_forward_output: torch.Tensor = self.feed_forward(
            self.feed_forward_norm(spatial_queries)
        )
        return spatial_queries + feed_forward_output


class BIT(nn.Module):
    """Predict binary change logits with a shared BIT-S4-DD8 network."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        embedding_dim: int = 32,
        token_length: int = 4,
        num_heads: int = 8,
        encoder_head_dim: int = 64,
        decoder_head_dim: int = 64,
        encoder_depth: int = 1,
        decoder_depth: int = 8,
        dropout: float = 0.0,
        pretrained_backbone: bool = False,
    ) -> None:
        """Initialize the shared S4 backbone and token Transformer."""

        super().__init__()

        positive_integer_parameters = {
            "embedding_dim": embedding_dim,
            "token_length": token_length,
            "num_heads": num_heads,
            "encoder_head_dim": encoder_head_dim,
            "decoder_head_dim": decoder_head_dim,
            "encoder_depth": encoder_depth,
            "decoder_depth": decoder_depth,
        }
        for name, value in positive_integer_parameters.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels != 3:
            raise ValueError("BIT requires in_channels=3 for its ResNet18-compatible backbone.")
        if (
            not isinstance(out_channels, int)
            or isinstance(out_channels, bool)
            or out_channels != 1
        ):
            raise ValueError("BIT supports binary change detection and requires out_channels=1.")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive).")
        if not isinstance(pretrained_backbone, bool):
            raise ValueError("pretrained_backbone must be a boolean.")
        if pretrained_backbone:
            raise ValueError(
                "pretrained_backbone=True is unsupported until the input-normalization "
                "pipeline matches the pretrained ResNet18 contract."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.token_length = token_length

        self.backbone = _ResNet18S4Backbone()
        self.feature_projection = nn.Conv2d(
            in_channels=256,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.token_attention = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=token_length,
            kernel_size=1,
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, 2 * token_length, embedding_dim)
        )
        nn.init.normal_(self.positional_embedding, std=0.02)

        self.encoder_layers = nn.ModuleList(
            [
                _TransformerEncoderBlock(
                    embedding_dim,
                    num_heads,
                    encoder_head_dim,
                    dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                _CrossAttentionDecoderBlock(
                    embedding_dim,
                    num_heads,
                    decoder_head_dim,
                    dropout,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.prediction_head = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=embedding_dim,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def _validate_inputs(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> None:
        """Validate the public two-image model contract."""

        if before.ndim != 4 or after.ndim != 4:
            raise ValueError("Before and after tensors must both be 4D [B, C, H, W].")
        if before.shape != after.shape:
            raise ValueError(
                "Before and after tensors must have identical shapes. "
                f"Received {before.shape} and {after.shape}."
            )
        if before.shape[1] != self.in_channels:
            raise ValueError(
                f"BIT expects {self.in_channels} input channels, received {before.shape[1]}."
            )

        height, width = before.shape[-2:]
        if height < 32 or width < 32 or height % 32 != 0 or width % 32 != 0:
            raise ValueError(
                "BIT spatial dimensions must each be at least 32 and divisible by 32. "
                f"Received {(height, width)}."
            )

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        """Map one image to projected features at one quarter resolution."""

        high_resolution_features, features = self.backbone.forward_with_high_resolution(image)
        features = F.interpolate(
            features,
            size=high_resolution_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        projected_features: torch.Tensor = self.feature_projection(features)
        return projected_features

    def _tokenize(self, features: torch.Tensor) -> torch.Tensor:
        """Compress a spatial feature map into learned semantic tokens."""

        attention = torch.softmax(self.token_attention(features).flatten(2), dim=-1)
        return torch.einsum("bln,bdn->bld", attention, features.flatten(2))

    def _encode_tokens(
        self,
        before_tokens: torch.Tensor,
        after_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Jointly contextualize and then split the two temporal token sets."""

        contextual_tokens = torch.cat((before_tokens, after_tokens), dim=1)
        contextual_tokens = contextual_tokens + self.positional_embedding
        for encoder_layer in self.encoder_layers:
            contextual_tokens = encoder_layer(contextual_tokens)

        before_context, after_context = contextual_tokens.split(self.token_length, dim=1)
        return before_context, after_context

    def _decode_features(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Use shared decoder weights to refine one temporal feature map."""

        batch_size, _, height, width = features.shape
        spatial_queries = features.flatten(2).transpose(1, 2)
        for decoder_layer in self.decoder_layers:
            spatial_queries = decoder_layer(spatial_queries, tokens)

        return spatial_queries.transpose(1, 2).reshape(
            batch_size,
            self.embedding_dim,
            height,
            width,
        )

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw binary change logits with the input spatial shape."""

        self._validate_inputs(before, after)

        before_features = self._encode(before)
        after_features = self._encode(after)
        before_tokens = self._tokenize(before_features)
        after_tokens = self._tokenize(after_features)
        before_tokens, after_tokens = self._encode_tokens(before_tokens, after_tokens)
        before_refined = self._decode_features(before_features, before_tokens)
        after_refined = self._decode_features(after_features, after_tokens)

        difference = F.interpolate(
            torch.abs(before_refined - after_refined),
            size=before.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits: torch.Tensor = self.prediction_head(difference)
        return logits
