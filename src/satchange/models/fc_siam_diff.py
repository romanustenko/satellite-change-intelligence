"""FC-Siam-Diff baseline model for binary change detection."""


import torch
from torch import nn


class FCSiamDiff(nn.Module):
    """Initial Siamese feature-difference model."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        """Initialize the model."""

        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        self.encoder = nn.Sequential(
            # First convolution creates base_channels feature maps
            nn.Conv2d(
                # nn.Conv2d is a function with defined number of weights
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=3,
                padding=1,
            ),

            # Pooling downsamples images to let understand broad context

            # ReLU adds non-linearity
            nn.ReLU(inplace=True),
            # Second convolution combines the first-level features
            nn.Conv2d(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=3,
                padding=1,
            ),
            # Again non-linearity by ReLU
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
    # is called when is written: model(before, after)
    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        """Predict change logits from two aligned images."""

        if before.shape != after.shape:
            raise ValueError(
                "Before and after tensors must have identical shapes. "
                f"Received {before.shape} and {after.shape}."
            )
        # Important since some channels can learn new features. Sharing weights
        # make more robust the model
        before_features = self.encoder(before)
        after_features = self.encoder(after)

        feature_difference = torch.abs(
            before_features - after_features
        )

        logits = self.output_layer(feature_difference)

        return logits
