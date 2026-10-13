"""Reusable neural network building blocks."""

import torch

from torch import Tensor, nn


class ConvBlock(nn.Module):
    '''Two 3x3 convolutions with batch normalization and ReLU.'''

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ) -> None:
        super().__init__()
        # Т.к. класс присваивается модели, то super.__init__ передает автоматически такую информацию как .device , веса , парамметры и т.д.

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            # анализ для каждого слоя. За перебор слоев отвечает nn.Sequential.


            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)
    

class SiameseEncoder(nn.Module):
    '''Shared encoder for both temporal images.'''

    def __init__(
            self,
            in_channels: int,
            base_channels: int,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock(
            in_channels,
            base_channels,
        )
        self.block2 = ConvBlock(
            base_channels,
            base_channels * 2,
        )
        self.block3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
        )
        self.block4 = ConvBlock(
            base_channels * 4,
            base_channels * 8,
        )
        self.bottleneck = ConvBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        feature1 = self.block1(inputs)
        feature2 = self.block2(self.pool(feature1))
        feature3 = self.block3(self.pool(feature2))
        feature4 = self.block4(self.pool(feature3))

        bottleneck = self.bottleneck(
            self.pool(feature4)
        )

        skip_features = [
            feature1,
            feature2,
            feature3,
            feature4,
        ]

        return skip_features, bottleneck


class UpBlock(nn.Module):
    """Upsample and combine decoder and skip features."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.fusion = ConvBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        inputs: Tensor,
        skip: Tensor,
    ) -> Tensor:
        inputs = self.upsample(inputs)

        if inputs.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "Decoder and skip features have different spatial sizes: "
                f"{inputs.shape[-2:]} and {skip.shape[-2:]}."
            )

        combined = torch.cat(
            [inputs, skip],
            dim=1,
        )

        return self.fusion(combined)
        