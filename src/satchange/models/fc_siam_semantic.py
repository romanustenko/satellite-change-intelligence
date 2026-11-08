"""FC-Siam with a date-aware decoder for dense semantic change detection (Research 2).

``FCSiamDiff`` feeds its decoder only ``|F(before) - F(after)|``: a symmetric quantity from which
neither the order of the dates nor the absolute land-cover class can be recovered, so per-date
class maps are structurally out of reach (system A reached SeK ~ 0 with it). This model keeps the
same shared Siamese encoder (``encoder_blocks`` weights are interchangeable with ``FCSiamDiff``,
so a LEVIR-CD encoder can initialize it) and widens the decoder to see, at every level,
``concat([F(before), F(after), |F(before) - F(after)|])``.
"""

import torch
from torch import nn

from satchange.models.fc_siam_diff import ConvBlock, FCSiamDiff


class FCSiamSemantic(FCSiamDiff):
    """Siamese encoder, date-aware decoder; ``[B,out_channels,H,W]`` logits."""

    def __init__(
        self, in_channels: int = 3, out_channels: int = 14, base_channels: int = 16
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for level in range(len(self.channels) - 1, 0, -1):
            # only the bottleneck is fused; later levels take the previous decoder output
            bottleneck = level == len(self.channels) - 1
            wide = (3 if bottleneck else 1) * self.channels[level]
            narrow = self.channels[level - 1]
            self.upconvs.append(nn.ConvTranspose2d(wide, narrow, kernel_size=2, stride=2))
            self.decoder_blocks.append(ConvBlock(narrow + 3 * narrow, narrow))

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.shape != after.shape:
            raise ValueError(
                f"Before and after tensors must have identical shapes. "
                f"Received {before.shape} and {after.shape}."
            )
        first, second = self._encode(before), self._encode(after)
        fused = [
            torch.cat([a, b, (a - b).abs()], dim=1)
            for a, b in zip(first, second, strict=True)
        ]
        return self.output_layer(self._decode(fused))
