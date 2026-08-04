"""FC-Siam-Diff baseline model for binary change detection."""


import torch
from torch import nn

# nn.Module provides tools, utils and infrastracture to deal with a neural network.
# without this inheritance, our new class ConvBlock cannot call methods such as
# .to(device) , .train() etc. Even though they were imported in the file. Still they can't
# be used in our own class


# Feature is an output by weights of Conv2d
# features = conv(inputs): [2, 16, 256, 256] // feature tensor //
# it depends on convolutional weights and current input


# feature map shows where it was found and how strongly it was
# for each filter (16: 1, 2, 3, 4, 5, 6, ... 16) there is one feature map
# we call it : convolution extracts 16 features , which means 16 feature maps


# ConvBlock inherits behaviour of nn.Module
class ConvBlock(nn.Module):
    """Apply two convolutional layers with ReLU activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize the convolution block."""
        # super().__init__() is needed to run methods from parent's class nn.Module to
        # have acessed to them from the child class, namely ConvBlock
        super().__init__()
        # the same
        # nn.Module.__init__(self)

        self.layer = nn.Sequential(
            # learns features such as edges, corners, britness etc.
            # it does not automatically learn it. In the pair with gradient, optimization, backward,
            # it makes everything that some weights become more sensitive to corners, edges etc.
            nn.Conv2d(
                # in_channels = 3 and out_channels = 16. The method Conv2d receives 16 features produced by convolution (feature learning) and combine them with in_channels
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                # padding specifies what should be placed outside 3x3 region, since it begins
                # from upper left corner. 1 places 0 around
                padding=1,
            ),
            
            # ReLU(x) = max(0, x) -> non-linearity = model can learn difficult patterns
            # inplace = edit the current tensor instead of creating a copy. Better to save memory
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform an input feature tensor."""
        
        # It calls nn.Sequential that calls Conv2d, ReLU etc.
        return self.layer(inputs)



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

        self.channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        
        # nn.ModuleList is much more reliable than python list 'cause it can save parameters
        # such as weights, state_dict etc. direct to class, while python list cannot do this:
        # [ConvBlock(3, 16), ConvBlock(16, 16)]
        self.encoder_blocks = nn.ModuleList()

        current_channels = in_channels

        for next_channels in self.channels:
            self.encoder_blocks.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                )
            )

            current_channels = next_channels

        # we define pooling that keeps the largest value from 2x2 region
        # Pooling decreases the resolution of an image and increase the context by the deeper view
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )


        self.upconvs = nn.ModuleList()


        self.decoder_blocks = nn.ModuleList()


        for level_index in range(
            len(self.channels) - 1,
            0,
            -1,
        ):
            input_channels = self.channels[level_index]
            output_channels = self.channels[level_index - 1]

            self.upconvs.append(
                # ConvTranspose2d upsamples and changes channels:
                # chanenls: 256 -> 128
                # resolution: 16x16 -> 32x32
                # the problem is that it cannot restore every exact boundary only from 16x16 tensor
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=2,
                    stride=2,
                )
            )

            # After encoding from 32x32 to 16x16 it saves change evidence calculated at higher resolution
            # 32x32. The skip connection carries it to the corresponding decoder stage and concatenate
            # skip tensor comes from calculation's encoder of hugh resolution image


            self.decoder_blocks.append(
                ConvBlock(
                    in_channels=output_channels * 2,
                    out_channels=output_channels,
                )
            )

        # output layer that returns prediction per pixel
        self.output_layer = nn.Conv2d(
            in_channels=self.channels[0],
            out_channels=out_channels,
            # 1x1 kernel size. It produces one logit per spatial position
            kernel_size=1,
        )


    def _encode(
        self,
        inputs: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract feature maps at all encoder levels."""

        features = []
        current = inputs

        for level_index, block in enumerate(self.encoder_blocks):
            current = block(current)
            features.append(current)

            if level_index < len(self.encoder_blocks) - 1:
                current = self.pool(current)

        return features


    def _compute_feature_differences(
    self,
    before_features: list[torch.Tensor],
    after_features: list[torch.Tensor],
) -> list[torch.Tensor]:
        """Compute absolute feature differences at all encoder levels."""

        if len(before_features) != len(after_features):
            raise ValueError(
                "Before and after feature lists must have equal lengths."
            )

        feature_differences = []

        for before_feature, after_feature in zip(
            before_features,
            after_features,
        ):
            if before_feature.shape != after_feature.shape:
                raise ValueError(
                    "Corresponding feature tensors must have identical shapes. "
                    f"Received {before_feature.shape} and "
                    f"{after_feature.shape}."
                )

            difference = torch.abs(
                before_feature - after_feature
            )

            feature_differences.append(difference)

        return feature_differences


    def _decode(
        self,
        feature_difference: list[torch.Tensor],
    ) -> torch.Tensor:
        """Decode feature differences to the original spatial resolution."""

        if len(feature_difference) != len(self.channels):
            raise ValueError(
                "The number of feature differences must match "
                "the number of encoder levels. "
                f"Expected {len(self.channels)}, "
                f"received {len(feature_difference)}."
            )
        
        # Start with the deepest, lowerst-resolution difference
        current = feature_difference[-1]

        # All other feature differences are used as skip tensors,
        # starting from the lowerst resolution
        skip_differences = reversed(
            feature_difference[:-1]
        )

        for upconv, decoder_block, skip_difference in zip(
            self.upconvs,
            self.decoder_blocks,
            skip_differences,
        ):
            # Increase spatial resolution and reduce channel count
            current = upconv(current)

            if current.shape[0] != skip_difference.shape[0]:
                raise ValueError(
                    "Decoder and skip tensors must have "
                    "identical batch sizes. "
                    f"Received {current.shape[0]} and "
                    f"{skip_difference.shape[0]}."
                )

            if current.shape[2:] != skip_difference.shape[2:]:
                raise ValueError(
                    "Decoder and skip tensors must have "
                    "identical spatial dimensions. "
                    f"Received {current.shape[2:]} and "
                    f"{skip_difference.shape[2:]}."
                )

        # torch.cat concatenate values along channel dimension
        current = torch.cat(
            [
                current,
                skip_difference,
            ],
            dim=1,
        )

        # Put furhter to ConvBlock (Conv2d, ReLU, Conv2d and ReLU)
        current = decoder_block(current)

        return current

    
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
        before_features = self._encode(before)
        after_features = self._encode(after)

        # feature difference is difference between before_image and after_image feature map
        # | F(before) - F(after) |
        feature_differences = self._compute_feature_differences(
            before_features,
            after_features,
        )

        decoded_features = self._decode(
            feature_differences
        )

        logits = self.output_layer(
            decoded_features
        )

        return logits
