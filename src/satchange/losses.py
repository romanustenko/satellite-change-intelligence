"""Loss functions for binary and semantic change detection."""


import torch
from torch import nn



class DiceLoss(nn.Module):
    """Compute Dice loss for binary segmentation logits."""

    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        """Initialize Dice loss."""

        super().__init__()

        if smooth <= 0:
            raise ValueError("smooth must be positive")

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss from logits and binary targets."""

        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have identical shapes. "
                f"Received {logits.shape} and {targets.shape}."
            )

        probabilities = torch.sigmoid(logits)

        probabilities  = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        probability_sum = probabilities.sum(dim=1)
        target_sum = targets.sum(dim=1)

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            probability_sum
            + target_sum
            + self.smooth
        )

        return 1.0 - dice_score.mean()



class BCEDiceloss(nn.Module):
    """Combine binary cross-entropy and Dice losses."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        """Initialize the combined loss."""

        super().__init__()

        if bce_weight < 0:
            raise ValueError("bce_weight cannot be negative.")

        if dice_weight < 0:
            raise ValueError("dice_weight cannot be negative.")

        if bce_weight + dice_weight <= 0:
            raise ValueError(
                "At least one loss weight must be positive."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(
            smooth=smooth,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted BCE and Dice loss."""

        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have identical shapes. "
                f"Received {logits.shape} and {targets.shape}."
            )

        targets = targets.to(
            dtype=logits.dtype,
        )

        bce = self.bce_loss(
            logits,
            targets,
        )

        dice = self.dice_loss(
            logits,
            targets,
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )