"""Calculating positive weights for BCELogitsWithLoss."""

def calculate_positive_weight(
    data_loader: DataLoader,
) -> tuple[float, int, int]:
    """Calculate the negative-to-positive pixel ratio."""

    positive_pixels = 0
    total_pixels = 0

    for batch in data_loader:
        masks = batch["mask"]

        if masks.ndim != 4:
            raise ValueError(
                "Masks must have shape [B, C, H, W]. "
                f"Received {masks.shape}."
            )

        target_mask = masks >= 0.5

        positive_pixels += int(
            target_mask.sum().item()
        )

        total_pixels += target_mask.numel()

    if total_pixels == 0:
        raise RuntimeError(
            "The training data loader contains no pixels."
        )

    negative_pixels = (
        total_pixels - positive_pixels
    )

    if positive_pixels == 0:
        raise RuntimeError(
            "The training split contains no changed pixels."
        )

    positive_weight = (
        negative_pixels / positive_pixels
    )

    return (
        positive_weight,
        positive_pixels,
        negative_pixels,
    )