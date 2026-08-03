"""Synchronized transformations for change-detection samples."""


import torch


class RandomFlipRotate:
    """Apply the same random flips and rotation to paired images and mask."""

    def __init__(
        self,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
    ) -> None:
        """initialize transformation probabilities."""

        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError(
                "horizontal_flip_probability must be between 0 and 1."
            )

        if not 0.0 <= vertical_flip_probability <= 1.0:
            raise ValueError(
                "vertical_flip_probability must be between 0 and 1."
            )

        self.horizontal_flip_probability = (
            horizontal_flip_probability
        )

        self.vertical_flip_probability = (
            vertical_flip_probability
        )

    def __call__(
        self,
        sample: dict[str, torch.Tensor | str],
    ) -> dict[str, torch.Tensor | str]:
        """Transform one aligned before-after-mask sample."""

        before = sample["before"]
        after = sample["after"]
        mask = sample["mask"]

        if not isinstance(before, torch.Tensor):
            raise TypeError("sample['before'] must be a tensor.")

        if not isinstance(after, torch.Tensor):
            raise ValueError("sample['after'] must be a tensor.")

        if not isinstance(mask, torch.Tensor):
            raise ValueError("sample['mask'] must be a tensor.")

        if torch.rand(1).item() < self.horizontal_flip_probability:
            before = torch.flip(before, dims=(-1,))
            after = torch.flip(after, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        if torch.rand(1).item() < self.vertical_flip_probability:
            before = torch.flip(before, dims=(-2,))
            after = torch.flip(after, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))

        quarter_turns = int(
            torch.randint(
                low=0,
                high=4,
                size=(1,),
            ).item()
        )

        before = torch.rot90(
            before,
            k=quarter_turns,
            dims=(-2, -1),
        )

        after = torch.rot90(
            after,
            k=quarter_turns,
            dims=(-2, -1),
        )

        mask = torch.rot90(
            mask,
            k=quarter_turns,
            dims=(-2, -1),
        )

        return {
            **sample,
            "before": before.contiguous(),
            "after": after.contiguous(),
            "mask": mask.contiguous(),
        }