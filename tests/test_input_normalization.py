"""Explicit per-model input normalization in the SECOND loaders (R2-V2-T01)."""

import pytest
import torch
from test_second_binary import make_second

from satchange.data import RandomFlipRotate, create_dataset_dataloader
from satchange.data.normalization import (
    INPUT_NORMALIZATIONS,
    normalize_image,
    normalize_sample,
    validate_input_normalization,
)
from satchange.data.second_semantic import SECONDSemanticDataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def test_default_is_unchanged_and_imagenet_matches_manual_formula(tmp_path):
    root = make_second(tmp_path / "SECOND")
    raw, _ = create_dataset_dataloader(root, "train", None, 1, 0, semantic_labels=True)
    normalized, loader = create_dataset_dataloader(
        root, "train", None, 2, 0, semantic_labels=True, input_normalization="imagenet"
    )
    assert isinstance(normalized, SECONDSemanticDataset)
    assert raw.input_normalization is None and normalized.input_normalization == "imagenet"
    plain, scaled = raw[0], normalized[0]
    for key in ("before", "after"):
        assert 0 <= float(plain[key].min()) and float(plain[key].max()) <= 1
        torch.testing.assert_close(scaled[key], (plain[key] - IMAGENET_MEAN) / IMAGENET_STD)
    for key in ("mask", "before_label", "after_label", "sample_id"):
        assert torch.equal(plain[key], scaled[key]) if key != "sample_id" else (
            plain[key] == scaled[key]
        )
    assert next(iter(loader))["before"].shape == (2, 3, 32, 32)


def test_binary_loader_and_transform_order(tmp_path):
    root = make_second(tmp_path / "SECOND")
    transform = RandomFlipRotate(horizontal_flip_probability=1.0, vertical_flip_probability=1.0)
    raw, _ = create_dataset_dataloader(root, "train", None, 1, 0, transform=transform)
    normalized, _ = create_dataset_dataloader(
        root, "train", None, 1, 0, transform=transform, input_normalization="imagenet"
    )
    torch.manual_seed(0)
    flipped = raw[1]
    torch.manual_seed(0)
    both = normalized[1]
    torch.testing.assert_close(both["before"], normalize_image(flipped["before"], "imagenet"))
    assert torch.equal(both["mask"], flipped["mask"])


def test_unknown_normalization_and_levir_root_fail(tmp_path):
    root = make_second(tmp_path / "SECOND")
    with pytest.raises(ValueError, match="Unknown input normalization"):
        create_dataset_dataloader(root, "train", None, 1, 0, input_normalization="imagenet21k")
    levir = tmp_path / "levir"
    levir.mkdir()
    with pytest.raises(ValueError, match="only supported for SECOND"):
        create_dataset_dataloader(levir, "train", None, 1, 0, input_normalization="imagenet")


def test_helpers_validate_and_keep_non_image_keys():
    assert validate_input_normalization(None) is None
    assert sorted(INPUT_NORMALIZATIONS) == ["imagenet"]
    batch = torch.rand(2, 3, 4, 4)
    torch.testing.assert_close(
        normalize_image(batch, "imagenet"), (batch - IMAGENET_MEAN) / IMAGENET_STD
    )
    with pytest.raises(ValueError, match="RGB tensor"):
        normalize_image(torch.rand(1, 4, 4), "imagenet")
    sample = {"before": torch.ones(3, 2, 2), "after": torch.zeros(3, 2, 2), "mask": torch.ones(1)}
    assert normalize_sample(sample, None) is sample
    result = normalize_sample(sample, "imagenet")
    assert result["mask"] is sample["mask"] and not torch.equal(result["after"], sample["after"])
