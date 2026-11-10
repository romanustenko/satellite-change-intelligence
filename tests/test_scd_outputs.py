"""Model-head to common T1/T2 map conversion (R2-V2-T02), on synthetic outputs."""

import pytest
import torch

from satchange.semantic.scd_metrics import SemanticChangeMetrics
from satchange.semantic.scd_outputs import (
    change_mask_from_logits,
    maps_from_change_and_semantic,
    semantic_class_map,
)


def _one_hot_logits(classes: torch.Tensor, channels: int, scale: float = 5.0) -> torch.Tensor:
    return torch.nn.functional.one_hot(classes, channels).permute(0, 3, 1, 2).float() * scale


def test_change_heads_one_and_two_channels():
    probability = torch.tensor([[[[0.2, 0.6], [0.5, 0.9]]]])
    sigmoid_logits = torch.logit(probability)
    assert change_mask_from_logits(sigmoid_logits).tolist() == [[[False, True], [False, True]]]
    assert change_mask_from_logits(sigmoid_logits, change_threshold=0.85).tolist() == [
        [[False, False], [False, True]]
    ]
    softmax_logits = torch.cat([torch.zeros_like(sigmoid_logits), sigmoid_logits], dim=1)
    assert torch.equal(change_mask_from_logits(softmax_logits), probability[:, 0] > 0.5)
    with pytest.raises(ValueError, match="Change logits"):
        change_mask_from_logits(torch.zeros(1, 3, 2, 2))
    with pytest.raises(ValueError, match="change_threshold"):
        change_mask_from_logits(sigmoid_logits, change_threshold=1.0)


def test_semantic_argmax_rules_for_seven_and_six_channels():
    logits = torch.zeros(1, 7, 1, 3)
    logits[0, 0, 0, 0] = 9.0  # untrained no-change channel wins
    logits[0, 4, 0, 0] = 3.0
    logits[0, 2, 0, 1] = 1.0
    logits[0, 6, 0, 2] = 1.0
    assert semantic_class_map(logits, semantic_argmax="all").tolist() == [[[0, 2, 6]]]
    assert semantic_class_map(logits, semantic_argmax="land_cover").tolist() == [[[4, 2, 6]]]
    six = logits[:, 1:]
    for rule in ("all", "land_cover"):
        assert semantic_class_map(six, semantic_argmax=rule).tolist() == [[[4, 2, 6]]]
    with pytest.raises(ValueError, match="semantic_argmax"):
        semantic_class_map(logits, semantic_argmax="first")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Semantic logits"):
        semantic_class_map(torch.zeros(1, 5, 1, 1), semantic_argmax="all")


def test_maps_are_zero_outside_change_and_feed_the_metric():
    generator = torch.Generator().manual_seed(0)
    before = torch.randint(1, 7, (2, 8, 8), generator=generator)
    after = torch.randint(1, 7, (2, 8, 8), generator=generator)
    change = torch.rand(2, 8, 8, generator=generator) > 0.5
    change_logits = torch.where(change, 4.0, -4.0).unsqueeze(1)
    first, second = maps_from_change_and_semantic(
        change_logits,
        _one_hot_logits(before, 7),
        _one_hot_logits(after - 1, 6),
        semantic_argmax="land_cover",
    )
    assert first.dtype == torch.int64 and first.shape == (2, 8, 8)
    assert torch.equal(first, torch.where(change, before, 0))
    assert torch.equal(second, torch.where(change, after, 0))
    metrics = SemanticChangeMetrics()
    metrics.update(first, torch.where(change, before, 0))
    metrics.update(second, torch.where(change, after, 0))
    assert metrics.compute()["miou"] == pytest.approx(1.0)


def test_shape_mismatch_between_heads_fails():
    with pytest.raises(ValueError, match="do not match the change head"):
        maps_from_change_and_semantic(
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 7, 4, 4),
            torch.zeros(1, 7, 2, 2),
            semantic_argmax="all",
        )
