"""WHAT stage: ROI-crop dataset, crop-to-scene reconstruction, oracle-ROI evaluation (CPU)."""

from typing import Any

import pytest
import torch

from satchange.models import FCSiamDiff
from satchange.semantic.roi import extract_rois
from satchange.semantic.what_stage import (
    WINDOW,
    PreloadedSemanticDataset,
    ROICropDataset,
    evaluate_what_pipeline,
    ground_truth_roi_boxes,
    ground_truth_roi_index,
    reconstruct_scene,
)

SIZE = 256


class _FakeBase:
    """Two scenes; each has one change block (10 px apart) and matching T1/T2 labels."""

    transform = None
    sample_ids = ["a", "b"]

    def __init__(self) -> None:
        self.blocks = {"a": [(40, 40, 70, 80)], "b": [(10, 10, 30, 30), (100, 100, 230, 240)]}

    def _change(self, sample_id: str) -> torch.Tensor:
        mask = torch.zeros(1, SIZE, SIZE)
        for x0, y0, x1, y1 in self.blocks[sample_id]:
            mask[:, y0:y1, x0:x1] = 1
        return mask

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        mask = self._change(sample_id)
        label = (mask[0] * 3).long()
        generator = torch.Generator().manual_seed(index)
        return {
            "before": torch.rand(3, SIZE, SIZE, generator=generator),
            "after": torch.rand(3, SIZE, SIZE, generator=generator),
            "mask": mask,
            "before_label": label,
            "after_label": label.clone(),
            "sample_id": sample_id,
        }


def test_roi_index_counts_components_in_dataset_order() -> None:
    assert ground_truth_roi_index(_FakeBase()) == [(0, 0), (1, 0), (1, 1)]  # type: ignore[arg-type]


def test_crop_dataset_windows_stay_inside_roi_and_are_fixed_size() -> None:
    base = _FakeBase()
    entries = ground_truth_roi_index(base)  # type: ignore[arg-type]
    centered = ROICropDataset(base, entries)  # type: ignore[arg-type]
    assert len(centered) == 3
    for index in range(3):
        sample = centered[index]
        assert sample["before"].shape == (3, WINDOW, WINDOW)
        assert sample["before_label"].shape == (WINDOW, WINDOW)
        assert sample["sample_id"].count("#") == 1
    assert torch.equal(centered[0]["before"], centered[0]["before"])  # eval windows are fixed
    random_window = ROICropDataset(base, entries, random_window=True)  # type: ignore[arg-type]
    torch.manual_seed(0)
    starts = {tuple(random_window[2]["before"][0, 0, :4].tolist()) for _ in range(8)}
    assert len(starts) > 1  # the 128 px window moves inside the larger crop


class _QuantizedBase(_FakeBase):
    """Same scenes with RGB values that are exact multiples of 1/255 (as real SECOND tiles)."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        for key in ("before", "after"):
            sample[key] = (sample[key] * 255).round() / 255
        return sample


def test_preloaded_dataset_and_cached_boxes_reproduce_the_slow_path() -> None:
    base = _QuantizedBase()
    preloaded = PreloadedSemanticDataset(base, num_workers=0)  # type: ignore[arg-type]
    for index in range(len(base)):
        for key, value in base[index].items():
            other = preloaded[index][key]
            assert torch.equal(other, value) if torch.is_tensor(value) else other == value
            if torch.is_tensor(value):
                assert other.dtype == value.dtype
    triples = ground_truth_roi_boxes(base)  # type: ignore[arg-type]
    entries = [(index, ordinal) for index, ordinal, _ in triples]
    assert entries == ground_truth_roi_index(base)  # type: ignore[arg-type]
    slow = ROICropDataset(base, entries)  # type: ignore[arg-type]
    fast = ROICropDataset(
        preloaded, entries, boxes=[box for _, _, box in triples]  # type: ignore[arg-type]
    )
    for index in range(len(entries)):
        for key, value in slow[index].items():
            other = fast[index][key]
            assert torch.equal(other, value) if torch.is_tensor(value) else other == value


def test_preloaded_dataset_rejects_non_uint8_images() -> None:
    with pytest.raises(ValueError, match="uint8"):
        PreloadedSemanticDataset(_FakeBase(), num_workers=0)  # type: ignore[arg-type]


def test_reconstruction_zero_outside_and_overlap_is_order_stable() -> None:
    base = _FakeBase()
    sample = base[1]
    rois = extract_rois(sample["before"], sample["after"], sample["mask"])
    logits = []
    for _ in rois:
        crop = torch.zeros(14, *rois[len(logits)].before.shape[-2:])
        crop[2] = 5.0  # T1 class 2
        crop[7 + 5] = 5.0  # T2 class 5
        logits.append(crop)
    first, second = reconstruct_scene(rois, logits, (SIZE, SIZE))
    covered = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    for roi in rois:
        x0, y0, x1, y1 = roi.crop_bbox
        covered[y0:y1, x0:x1] = True
    assert (first[covered] == 2).all() and (second[covered] == 5).all()
    assert (first[~covered] == 0).all() and (second[~covered] == 0).all()
    again = reconstruct_scene(rois, logits, (SIZE, SIZE))
    assert torch.equal(again[0], first) and torch.equal(again[1], second)
    with pytest.raises(ValueError):
        reconstruct_scene(rois, logits[:1], (SIZE, SIZE))


def test_oracle_pipeline_runs_end_to_end_with_odd_crop_sizes() -> None:
    torch.manual_seed(0)
    model = FCSiamDiff(in_channels=3, out_channels=14, base_channels=8)
    metrics = evaluate_what_pipeline(model, _FakeBase(), torch.device("cpu"))  # type: ignore[arg-type]
    assert {"sek", "miou", "fscd"} <= set(metrics)
    assert all(0.0 <= float(metrics[key]) <= 1.0 for key in ("miou", "fscd"))
