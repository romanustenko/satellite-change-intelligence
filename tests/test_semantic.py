"""ROI extraction, detector probability maps and map I/O on synthetic CPU data."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from satchange.execution import resolve_device
from satchange.semantic.data import read_map
from satchange.semantic.inference import detector_probability_map
from satchange.semantic.roi import ROISettings, align_whole_image_probabilities, extract_rois


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def scene_tensors(height=171, width=197):
    before = torch.arange(height * width).reshape(1, height, width).float()
    before = (before / (height * width)).expand(3, -1, -1).contiguous()
    return before, 1 - before, torch.zeros(1, height, width)


@pytest.mark.parametrize("y,x", [(0, 0), (0, 196), (170, 0), (170, 196), (90, 100)])
def test_real_context_near_all_borders(y, x):
    before, after, probability = scene_tensors()
    probability[0, y, x] = 0.9
    (roi,) = extract_rois(before, after, probability)
    assert roi.before.shape == (3, 128, 128)
    assert not roi.minimum_extent_limited
    x0, y0, x1, y1 = roi.crop_bbox
    torch.testing.assert_close(roi.before, before[:, y0:y1, x0:x1])
    torch.testing.assert_close(roi.after, after[:, y0:y1, x0:x1])
    assert roi.component_bbox == (x, y, x + 1, y + 1)
    assert roi.mask.sum() == pytest.approx(0.9)
    assert x0 <= x < x1 and y0 <= y < y1


def test_scale_large_components_small_images_neighbors_and_order():
    before, after, probability = scene_tensors(180, 240)
    probability[0, 2:32, 4:44] = 0.8
    probability[0, 40:42, 50:52] = 0.7  # Included in context but excluded from first pooling mask.
    rois = extract_rois(before, after, probability)
    assert [roi.area for roi in rois] == [30 * 40, 4]
    assert rois[0].probabilities.sum() > rois[0].mask.sum()
    assert int(rois[0].support.sum()) == 30 * 40
    assert [roi.crop_bbox for roi in rois] == [
        roi.crop_bbox for roi in extract_rois(before, after, probability)
    ]
    large = torch.ones_like(probability)
    (roi,) = extract_rois(before, after, large)
    assert roi.before.shape == before.shape and roi.area == 180 * 240
    before, after, probability = scene_tensors(37, 51)
    probability[:, 1, 1] = 1
    (roi,) = extract_rois(before, after, probability)
    assert roi.minimum_extent_limited and roi.before.shape == (3, 37, 51)
    assert extract_rois(before, after, torch.zeros_like(probability)) == []


def test_connectivity_min_area_and_probability_alignment():
    before, after, probability = scene_tensors(10, 12)
    probability[0, 2, 2] = probability[0, 3, 3] = 1
    settings = ROISettings(min_crop_size=4, context_pixels=0)
    assert len(extract_rois(before, after, probability, settings)) == 1
    assert len(extract_rois(before, after, probability, replace(settings, connectivity=4))) == 2
    assert extract_rois(before, after, probability, replace(settings, min_component_area=3)) == []
    resized = align_whole_image_probabilities(torch.ones(1, 4, 5), (10, 12), (4, 5))
    assert resized.shape == (1, 10, 12)
    torch.testing.assert_close(resized, torch.ones_like(resized))
    with pytest.raises(ValueError):
        align_whole_image_probabilities(probability, (10, 12), (4, 5))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_roi_rejects_invalid_probabilities(bad):
    before, after, probabilities = scene_tensors(8, 8)
    probabilities[0, 1, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        extract_rois(before, after, probabilities)


def test_roi_rejects_spatial_mismatch_and_small_input():
    before, after, probabilities = scene_tensors(8, 8)
    with pytest.raises(ValueError, match="align"):
        extract_rois(before, after, probabilities[:, :-1])
    with pytest.raises(ValueError, match="too small"):
        extract_rois(before[:, :3], after[:, :3], probabilities[:, :3])


def test_component_ids_stay_stable_after_area_filtering():
    before, after, probability = scene_tensors(20, 20)
    probability[0, 1, 1] = 1
    probability[0, 10:12, 10:12] = 1
    settings = ROISettings(min_crop_size=4, context_pixels=0)
    assert [roi.component_id for roi in extract_rois(before, after, probability, settings)] == [
        1,
        2,
    ]
    filtered = extract_rois(before, after, probability, replace(settings, min_component_area=2))
    assert [roi.component_id for roi in filtered] == [2]


@pytest.mark.parametrize("bad", [1.000000001, -0.000000001, float("nan"), 2 + 1j])
def test_map_validation_precedes_float32_conversion(tmp_path, bad):
    path = tmp_path / "probabilities.npy"
    np.save(path, np.full((4, 5), bad))
    with pytest.raises(ValueError, match="finite"):
        read_map(path)


def test_device_indices_and_detector_explicit_output_geometry(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_device("cuda") == torch.device("cuda:2")
    assert resolve_device("cuda:3") == torch.device("cuda:3")
    with pytest.raises(ValueError, match="not visible"):
        resolve_device("cuda:4")

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, before, after):
            return torch.zeros_like(before[:, :1]) + self.parameter

    detector = Detector()
    before, after, _ = scene_tensors(17, 23)
    output = detector_probability_map(
        detector, before, after, input_size=(8, 8), output_kind="logits"
    )
    assert output.shape == (1, 17, 23)
    torch.testing.assert_close(output, torch.full_like(output, 0.5))
    assert not detector.training and detector.parameter.grad is None
    output = detector_probability_map(
        detector, before, after, input_size=(8, 8), output_kind="probabilities"
    )
    assert (output == 0).all()  # No double sigmoid or heuristic.
    with pytest.raises(ValueError, match="Unsupported"):
        detector_probability_map(
            detector, before, after, input_size=(8, 8), output_kind="logits", geometry="tiles"
        )
