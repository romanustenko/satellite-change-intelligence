"""Deterministic component crops in the original image coordinate frame."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from satchange.models.tiny_siamese_change_classifier import validate_unit_tensor


@dataclass(frozen=True)
class ROISettings:
    """Starting crop settings, not optimized values; all distances are native pixels."""

    threshold: float = 0.5
    connectivity: int = 8
    context_pixels: int = 16
    min_crop_size: int = 128
    min_component_area: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1], excluding zero-weight components.")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")
        for name, minimum in (
            ("context_pixels", 0),
            ("min_crop_size", 4),
            ("min_component_area", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass
class ROI:
    """Aligned crops with component and crop boxes (x0,y0,x1,y1), upper bounds exclusive."""

    component_id: int
    component_bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]
    area: int
    before: torch.Tensor
    after: torch.Tensor
    probabilities: torch.Tensor
    support: torch.Tensor
    mask: torch.Tensor
    minimum_extent_limited: bool


DEFAULT_ROI_SETTINGS = ROISettings()


def validate_scene(before: torch.Tensor, after: torch.Tensor, probabilities: torch.Tensor) -> None:
    """Require native, aligned RGB [3,H,W] and probability [1,H,W] tensors."""
    if before.ndim != 3 or before.shape[0] != 3 or before.shape != after.shape:
        raise ValueError("Scene images must be aligned [3,H,W].")
    if min(before.shape[-2:]) < 4:
        raise ValueError("Native images are too small: the backbone requires height/width >= 4.")
    if probabilities.shape != (1, *before.shape[-2:]):
        raise ValueError("Probability map must already align with native RGB as [1,H,W].")
    if before.dtype != torch.float32 or after.dtype != torch.float32:
        raise ValueError("Native RGB must be float32 normalized to [0, 1].")
    if before.device != after.device or before.device != probabilities.device:
        raise ValueError("All scene tensors must use the same device.")
    for name, value in (("before", before), ("after", after), ("probabilities", probabilities)):
        validate_unit_tensor(value, name)


def _extend_axis(low: int, high: int, extent: int, settings: ROISettings) -> tuple[int, int]:
    length = min(extent, max(high - low + 2 * settings.context_pixels, settings.min_crop_size))
    start = max(0, min((low + high - length) // 2, extent - length))
    return start, start + length


def extract_rois(
    before: torch.Tensor,
    after: torch.Tensor,
    probabilities: torch.Tensor,
    settings: ROISettings = DEFAULT_ROI_SETTINGS,
) -> list[ROI]:
    """Extract real context without RGB resizing/padding or truncating components.

    Components are sorted by their first row-major pixel, independently of OpenCV's
    label numbering. Only the selected component contributes soft pooling weights.
    Detector output geometry must be mapped by the caller before using this API.
    """
    validate_scene(before, after, probabilities)
    binary = (probabilities[0] >= settings.threshold).detach().cpu().numpy().astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=settings.connectivity
    )
    components = []
    unique_labels, first_pixels = np.unique(labels, return_index=True)
    first_by_label = dict(zip(unique_labels.tolist(), first_pixels.tolist(), strict=True))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        first = first_by_label[label]
        components.append((first, label, area))
    height, width = before.shape[-2:]
    results = []
    for component_id, (_, label, area) in enumerate(sorted(components), start=1):
        if area < settings.min_component_area:
            continue
        x0, y0, w, h = (int(value) for value in stats[label, :4])
        x1, y1 = x0 + w, y0 + h
        left, right = _extend_axis(x0, x1, width, settings)
        top, bottom = _extend_axis(y0, y1, height, settings)
        support = torch.from_numpy((labels[top:bottom, left:right] == label).copy())
        support = support.unsqueeze(0).to(device=before.device)
        probability_crop = probabilities[:, top:bottom, left:right].contiguous()
        results.append(
            ROI(
                component_id=component_id,
                component_bbox=(x0, y0, x1, y1),
                crop_bbox=(left, top, right, bottom),
                area=area,
                before=before[:, top:bottom, left:right].contiguous(),
                after=after[:, top:bottom, left:right].contiguous(),
                probabilities=probability_crop,
                support=support,
                mask=(probability_crop * support).float().contiguous(),
                minimum_extent_limited=height < settings.min_crop_size
                or width < settings.min_crop_size,
            )
        )
    return results


def align_whole_image_probabilities(
    probabilities: torch.Tensor,
    original_size: tuple[int, int],
    detector_input_size: tuple[int, int],
) -> torch.Tensor:
    """Map a known *whole-image resize* detector probability grid back to native (H,W).

    Caller must verify that preprocessing was only a whole-image resize. Cropped,
    padded, and tiled outputs need their own known inverse geometry; this helper
    must not be used to guess it. RGB is never resampled by this function.
    """
    if probabilities.shape != (1, *detector_input_size):
        raise ValueError("Probability grid must match the documented detector input size.")
    if len(original_size) != 2 or any(
        type(value) is not int or value < 4 for value in original_size
    ):
        raise ValueError("original_size must be native (height,width), both >= 4.")
    validate_unit_tensor(probabilities, "probabilities")
    return (
        F.interpolate(
            probabilities.float().unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )[0]
        .clamp(0, 1)
        .contiguous()
    )
