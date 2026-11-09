"""Research 2 WHAT stage: ROI-crop semantic classifier and crop-to-scene reconstruction.

The WHAT model is the same dense semantic model as system A (``[B,14,h,w]`` logits, see
``scd_training``) applied to native-resolution ROI crops from ``roi.extract_rois``; outside every
ROI the prediction is "no change" (class 0) in both maps (R2-P10-T03). Training reuses the shared
engine through ``ROICropDataset`` (same batch contract as ``SECONDSemanticDataset``); ROIs for
training/oracle evaluation come from the ground-truth change mask, detector ROIs from any
probability source. Overlaps between crops are merged deterministically: log-softmax values are
averaged over the crops covering a pixel, in ROI order.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from satchange.data.second_semantic import SECONDSemanticDataset
from satchange.semantic.roi import DEFAULT_ROI_SETTINGS, ROI, ROISettings, extract_rois
from satchange.semantic.scd_metrics import SemanticChangeMetrics
from satchange.semantic.scd_training import NUM_CLASSES, split_logits

WINDOW = 128  # fixed training window (native px); every ROI crop is >= min_crop_size = 128
PAD_MULTIPLE = 16  # encoder-decoder models need spatial sizes divisible by this
LABEL_KEYS = ("before_label", "after_label")


def ground_truth_roi_index(
    dataset: SECONDSemanticDataset, settings: ROISettings = DEFAULT_ROI_SETTINGS
) -> list[tuple[int, int]]:
    """(sample index, ROI ordinal) for every ground-truth change component, in dataset order."""
    entries: list[tuple[int, int]] = []
    for index, sample_id in enumerate(dataset.sample_ids):
        mask = dataset._change(sample_id)
        scene = torch.zeros(3, *mask.shape[-2:])
        count = len(extract_rois(scene, scene, mask, settings))
        entries.extend((index, ordinal) for ordinal in range(count))
    return entries


def ground_truth_roi_boxes(
    dataset: SECONDSemanticDataset, settings: ROISettings = DEFAULT_ROI_SETTINGS
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """``(sample index, ROI ordinal, crop bbox)`` for every ground-truth change component."""
    entries: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for index, sample_id in enumerate(dataset.sample_ids):
        mask = dataset._change(sample_id)
        scene = torch.zeros(3, *mask.shape[-2:])
        for ordinal, roi in enumerate(extract_rois(scene, scene, mask, settings)):
            entries.append((index, ordinal, roi.crop_bbox))
    return entries


class PreloadedSemanticDataset(Dataset):
    """In-RAM uint8 copy of a ``SECONDSemanticDataset`` split; items equal the source's.

    The ROI window loader would otherwise decode a whole 512x512 tile (about 0.25 s) for every
    128x128 window. Images are exact multiples of 1/255, labels are class ids <= 6 and the mask is
    binary, so the uint8 round trip is lossless.
    """

    def __init__(self, base: SECONDSemanticDataset, *, num_workers: int = 8) -> None:
        if base.transform is not None:
            raise ValueError("The base dataset must not carry a transform.")
        self.sample_ids = list(base.sample_ids)
        self.transform = None
        loader = DataLoader(base, batch_size=None, num_workers=num_workers)
        self._items: list[dict[str, Any]] = []
        for sample in loader:
            item: dict[str, Any] = {"sample_id": sample["sample_id"]}
            for key in ("before", "after"):
                scaled = (sample[key] * 255).round()
                if not torch.equal(scaled / 255, sample[key]):
                    raise ValueError(f"{key} is not a uint8 image: {sample['sample_id']}")
                item[key] = scaled.to(torch.uint8)
            item["mask"] = sample["mask"].to(torch.uint8)
            for key in LABEL_KEYS:
                item[key] = sample[key].to(torch.uint8)
            self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._items[index]
        return {
            "before": item["before"].float() / 255,
            "after": item["after"].float() / 255,
            "mask": item["mask"].float(),
            "sample_id": item["sample_id"],
            **{key: item[key].long() for key in LABEL_KEYS},
        }


class ROICropDataset(Dataset):
    """Fixed ``window`` x ``window`` windows inside ground-truth ROI crops.

    Training draws a random window position (torch RNG, seeded per DataLoader worker); evaluation
    takes the centered window. The sample keeps the ``SECONDSemanticDataset`` keys, so
    ``semantic.scd_training.run_epoch`` and ``fit_binary_change_model`` work unchanged.
    """

    def __init__(
        self,
        base: SECONDSemanticDataset,
        entries: list[tuple[int, int]],
        *,
        settings: ROISettings = DEFAULT_ROI_SETTINGS,
        window: int = WINDOW,
        random_window: bool = False,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        if base.transform is not None:
            raise ValueError("The base dataset must not carry a transform.")
        if window > settings.min_crop_size:
            raise ValueError("window cannot exceed the minimum ROI crop size.")
        if boxes is not None and len(boxes) != len(entries):
            raise ValueError("boxes must align with entries.")
        self.base, self.entries, self.settings, self.boxes = base, entries, settings, boxes
        self.window, self.random_window, self.transform = window, random_window, transform

    def __len__(self) -> int:
        return len(self.entries)

    def _offset(self, extent: int) -> int:
        spare = extent - self.window
        if spare <= 0:
            return 0
        if self.random_window:
            return int(torch.randint(0, spare + 1, (1,)))
        return spare // 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, ordinal = self.entries[index]
        sample = self.base[sample_index]
        if self.boxes is not None:
            x0, y0, x1, y1 = self.boxes[index]
        else:
            rois = extract_rois(sample["before"], sample["after"], sample["mask"], self.settings)
            x0, y0, x1, y1 = rois[ordinal].crop_bbox
        if x1 - x0 < self.window or y1 - y0 < self.window:
            raise ValueError(f"ROI crop smaller than the window: {sample['sample_id']}")
        left = x0 + self._offset(x1 - x0)
        top = y0 + self._offset(y1 - y0)
        region = (slice(top, top + self.window), slice(left, left + self.window))
        out: dict[str, Any] = {
            "before": sample["before"][(slice(None), *region)].contiguous(),
            "after": sample["after"][(slice(None), *region)].contiguous(),
            "mask": sample["mask"][(slice(None), *region)].contiguous(),
            "sample_id": f"{sample['sample_id']}#{ordinal}",
        }
        for key in LABEL_KEYS:
            out[key] = sample[key][region].contiguous()
        return self.transform(out) if self.transform is not None else out


@torch.no_grad()
def predict_roi_logits(model: torch.nn.Module, roi: ROI, device: torch.device) -> torch.Tensor:
    """Logits ``[14,h,w]`` for one full-size ROI crop (reflect padding removed afterwards)."""
    height, width = roi.before.shape[-2:]
    pad_h, pad_w = (-height) % PAD_MULTIPLE, (-width) % PAD_MULTIPLE

    def pad(tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.unsqueeze(0).to(device)
        return F.pad(batch, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else batch

    logits: torch.Tensor = model(pad(roi.before), pad(roi.after)).float()[0, :, :height, :width]
    return logits


def reconstruct_scene(
    rois: list[ROI], logits: list[torch.Tensor], shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge crop logits into native-coordinate T1/T2 class maps ``[H,W]``; outside ROIs = 0."""
    if len(rois) != len(logits):
        raise ValueError("One logit tensor per ROI is required.")
    device = logits[0].device if logits else torch.device("cpu")
    accumulated = torch.zeros(2, NUM_CLASSES, *shape, device=device)
    covered = torch.zeros(*shape, device=device)
    for roi, crop_logits in zip(rois, logits, strict=True):
        left, top, right, bottom = roi.crop_bbox
        first, second = split_logits(crop_logits.unsqueeze(0))
        log_probability = torch.stack(
            [F.log_softmax(first[0], dim=0), F.log_softmax(second[0], dim=0)]
        )
        accumulated[:, :, top:bottom, left:right] += log_probability
        covered[top:bottom, left:right] += 1
    maps = accumulated.argmax(dim=1)
    maps = torch.where(covered > 0, maps, torch.zeros_like(maps))
    return maps[0], maps[1]


def evaluate_what_pipeline(
    model: torch.nn.Module,
    dataset: SECONDSemanticDataset,
    device: torch.device,
    *,
    probability_source: Callable[[dict[str, Any]], torch.Tensor] | None = None,
    settings: ROISettings = DEFAULT_ROI_SETTINGS,
) -> dict[str, float]:
    """Scene-level SeK/mIoU/Fscd of detector -> crops -> WHAT -> reconstruction.

    ``probability_source`` maps a dataset sample to a native ``[1,H,W]`` probability map;
    ``None`` uses the ground-truth change mask (oracle ROIs: the WHAT upper bound).
    """
    model.eval()
    metrics = SemanticChangeMetrics(NUM_CLASSES)
    for index in range(len(dataset)):
        sample = dataset[index]
        probabilities = (
            sample["mask"] if probability_source is None else probability_source(sample)
        )
        rois = extract_rois(sample["before"], sample["after"], probabilities, settings)
        logits = [predict_roi_logits(model, roi, device) for roi in rois]
        shape = tuple(sample["before"].shape[-2:])
        if logits:
            first, second = reconstruct_scene(rois, logits, shape)  # type: ignore[arg-type]
        else:
            first = second = torch.zeros(shape, dtype=torch.long, device=device)
        metrics.update(first.cpu(), sample["before_label"])
        metrics.update(second.cpu(), sample["after_label"])
    return metrics.compute()
