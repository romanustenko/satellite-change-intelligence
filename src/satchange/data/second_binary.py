"""SECOND binary change detection through a frozen split manifest (Research 2 WHERE stage).

A SECOND root that contains ``second_split_manifest.json`` (written by
``satchange.semantic.second.prepare_second_splits``) is a binary change-detection dataset for the
shared trainer: change = any pixel whose T1 label is not non-change (ID 0). T1 and T2 change
support must agree, as in the SECOND annotation protocol; disagreement fails loudly. Samples keep
the LEVIR-CD contract: ``before``/``after`` float32 [3,H,W] in [0,1], ``mask`` float32 [1,H,W]
in {0,1}, ``sample_id``. No normalization; optional whole-scene resize (bilinear RGB, nearest
mask) exactly as for LEVIR-CD.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from satchange.semantic.data import file_digest
from satchange.semantic.second import decode_label, read_second_split_manifest

SECOND_MANIFEST_NAME = "second_split_manifest.json"
SPLITS = ("train", "val", "test")


def is_second_binary_root(dataset_root: str | Path) -> bool:
    """True when the directory carries a frozen SECOND split manifest."""
    return (Path(dataset_root) / SECOND_MANIFEST_NAME).is_file()


def verify_second_sources(dataset_root: str | Path) -> int:
    """Re-hash every manifest file (expensive; run once after installation). Returns count."""
    root = Path(dataset_root)
    manifest = read_second_split_manifest(root / SECOND_MANIFEST_NAME)
    checked = 0
    for sample_id, sample in manifest["samples"].items():
        for key, entry in sample["files"].items():
            if file_digest(root / entry["path"]) != entry["sha256"]:
                raise ValueError(f"SECOND source changed since the manifest: {sample_id}/{key}")
            checked += 1
    return checked


class SECONDBinaryDataset(Dataset):
    """One split of a manifest-defined SECOND copy as binary change detection."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int | None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split {split!r}; expected one of {SPLITS}.")
        if image_size is not None and image_size <= 0:
            raise ValueError(f"Image size must be positive, received {image_size}")
        self.dataset_root = Path(dataset_root)
        self.manifest = read_second_split_manifest(self.dataset_root / SECOND_MANIFEST_NAME)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.sample_ids = list(self.manifest["splits"][split])
        if not self.sample_ids:
            raise ValueError(f"SECOND manifest split {split!r} is empty.")
        self.files = {
            sample_id: {
                key: self.dataset_root / entry["path"]
                for key, entry in self.manifest["samples"][sample_id]["files"].items()
            }
            for sample_id in self.sample_ids
        }
        missing = [str(p) for f in self.files.values() for p in f.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"SECOND manifest files are missing, e.g. {missing[:3]}")

        # LEVIRCDDataset-compatible per-sample path maps (used by final_benchmark).
        self.before_files = {k: f["before"] for k, f in self.files.items()}
        self.after_files = {k: f["after"] for k, f in self.files.items()}
        self.label_files = {k: f["before_semantic"] for k, f in self.files.items()}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _rgb(self, path: Path) -> torch.Tensor:
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.image_size is not None:
            image = image.resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BILINEAR
            )
        array = np.array(image, dtype=np.float32, copy=True)
        return torch.from_numpy(array / 255.0).permute(2, 0, 1).contiguous()

    def _change(self, sample_id: str) -> torch.Tensor:
        files = self.files[sample_id]
        before = decode_label(files["before_semantic"]) != 0
        after = decode_label(files["after_semantic"]) != 0
        if before.shape != after.shape or not np.array_equal(before, after):
            raise ValueError(f"SECOND T1/T2 change support disagrees: {sample_id}")
        mask = Image.fromarray(before.astype(np.uint8) * 255)
        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            mask = mask.resize(size, resample=Image.Resampling.NEAREST)
        binary = (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        before = self._rgb(self.files[sample_id]["before"])
        after = self._rgb(self.files[sample_id]["after"])
        mask = self._change(sample_id)
        if before.shape != after.shape or before.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"SECOND images and labels are not aligned: {sample_id}")
        sample: dict[str, Any] = {
            "before": before, "after": after, "mask": mask, "sample_id": sample_id,
        }
        return self.transform(sample) if self.transform is not None else sample
