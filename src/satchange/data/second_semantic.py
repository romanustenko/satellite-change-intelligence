"""SECOND dense semantic change detection (Research 2 system A and WHAT-stage labels).

Same manifest-defined samples as ``SECONDBinaryDataset`` (frozen splits, hash-checked files) plus the
decoded T1/T2 class maps. Class ids are canonical: 0 = no change, 1..6 = the SECOND land-cover
classes (``semantic.second.SECOND_CLASSES`` order). A sample adds ``before_label`` and
``after_label`` (int64 ``[H,W]``) to the binary contract; ``mask`` stays the binary change map.
"""

from typing import Any

import numpy as np
import torch
from PIL import Image

from satchange.data.second_binary import SECONDBinaryDataset
from satchange.semantic.second import decode_label

SEMANTIC_CLASS_COUNT = 7  # no change + six SECOND classes
LABEL_KEYS = ("before_label", "after_label")


class SECONDSemanticDataset(SECONDBinaryDataset):
    """One manifest split with dense T1/T2 semantic labels."""

    def _label(self, sample_id: str, key: str) -> torch.Tensor:
        array = decode_label(self.files[sample_id][key])
        if self.image_size is not None:
            image = Image.fromarray(array.astype(np.uint8)).resize(
                (self.image_size, self.image_size), resample=Image.Resampling.NEAREST
            )
            array = np.array(image)
        return torch.from_numpy(array.astype(np.int64))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        before = self._rgb(self.files[sample_id]["before"])
        after = self._rgb(self.files[sample_id]["after"])
        mask = self._change(sample_id)
        labels = {
            name: self._label(sample_id, source)
            for name, source in zip(LABEL_KEYS, ("before_semantic", "after_semantic"), strict=True)
        }
        spatial = before.shape[-2:]
        if after.shape[-2:] != spatial or any(v.shape != spatial for v in labels.values()):
            raise ValueError(f"SECOND images and labels are not aligned: {sample_id}")
        sample: dict[str, Any] = {
            "before": before, "after": after, "mask": mask, **labels, "sample_id": sample_id,
        }
        return self.transform(sample) if self.transform is not None else sample
