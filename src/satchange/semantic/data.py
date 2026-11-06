"""Small, dependency-free I/O helpers for SECOND preparation (hashing, RGB, maps, JSON)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def write_json(path: Path, value: Any) -> None:
    """Save readable JSON with finite numeric values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_digest(path: Path) -> str:
    """Hash source content so a resume cannot silently use changed annotations/images."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: str | Path) -> torch.Tensor:
    """Load an actual RGB uint8 image as native [3,H,W] float32 in [0,1]."""
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB imagery, not {image.mode}: {path}")
        pixels = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255).contiguous()


def read_map(path: str | Path, *, semantic: bool = False) -> torch.Tensor:
    """Read a non-pickled .npy [H,W] map, preserving integer semantic IDs."""
    if Path(path).suffix.lower() != ".npy":
        raise ValueError(f"Maps must be .npy [H,W] arrays: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"Map must have shape [H,W]: {path}")
    if semantic and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Semantic maps require integer IDs: {path}")
    if not semantic and not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"Change maps require numeric probabilities: {path}")
    if not semantic and (
        np.iscomplexobj(array)
        or not np.isfinite(array).all()
        or (array < 0).any()
        or (array > 1).any()
    ):
        raise ValueError(f"Change probabilities must be finite and in [0,1]: {path}")
    if semantic and np.issubdtype(array.dtype, np.unsignedinteger):
        if array.size and array.max() > np.iinfo(np.int64).max:
            raise ValueError(f"Semantic IDs must fit signed int64: {path}")
    # int64 also allows explicit negative ignore IDs without binarizing labels.
    return torch.from_numpy(array.astype(np.int64 if semantic else np.float32, copy=True))
