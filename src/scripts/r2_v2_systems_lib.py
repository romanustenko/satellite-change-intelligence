"""Shared pieces of the evaluation of the 12 systems (R2-V9, D-074/D-075).

Case 1 = single-stage SCD on the whole tile. Cases 2-4 = two-stage: a SECOND-trained detector gives
a probability map -> ROI boxes (screening generator) -> the *same* case-1 SCD checkpoint on every
box via the ``batch_ctx`` path (D-074: crops replaced by 128/256/384/512 context windows cut from
the
tile, batched; only the original box is merged back) -> outside every ROI = no change.

* case 2 detector ``medium_ft``  : BIT fine-tuned from the Research 1 (LEVIR-CD) checkpoint
* case 3 detector ``medium_scratch``: BIT random init, SECOND only
* case 4 detector ``or``         : soft-OR = pixel-wise max of the ``medium_ft`` map and the map of
  the partner chosen on validation (``light_full`` or ``light_resized``), swept at one common
  threshold by the ROI generator.

Seed pairing: the two-stage system of SCD seed s uses detector seed s. The test split is read
only by
``eval_r2_v2_systems.py --split test`` after a freeze (single test claim); nothing here opens it.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_r2_v2_detector_screening as scr  # noqa: E402

from satchange.data.levir_cd import create_dataset_dataloader  # noqa: E402
from satchange.semantic import roi_screening as rs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/SECOND"
RUNS = ROOT / "artifacts/research2/v2/runs"
V9 = ROOT / "artifacts/research2/v2/v9"
SEEDS = (42, 123, 777)
SCD_MODELS = ("siam_unet_resnet18", "scannet", "mamba_scd")
SHORT = {"siam_unet_resnet18": "unet", "scannet": "scannet", "mamba_scd": "mamba"}
TILE = 512
BUCKETS = (128, 256, 384, 512)
# detector variant -> (run group, detector name, run name template, input size or None = native)
DETECTOR_RUNS: dict[str, tuple[str, str, str, int | None]] = {
    "medium_ft": ("screening", "medium_full", "medium_full_s{seed}", None),
    "medium_scratch": ("v9_medium_scratch", "medium_full", "medium_scratch_s{seed}", None),
    "light_full": ("screening", "light_full", "light_full_s{seed}", None),
    "light_resized": ("v9_light_resized", "light_resized", "light_resized_s{seed}", 256),
}
PARTNER_CANDIDATES = ("light_full", "light_resized")
CASES = {"case2": "medium_ft", "case3": "medium_scratch", "case4": "or"}
# ROI policies compared for every two-stage system: one fixed near-full-coverage setting and three
# area-budget frontier settings chosen on validation per case (``select`` in prepare_r2_v2_val.py).
FIXED_POLICY = {"balanced": rs.ScreeningSettings(0.5, 16, 32, 64)}
FRONTIER_BUDGETS = (0.7, 0.5, 0.3)


def scd_run_dir(model: str, seed: int) -> Path:
    return RUNS / "final" / model / f"{SHORT[model]}_final_s{seed}"


def map_path(split: str, variant: str, seed: int) -> Path:
    return V9 / "maps" / split / f"{variant}_s{seed}.npy"


def or_map(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Soft-OR of two probability maps: pixel-wise maximum (equals OR of the two binary masks at any
    common threshold)."""
    if first.shape != second.shape:
        raise ValueError("OR fusion needs maps of one shape.")
    return np.maximum(first, second)


def window_bucket(box: list[int]) -> int:
    """Context-window side for a box (smallest bucket covering its longer side)."""
    size = max(box[2] - box[0], box[3] - box[1])
    return next(bucket for bucket in BUCKETS if bucket >= size)


def processed_window_area(boxes: np.ndarray) -> int:
    """Pixels the SCD model actually sees on a tile under ``batch_ctx`` (sum of window areas)."""
    return int(sum(window_bucket(box) ** 2 for box in boxes.tolist()))


@torch.inference_mode()
def compute_detector_maps(
    variant: str, seed: int, split: str, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fp16 probability maps ``[N,512,512]`` (aligned to the native grid), binary masks and ids."""
    group, detector, template, input_size = DETECTOR_RUNS[variant]
    model, _ = scr._load_detector(group, detector, seed, template.format(seed=seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, loader = create_dataset_dataloader(DATA, split, None, 8, 4, pin_memory=True)
    model = model.to(device)
    probabilities, masks, ids = [], [], []
    for batch in loader:
        before, after = batch["before"].to(device), batch["after"].to(device)
        if input_size is not None:  # Resized regime: bilinear RGB resize, bilinear probability back
            before, after = (
                F.interpolate(x, size=(input_size, input_size), mode="bilinear",
                              align_corners=False) for x in (before, after)
            )
        probability = model(before, after).float().sigmoid()
        if input_size is not None:
            probability = F.interpolate(probability, size=(TILE, TILE), mode="bilinear",
                                        align_corners=False).clamp(0, 1)
        probabilities.append(probability[:, 0].cpu().numpy())
        masks.append(batch["mask"][:, 0].numpy() > 0.5)
        ids.extend(batch["sample_id"])
    return np.concatenate(probabilities).astype(np.float16), np.concatenate(masks), np.array(ids)


def load_maps(split: str, variant: str, seed: int, partner: str | None = None) -> np.ndarray:
    """Saved maps of a detector variant (``or`` combines ``medium_ft`` with ``partner``)."""
    if variant == "or":
        if partner is None:
            raise ValueError("The OR variant needs the case-4 partner.")
        first = np.load(map_path(split, "medium_ft", seed), mmap_mode="r")
        second = np.load(map_path(split, partner, seed), mmap_mode="r")
        return or_map(np.asarray(first), np.asarray(second))
    return np.load(map_path(split, variant, seed), mmap_mode="r")


def partner_name() -> str:
    path = V9 / "case4_partner.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run prepare_r2_v2_val.py partner first.")
    return str(json.loads(path.read_text())["partner"])


def policies_for(case: str) -> dict[str, rs.ScreeningSettings]:
    """Declared ROI settings of one case from ``roi_policies.json`` (selected on validation)."""
    chosen = json.loads((V9 / "roi_policies.json").read_text())[case]
    return {name: rs.ScreeningSettings(**values) for name, values in chosen.items()}


def settings_to_json(settings: rs.ScreeningSettings) -> dict[str, Any]:
    return {"threshold": settings.threshold, "margin": settings.margin,
            "merge_gap": settings.merge_gap, "min_crop": settings.min_crop}
