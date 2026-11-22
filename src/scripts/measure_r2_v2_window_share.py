"""Share of a scene the SCD model must process in a two-stage system: SECOND tiles versus larger
real scenes (LEVIR-CD 1024 x 1024), whole scene versus the same scene cut into 512 tiles.

Exploratory check (user request 2026-11-22; no accuracy, no test split). Boxes come from the frozen
case-2 ``frontier0.5`` ROI settings applied to three sources:

* ``oracle``    : the ground-truth change mask (detector-independent geometry),
* ``medium_ft`` : the frozen case-2 detector (BIT, LEVIR-CD -> SECOND, seed 42),
* ``r1_medium`` : the Research 1 LEVIR-CD detector (BIT medium_full deployment weights; LEVIR only).

Window cost follows ``batch_ctx``: a box gets the smallest 128/256/384/512 window covering its
longer side; a box longer than 512 (possible only on scenes > 512) is counted as a grid of 512
windows. The relative FLOPs of a two-stage system on a scene is
``detector/SCD + sum(window FLOPs)/single-stage FLOPs``, with the single stage run as 512 tiles.

Usage: python src/scripts/measure_r2_v2_window_share.py --device cuda:N
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import r2_v2_systems_lib as lib
import torch

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.execution import resolve_device
from satchange.final_benchmark import _instantiate, _load_checkpoint
from satchange.semantic import roi_screening as rs

LEVIR = lib.ROOT / "data/LEVIR-CD"
OUT = lib.ROOT / "artifacts/research2/v2/window_share"
POLICY = lib.policies_for("case2")["frontier0.5"]
TILE = lib.TILE


def boxes_of(probabilities: np.ndarray) -> np.ndarray:
    detections = rs.detections_by_threshold(probabilities.astype(np.float32), (POLICY.threshold,))
    return rs.generate_boxes(detections, POLICY, *probabilities.shape)


def window_cost(boxes: np.ndarray, flops: dict[str, float]) -> float:
    """SCD FLOPs spent on the windows of one scene."""
    total = 0.0
    for x0, y0, x1, y1 in boxes.tolist():
        side = max(x1 - x0, y1 - y0)
        if side <= TILE:
            total += flops[str(lib.window_bucket([x0, y0, x1, y1]))]
        else:
            total += math.ceil((x1 - x0) / TILE) * math.ceil((y1 - y0) / TILE) * flops[str(TILE)]
    return total


def coverage(boxes: np.ndarray, mask: np.ndarray) -> float:
    """Share of truly changed pixels inside the boxes (1.0 when nothing changed)."""
    inside = np.zeros(mask.shape, dtype=bool)
    for x0, y0, x1, y1 in boxes.tolist():
        inside[y0:y1, x0:x1] = True
    return float((inside & mask).sum() / mask.sum()) if mask.any() else 1.0


def tiles(array: np.ndarray) -> list[np.ndarray]:
    height, width = array.shape
    return [array[y:y + TILE, x:x + TILE] for y in range(0, height, TILE)
            for x in range(0, width, TILE)]


def scene_row(probabilities: np.ndarray, mask: np.ndarray, flops: dict) -> dict:
    """Window cost relative to the single stage, whole scene and as 512 tiles."""
    tiles_count = mask.size / TILE ** 2
    row = {}
    for scd, table in flops["scd"].items():
        single = tiles_count * table[str(TILE)]
        whole = boxes_of(probabilities)
        split = [boxes_of(p) for p in tiles(probabilities)]
        row[scd] = {
            "whole": window_cost(whole, table) / single,
            "tiled": sum(window_cost(b, table) for b in split) / single,
        }
    whole = boxes_of(probabilities)
    row["box_share_whole"] = float(sum((b[2] - b[0]) * (b[3] - b[1]) for b in whole.tolist())
                                   / mask.size)
    row["recall_whole"] = coverage(whole, mask)
    return row


@torch.inference_mode()
def levir_maps(device: torch.device) -> dict[str, list[np.ndarray]]:
    """Detector probability maps on LEVIR-CD val (native 1024) plus the masks."""
    _, loader = create_dataset_dataloader(LEVIR, "val", None, 1, 2)
    detector_ft, _ = lib.scr._load_detector("screening", "medium_full", 42, "medium_full_s42")
    metadata = json.loads((lib.ROOT / "artifacts/final_models/medium_full/metadata.json")
                          .read_text())
    weights = lib.ROOT / "artifacts/final_models/medium_full/deployment_weights.pt"
    state, _ = _load_checkpoint(weights)
    detector_r1 = _instantiate(metadata["model_name"], metadata["model_parameters"], state)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out: dict[str, list[np.ndarray]] = {"mask": [], "medium_ft": [], "r1_medium": []}
    for name, model in (("medium_ft", detector_ft), ("r1_medium", detector_r1)):
        model = model.to(device).eval()
        for batch in loader:
            before, after = batch["before"].to(device), batch["after"].to(device)
            out[name].append(model(before, after).float().sigmoid()[0, 0].cpu().numpy())
            if name == "medium_ft":
                out["mask"].append(batch["mask"][0, 0].numpy() > 0.5)
        model.cpu()
    return out


def summarise(rows: list[dict], density: np.ndarray, flops: dict) -> dict:
    """Mean relative cost per model (detector included) and the density profile."""
    result: dict = {"scenes": len(rows), "density_mean": float(density.mean()),
                    "density_median": float(np.median(density)),
                    "box_share_whole": float(np.mean([r["box_share_whole"] for r in rows])),
                    "recall_whole_pixels": float(np.mean([r["recall_whole"] for r in rows])),
                    "models": {}}
    detector = flops["detectors"]["medium_ft"]["forward_flops"]
    for scd, table in flops["scd"].items():
        tax = detector / table[str(TILE)]
        whole = np.array([r[scd]["whole"] for r in rows])
        tiled = np.array([r[scd]["tiled"] for r in rows])
        result["models"][scd] = {
            "detector_tax": tax,
            "window_share_whole": float(whole.mean()),
            "window_share_tiled": float(tiled.mean()),
            "flops_ratio_whole": float(tax + whole.mean()),
            "flops_ratio_tiled": float(tax + tiled.mean()),
            "cheaper_scenes_whole": float(np.mean(tax + whole < 1)),
        }
    return result


def density_profile(rows: list[dict], density: np.ndarray, flops: dict) -> list[dict]:
    """Relative FLOPs by change-density band (scenes pooled)."""
    detector = flops["detectors"]["medium_ft"]["forward_flops"]
    edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 1.01]
    out = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        chosen = [r for r, d in zip(rows, density, strict=True) if low <= d < high]
        if not chosen:
            continue
        band = {"density": [low, min(high, 1.0)], "scenes": len(chosen)}
        for scd, table in flops["scd"].items():
            tax = detector / table[str(TILE)]
            band[scd] = float(tax + np.mean([r[scd]["whole"] for r in chosen]))
        out.append(band)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    flops = json.loads((lib.V9 / "inference_flops.json").read_text())
    report: dict = {"policy": lib.settings_to_json(POLICY),
                    "flops_source": "v9/inference_flops.json"}

    # SECOND val (512 tiles): saved case-2 detector maps and ground-truth masks
    dataset, _ = create_dataset_dataloader(lib.DATA, "val", None, 1, 0)
    masks = [dataset[i]["mask"][0].numpy() > 0.5 for i in range(len(dataset))]
    maps = np.load(lib.map_path("val", "medium_ft", 42), mmap_mode="r")
    density = np.array([m.mean() for m in masks])
    sources = {"oracle": [m.astype(np.float32) for m in masks],
               "medium_ft": [np.asarray(maps[i], dtype=np.float32) for i in range(len(masks))]}
    for name, probabilities in sources.items():
        rows = [scene_row(p, m, flops) for p, m in zip(probabilities, masks, strict=True)]
        report[f"second_val__{name}"] = summarise(rows, density, flops)
        report[f"second_val__{name}"]["by_density"] = density_profile(rows, density, flops)
        print("SECOND", name, json.dumps(report[f"second_val__{name}"]["models"]), flush=True)

    # LEVIR-CD val (1024 scenes)
    levir = levir_maps(device)
    density = np.array([m.mean() for m in levir["mask"]])
    sources = {"oracle": [m.astype(np.float32) for m in levir["mask"]],
               "medium_ft": levir["medium_ft"], "r1_medium": levir["r1_medium"]}
    for name, probabilities in sources.items():
        rows = [scene_row(p, m, flops) for p, m in zip(probabilities, levir["mask"], strict=True)]
        report[f"levir_val__{name}"] = summarise(rows, density, flops)
        report[f"levir_val__{name}"]["by_density"] = density_profile(rows, density, flops)
        print("LEVIR", name, json.dumps(report[f"levir_val__{name}"]["models"]), flush=True)

    report["git_revision"] = lib.subprocess_git("rev-parse", "HEAD")
    report["gpu"] = torch.cuda.get_device_name(device)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "window_share.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)
    print("written", Path(path).relative_to(lib.ROOT), flush=True)


if __name__ == "__main__":
    main()
