"""Validation-only preparation of the two-stage systems (R2-V9-T03/T04/T05, D-074/D-075).

``maps``      probability maps of every detector variant and seed on the 444 validation tiles
              (``v9/maps/val/``); the Case-2 maps are checked against the saved screening maps.
``partner``   case-4 partner for Medium among Light Full / Light Resized: the candidate whose
soft-OR
              (pixel-wise max) with Medium raises the best-threshold pixel F1 most (mean over the
              three detector seeds); also reports unique true positives and the precision of what
              the
              partner adds. Metric and fusion rule were declared in D-074/D-075 before any run.
``policies``  ROI settings per case from the 324-setting generator grid: the fixed ``balanced``
              setting plus, for area budgets 0.7 / 0.5 / 0.3, the setting with the highest seed-mean
              pixel recall whose seed-mean processed-area ratio fits the budget.

The SECOND test split is never read here.

Usage: python src/scripts/prepare_r2_v2_val.py {maps,partner,policies} [--device cuda:N]
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import r2_v2_systems_lib as lib
import torch

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.execution import resolve_device
from satchange.semantic import roi_screening as rs

VAL = "val"


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, cwd=lib.ROOT,
                          check=True).stdout.strip()


def make_maps(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    out = lib.V9 / "maps" / VAL
    out.mkdir(parents=True, exist_ok=True)
    reference_ids = reference_masks = None
    for variant in lib.DETECTOR_RUNS:
        for seed in lib.SEEDS:
            path = lib.map_path(VAL, variant, seed)
            if path.exists():
                continue
            probabilities, masks, ids = lib.compute_detector_maps(variant, seed, VAL, device)
            if reference_ids is None:
                reference_ids, reference_masks = ids, masks
            if not np.array_equal(ids, reference_ids) or not np.array_equal(masks, reference_masks):
                raise ValueError(f"{variant} s{seed}: tile order or masks differ")
            np.save(path, probabilities)
            print("saved", path.name, probabilities.shape, flush=True)
    # consistency with the saved screening maps (Case 2 and Light Full were produced earlier)
    for variant, source in (("medium_ft", "medium_full"), ("light_full", "light_full")):
        for seed in lib.SEEDS:
            old = np.load(lib.ROOT / f"artifacts/research2/v2/screening/{source}_s{seed}"
                          "/probabilities_fp16.npy", mmap_mode="r")
            new = np.load(lib.map_path(VAL, variant, seed), mmap_mode="r")
            difference = float(np.abs(old.astype(np.float32) - new.astype(np.float32)).max())
            print(f"{variant} s{seed}: max |new - screening| = {difference:.5f}", flush=True)


def _best_f1(probabilities: np.ndarray, masks: np.ndarray) -> tuple[float, float]:
    best = (0.0, 0.5)
    positives = float(masks.sum())
    for threshold in (round(0.05 * k, 2) for k in range(1, 20)):
        detected = probabilities >= threshold
        tp = float(np.logical_and(detected, masks).sum())
        fp = float(detected.sum()) - tp
        f1 = 2 * tp / max(2 * tp + fp + (positives - tp), 1.0)
        if f1 > best[0]:
            best = (f1, threshold)
    return best


def select_partner(args: argparse.Namespace) -> None:
    masks = None
    report: dict[str, dict] = {}
    for candidate in lib.PARTNER_CANDIDATES:
        rows = []
        for seed in lib.SEEDS:
            medium = np.asarray(np.load(lib.map_path(VAL, "medium_ft", seed))).astype(np.float32)
            other = np.asarray(np.load(lib.map_path(VAL, candidate, seed))).astype(np.float32)
            if masks is None:
                _, loader = create_dataset_dataloader(lib.DATA, VAL, None, 8, 4)
                masks = np.concatenate([b["mask"][:, 0].numpy() > 0.5 for b in loader])
            fused = lib.or_map(medium, other)
            f1_medium, t_medium = _best_f1(medium, masks)
            f1_other, t_other = _best_f1(other, masks)
            f1_or, t_or = _best_f1(fused, masks)
            at = t_or
            m_det, f_det = medium >= at, fused >= at
            unique_tp = float(np.logical_and(np.logical_and(f_det, ~m_det), masks).sum())
            added = float(np.logical_and(f_det, ~m_det).sum())
            rows.append({
                "seed": seed, "f1_medium": f1_medium, "t_medium": t_medium, "f1_partner": f1_other,
                "t_partner": t_other, "f1_or": f1_or, "t_or": t_or, "delta_f1": f1_or - f1_medium,
                "unique_tp_pixels": unique_tp, "added_pixels": added,
                "added_precision": unique_tp / added if added else None,
            })
            print(candidate, rows[-1], flush=True)
        report[candidate] = {
            "per_seed": rows, "mean_delta_f1": float(np.mean([r["delta_f1"] for r in rows])),
            "sd_delta_f1": float(np.std([r["delta_f1"] for r in rows], ddof=1)),
        }
    chosen = max(report, key=lambda name: report[name]["mean_delta_f1"])
    result = {
        "partner": chosen, "rule": "max mean (over 3 seeds) of best-threshold pixel F1 of "
        "max(p_medium_ft, p_candidate) minus that of medium_ft alone, SECOND validation",
        "candidates": report, "git_revision": _git("rev-parse", "HEAD"),
        "created_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    path = lib.V9 / "case4_partner.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; partner selection is decided once")
    path.write_text(json.dumps(result, indent=2))
    print("partner:", chosen, flush=True)


def select_policies(args: argparse.Namespace) -> None:
    partner = lib.partner_name()
    _, loader = create_dataset_dataloader(lib.DATA, VAL, None, 8, 4)
    masks = np.concatenate([b["mask"][:, 0].numpy() > 0.5 for b in loader])
    grid = rs.settings_grid()
    (lib.V9 / "tables").mkdir(parents=True, exist_ok=True)
    chosen: dict[str, dict] = {}
    for case, variant in lib.CASES.items():
        pooled = []
        for seed in lib.SEEDS:
            path = lib.V9 / "tables" / f"{case}_s{seed}.npz"
            if path.exists():
                table = np.load(path)["table"]
            else:
                maps = np.asarray(lib.load_maps(VAL, variant, seed, partner)).astype(np.float32)
                with ProcessPoolExecutor(args.workers) as pool:
                    tiles = list(pool.map(lib.scr._tile_table, zip(maps, masks, strict=True),
                                          chunksize=4))
                table = np.stack(tiles, axis=1)
                np.savez_compressed(path, table=table)
            pooled.append(rs.pooled_metrics(table.sum(axis=1), float(table.shape[1]), 512 * 512))
        recall = np.nanmean([np.nan_to_num(p["pixel_recall"]) for p in pooled], axis=0)
        area = np.mean([p["area_ratio"] for p in pooled], axis=0)
        entry = {"balanced": lib.settings_to_json(lib.FIXED_POLICY["balanced"])}
        stats = {}
        for budget in lib.FRONTIER_BUDGETS:
            score = np.where(area <= budget, recall, -1.0)
            index = int(score.argmax())
            entry[f"frontier{budget}"] = lib.settings_to_json(grid[index])
            stats[f"frontier{budget}"] = {"pixel_recall": float(recall[index]),
                                          "area_ratio": float(area[index])}
        index = grid.index(lib.FIXED_POLICY["balanced"])
        stats["balanced"] = {"pixel_recall": float(recall[index]), "area_ratio": float(area[index])}
        chosen[case] = entry
        print(case, json.dumps(stats), flush=True)
    path = lib.V9 / "roi_policies.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; ROI policies are selected once")
    path.write_text(json.dumps(chosen, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("maps")
    p.add_argument("--device", required=True)
    p.set_defaults(func=make_maps)
    p = sub.add_parser("partner")
    p.set_defaults(func=select_partner)
    p = sub.add_parser("policies")
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(func=select_policies)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.func(args)


if __name__ == "__main__":
    main()
