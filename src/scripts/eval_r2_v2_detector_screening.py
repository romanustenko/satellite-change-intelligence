"""Detector screening evaluation on SECOND validation (plan VS, tasks T05/T07).

Three subcommands (docs/research2/DETECTOR_SCREENING_PLAN.md §3):

``infer``      fine-tuned detector -> probability maps on the 444 validation tiles -> per-tile ROI
               count tables for the whole 324-setting generator grid (M1-M6 inputs) and detector
               diagnostics (M8). Writes ``artifacts/research2/v2/screening/<det>_s<seed>/``.
``calibrate``  noise of FMR and of the small-stratum coverage on Light only -> writes the frozen
               rule ``configs/SCREENING_RULE.json`` (it must be committed before any other detector
               is evaluated).
``report``     paired tile bootstrap over all detectors -> tables + eligibility by the frozen rule.

``infer`` for any detector other than ``light_full`` refuses to run until the rule file exists and
is tracked by git: the ROI metrics of Medium/Heavy must not influence the rule. The SECOND test
split is never read.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.execution import resolve_device
from satchange.models.factory import BINARY_TASK, create_task_model
from satchange.semantic import roi_screening as rs

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/SECOND"
RUNS = ROOT / "artifacts/research2/v2/runs"
OUT = ROOT / "artifacts/research2/v2/screening"
RULE = ROOT / "configs/SCREENING_RULE.json"
FIRST_DETECTOR = "light_full"
TILE = 512
BOOTSTRAP_SEED = 20261110
ILLUSTRATIVE = {  # fixed operating points shown as diagnostics only (never used by the rule)
    "balanced (t0.5 m16 g32 s64)": rs.ScreeningSettings(0.5, 16, 32, 64),
    "recall-oriented (t0.35 m32 g64 s64)": rs.ScreeningSettings(0.35, 32, 64, 64),
    "one box (t0.35 m16 g=inf s128)": rs.ScreeningSettings(0.35, 16, None, 128),
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()


def _rule_is_committed() -> bool:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(RULE.relative_to(ROOT))],
        capture_output=True, cwd=ROOT, check=False,
    ).returncode == 0
    clean = not _git("status", "--porcelain", "--", str(RULE.relative_to(ROOT)))
    return tracked and clean


def run_directory(group: str, detector: str, seed: int, name: str | None = None) -> Path:
    return RUNS / group / detector / (name or f"{detector}_s{seed}")


def _load_detector(
    group: str, detector: str, seed: int, name: str | None = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    run = run_directory(group, detector, seed, name)
    record = json.loads((run / "run_record.json").read_text())
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)  # own file
    weights = state.get("best_model_state_dict") or state["model_state_dict"]
    config = record["config"]
    model = create_task_model(config["init"]["model_name"], config["model_parameters"], BINARY_TASK)
    model.load_state_dict(weights, strict=True)
    return model.eval(), record


@torch.inference_mode()
def _probability_maps(model: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, ...]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, loader = create_dataset_dataloader(DATA, "val", None, 8, 4, pin_memory=True)
    probabilities, masks, ids = [], [], []
    model = model.to(device)
    for batch in loader:
        logits = model(batch["before"].to(device), batch["after"].to(device))
        probabilities.append(logits.float().sigmoid()[:, 0].cpu().numpy())
        masks.append(batch["mask"][:, 0].numpy() > 0.5)
        ids.extend(batch["sample_id"])
    return np.concatenate(probabilities), np.concatenate(masks), np.array(ids)


def _tile_table(args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    cv2.setNumThreads(1)
    probabilities, mask = args
    return rs.scene_table(rs.build_scene_gt(mask), probabilities, rs.settings_grid())


def _diagnostics(probabilities: np.ndarray, masks: np.ndarray, threshold: float) -> dict[str, Any]:
    positive = np.zeros(rs.HISTOGRAM_BINS)
    negative = np.zeros(rs.HISTOGRAM_BINS)
    matched = total = 0
    for probability, mask in zip(probabilities, masks, strict=True):
        rs.update_histograms(positive, negative, probability, mask)
        hit, count = rs.strict_component_matches(
            rs.build_scene_gt(mask), probability >= threshold
        )
        matched, total = matched + hit, total + count
    return {
        "threshold": threshold,
        **rs.confusion_at_threshold(positive, negative, threshold),
        "average_precision": rs.average_precision(positive, negative),
        "strict_component_recall_iou50": matched / max(total, 1),
        "gt_components": total,
    }


def infer(args: argparse.Namespace) -> None:
    if args.detector != FIRST_DETECTOR and not (RULE.is_file() and _rule_is_committed()):
        raise SystemExit(
            f"Refusing to evaluate {args.detector}: {RULE.relative_to(ROOT)} must exist and be "
            "committed first (calibration on Light only, plan VS-T05)."
        )
    model, record = _load_detector(args.group, args.detector, args.seed, args.name)
    started = time.perf_counter()
    probabilities, masks, ids = _probability_maps(model, resolve_device(args.device))
    infer_seconds = time.perf_counter() - started
    with ProcessPoolExecutor(args.workers) as pool:
        tables = list(pool.map(_tile_table, zip(probabilities, masks, strict=True), chunksize=4))
    table = np.stack(tables, axis=1)  # [settings, tiles, fields]
    out = OUT / ("" if args.group == "screening" else f"_{args.group}") / (
        args.name or f"{args.detector}_s{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "table.npz", table=table, ids=ids)
    np.save(out / "probabilities_fp16.npy", probabilities.astype(np.float16))
    diagnostics = _diagnostics(probabilities, masks, record["best_threshold"])
    (out / "diagnostics.json").write_text(json.dumps({
        "detector": args.detector, "seed": args.seed, "tiles": len(ids),
        "checkpoint_epoch": record["best_epoch"], "training_flops": record["training_flops"],
        "optimizer_steps": record["optimizer_steps"], "lr": record["lr"],
        "diagnostics": diagnostics, "inference_seconds": infer_seconds,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "tests")),
    }, indent=2))
    print(json.dumps({"detector": args.detector, "seed": args.seed, **diagnostics}), flush=True)


# --- aggregation ---------------------------------------------------------------------------


def _load_tables(directories: list[Path]) -> list[np.ndarray]:
    detector = directories[0].name
    tables = [np.load(directory / "table.npz")["table"] for directory in directories]
    if len({t.shape for t in tables}) != 1:
        raise ValueError(f"{detector}: tables of different shape across seeds.")
    return tables


def _statistics(tables: list[np.ndarray], resamples: int) -> dict[str, np.ndarray]:
    """Per-seed pooled metrics for the full sample and for ``resamples`` tile bootstraps.

    The same multinomial tile counts are used for every seed and every detector (same bootstrap
    seed, same number of tiles), so differences between detectors are paired.
    """
    tiles = tables[0].shape[1]
    pixels = TILE * TILE
    point = [rs.pooled_metrics(t.sum(axis=1), 1.0 * tiles, pixels) for t in tables]
    boot = [
        rs.pooled_metrics(rs.bootstrap_sums(t, resamples, BOOTSTRAP_SEED), 1.0 * tiles, pixels)
        for t in tables
    ]
    out: dict[str, np.ndarray] = {}
    for key in point[0]:
        out[f"point_{key}"] = np.stack([p[key] for p in point])  # [seeds, settings]
        out[f"boot_{key}"] = np.stack([b[key] for b in boot])  # [seeds, B, settings]
    return out


def _frontier(stats: dict[str, np.ndarray], key: str, budgets: tuple[float, ...]) -> tuple:
    """Frontier of metric ``key`` per budget: point [seeds, nb] and bootstrap [seeds, B, nb]."""
    point = rs.best_under_budget(stats[f"point_{key}"], stats["point_area_ratio"], budgets)
    boot = rs.best_under_budget(stats[f"boot_{key}"], stats["boot_area_ratio"], budgets)
    return point, boot


def detector_summary(
    stats: dict[str, np.ndarray], budgets: tuple[float, ...], small_budget: float
) -> dict[str, Any]:
    """Seed-mean scalars of one detector with their bootstrap distribution (axis ``B``)."""
    recall_point, recall_boot = _frontier(stats, "pixel_recall", budgets)
    small_point, small_boot = _frontier(stats, "small_single95", (small_budget,))
    return {
        "frontier_point": recall_point,  # [seeds, nb]
        "fmr_seed": recall_point.mean(axis=-1),  # [seeds]
        "fmr_boot": recall_boot.mean(axis=-1).mean(axis=0),  # [B] (seed-mean)
        "small_seed": small_point[:, 0],
        "small_boot": small_boot[:, :, 0].mean(axis=0),
    }


def _interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def calibrate(args: argparse.Namespace) -> None:
    tables = _load_tables([OUT / f"{FIRST_DETECTOR}_s{seed}" for seed in args.seeds])
    budgets, small_budget = rs.BUDGETS, 0.5
    summary = detector_summary(_statistics(tables, args.resamples), budgets, small_budget)
    fmr_ci = _interval(summary["fmr_boot"])
    small_ci = _interval(summary["small_boot"])
    parts: dict[str, dict[str, Any]] = {
        "fmr": {
            "mean": float(summary["fmr_seed"].mean()), "ci95": fmr_ci,
            "ci_half_width": (fmr_ci[1] - fmr_ci[0]) / 2,
            "two_sd_over_seeds": float(2 * summary["fmr_seed"].std(ddof=1)),
        },
        "small": {
            "mean": float(summary["small_seed"].mean()), "ci95": small_ci,
            "ci_half_width": (small_ci[1] - small_ci[0]) / 2,
            "two_sd_over_seeds": float(2 * summary["small_seed"].std(ddof=1)),
        },
    }
    rule = {
        "frozen_utc": time.strftime("%FT%TZ", time.gmtime()),
        "calibrated_on": FIRST_DETECTOR, "seeds": args.seeds, "resamples": args.resamples,
        "bootstrap_seed": BOOTSTRAP_SEED, "code_git_revision": _git("rev-parse", "HEAD"),
        "grid": {
            "thresholds": list(rs.THRESHOLDS), "margins": list(rs.MARGINS),
            "merge_gaps": [g if g is not None else "inf" for g in rs.MERGE_GAPS],
            "min_crops": list(rs.MIN_CROPS),
        },
        "budgets": list(budgets), "small_budget": small_budget,
        "small_area_px": rs.SMALL_AREA, "large_area_px": rs.LARGE_AREA,
        "single_crop_coverage": rs.SINGLE_CROP_COVERAGE,
        "noise_components": parts,
        "delta_fmr": (
            max(parts["fmr"]["ci_half_width"], parts["fmr"]["two_sd_over_seeds"])
        ),
        "delta_small": (
            max(parts["small"]["ci_half_width"], parts["small"]["two_sd_over_seeds"])
        ),
        "rule": (
            "recall-eligible: seed-mean FMR >= best FMR - delta_fmr AND seed-mean small-stratum "
            "single-crop coverage at area budget small_budget >= best - delta_small; among "
            "eligible detectors the lowest forward latency (M7) wins, ties -> fewer crops (M4). "
            "Confidence intervals are not equivalence tests; the user makes the decision."
        ),
    }
    RULE.parent.mkdir(parents=True, exist_ok=True)
    RULE.write_text(json.dumps(rule, indent=2) + "\n")
    keys = ("delta_fmr", "delta_small", "noise_components")
    print(json.dumps({k: rule[k] for k in keys}, indent=2))


def report(args: argparse.Namespace) -> None:
    if not (RULE.is_file() and _rule_is_committed()):
        raise SystemExit("SCREENING_RULE.json must exist and be committed before the report.")
    rule = json.loads(RULE.read_text())
    budgets, small_budget = tuple(rule["budgets"]), rule["small_budget"]
    latency = json.loads(Path(args.latency).read_text()) if args.latency else {}
    grid = rs.settings_grid()
    result: dict[str, Any] = {"rule": rule, "detectors": {}}
    summaries: dict[str, dict[str, Any]] = {}
    # Exploratory single-run entries (LABEL=SUBDIR under screening/, e.g. the D-068 extended Heavy
    # run) are reported next to the frozen-rule detectors but never enter the eligibility rule.
    sources = {d: [OUT / f"{d}_s{seed}" for seed in args.seeds] for d in args.detectors}
    for item in args.extra:
        label, _, subdir = item.partition("=")
        sources[label] = [OUT / subdir]
    for detector, directories in sources.items():
        tables = _load_tables(directories)
        stats = _statistics(tables, rule["resamples"])
        summary = detector_summary(stats, budgets, small_budget)
        summaries[detector] = summary
        diagnostics = [
            json.loads((directory / "diagnostics.json").read_text())["diagnostics"]
            for directory in directories
        ]
        entry: dict[str, Any] = {
            "exploratory": detector not in args.detectors,
            "fmr": {"seed_values": summary["fmr_seed"].tolist(),
                    "mean": float(summary["fmr_seed"].mean()),
                    "sd_seeds": (
                        float(summary["fmr_seed"].std(ddof=1)) if len(directories) > 1 else None
                    ),
                    "ci95_tiles": _interval(summary["fmr_boot"])},
            "small_coverage_at_budget": {
                "seed_values": summary["small_seed"].tolist(),
                "mean": float(summary["small_seed"].mean()),
                "ci95_tiles": _interval(summary["small_boot"])},
            "frontier_recall": {
                str(b): float(v) for b, v in zip(budgets, summary["frontier_point"].mean(0),
                                                 strict=True)},
            "m10": {},
            "illustrative_points": {},
            "m8_diagnostics": {
                key: float(np.mean([d[key] for d in diagnostics]))
                for key in ("f1", "iou", "precision", "recall", "average_precision",
                            "strict_component_recall_iou50")
            },
            "latency_ms": latency.get(detector),
        }
        recall, area = stats["point_pixel_recall"], stats["point_area_ratio"]
        crops = stats["point_crops_per_tile"]
        for target in (0.95, 0.98):
            entry["m10"][str(target)] = {
                "min_area_ratio": [float(x) for x in rs.min_cost_at_recall(recall, area, target)],
                "min_crops_per_tile": [
                    float(x) for x in rs.min_cost_at_recall(recall, crops, target)
                ],
            }
        for label, setting in ILLUSTRATIVE.items():
            index = grid.index(setting)
            entry["illustrative_points"][label] = {
                key.removeprefix("point_"): float(np.nanmean(value[:, index]))
                for key, value in stats.items() if key.startswith("point_")
            }
        result["detectors"][detector] = entry
    ruled = {d: s for d, s in summaries.items() if d in args.detectors}
    best_fmr = max(s["fmr_seed"].mean() for s in ruled.values())
    best_small = max(s["small_seed"].mean() for s in ruled.values())
    for detector, summary in ruled.items():
        entry = result["detectors"][detector]
        entry["eligible_fmr"] = bool(summary["fmr_seed"].mean() >= best_fmr - rule["delta_fmr"])
        entry["eligible_small"] = bool(
            summary["small_seed"].mean() >= best_small - rule["delta_small"]
        )
        entry["eligible"] = entry["eligible_fmr"] and entry["eligible_small"]
    names = list(summaries)
    result["paired_differences_fmr"] = {
        f"{a} - {b}": {
            "mean": float(summaries[a]["fmr_seed"].mean() - summaries[b]["fmr_seed"].mean()),
            "ci95_tiles": _interval(summaries[a]["fmr_boot"] - summaries[b]["fmr_boot"]),
        }
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({d: {k: e[k] for k in ("eligible_fmr", "eligible_small", "eligible")}
                      for d, e in result["detectors"].items() if d in ruled}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("infer")
    p.add_argument("--detector", required=True, choices=["light_full", "medium_full", "heavy_full"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--group", default="screening", help="run group under runs/ (default: finals)")
    p.add_argument("--name", help="run directory name if not <detector>_s<seed> (e.g. extended)")
    p.add_argument("--device", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=infer)
    p = sub.add_parser("calibrate")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    p.add_argument("--resamples", type=int, default=2000)
    p.set_defaults(func=calibrate)
    p = sub.add_parser("report")
    p.add_argument("--detectors", nargs="+", default=["light_full", "medium_full", "heavy_full"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    p.add_argument("--extra", nargs="*", default=[], metavar="LABEL=SUBDIR",
                   help="exploratory single runs under screening/, outside the rule")
    p.add_argument("--latency", help="JSON {detector: median forward latency ms}")
    p.add_argument("--output", required=True)
    p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
