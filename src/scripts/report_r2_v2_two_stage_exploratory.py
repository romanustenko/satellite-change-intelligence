"""Markdown comparison table from the JSON files of eval_r2_v2_two_stage_exploratory.py."""

import argparse
import json
import statistics as st
from pathlib import Path

NAMES = {"siam_unet_resnet18": "Siamese U-Net R18", "scannet": "SCanNet", "mamba_scd": "MambaSCD-T"}
SEEDS = (42, 123, 777)
ROWS = (
    ("two-stage, one box (t0.35, 1 crop)", "one box", "fixed"),
    ("two-stage, recall-oriented", "recall-oriented", "fixed"),
    ("two-stage, balanced", "balanced", "fixed"),
    ("two-stage, area budget <= 0.7", "frontier0.7", "frontier"),
    ("two-stage, area budget <= 0.5", "frontier0.5", "frontier"),
    ("two-stage, area budget <= 0.3", "frontier0.3", "frontier"),
)


def _mean(system: dict, seeds: tuple[int, ...], key: str, policy: str) -> float:
    return st.mean(system[f"medium_s{s}|{policy}"][key] for s in seeds)


def table(directory: Path, checkpoint_set: str) -> str:
    fixed = json.load(open(directory / f"two_stage_vs_single_{checkpoint_set}.json"))
    frontier_path = directory / f"two_stage_vs_single_frontier_{checkpoint_set}.json"
    frontier = json.load(open(frontier_path)) if frontier_path.is_file() else None
    detector_ms = fixed["detector_latency_ms_512"]
    lines = [
        "| SCD model | System | SeK | dSeK vs single (CI95, seed 42) | mIoU | Fscd | area | crops/tile "
        "| latency ms (SCD stage + detector) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, entry in fixed["models"].items():
        single = entry["systems"]["single"]
        metrics = single["metrics"]
        lines.append(
            f"| {NAMES[model]} (`{entry['checkpoint'].split('/')[-1]}`) | single-stage "
            f"| **{metrics['sek']:.4f}** | — | {metrics['miou']:.4f} | {metrics['fscd']:.4f} | 1.00 | 1 "
            f"| {single['scd_pipeline_ms_median']:.1f} |"
        )
        for label, policy, source in ROWS:
            data = fixed if source == "fixed" else frontier
            if data is None:
                continue
            systems = data["models"][model]["systems"]
            sek = [systems[f"medium_s{s}|{policy}"]["metrics"]["sek"] for s in SEEDS]
            low, high = systems[f"medium_s42|{policy}"]["delta_sek_vs_single_ci95"]
            scd_ms = _mean(systems, SEEDS, "scd_pipeline_ms_median", policy)
            miou = st.mean(systems[f"medium_s{s}|{policy}"]["metrics"]["miou"] for s in SEEDS)
            fscd = st.mean(systems[f"medium_s{s}|{policy}"]["metrics"]["fscd"] for s in SEEDS)
            lines.append(
                f"| | {label} | {st.mean(sek):.4f} | {st.mean(sek) - metrics['sek']:+.4f} "
                f"({low:+.4f}…{high:+.4f}) | {miou:.4f} | {fscd:.4f} "
                f"| {_mean(systems, SEEDS, 'area_ratio', policy):.2f} "
                f"| {_mean(systems, SEEDS, 'crops_per_tile', policy):.2f} "
                f"| {scd_ms:.1f} + {detector_ms:.1f} = {scd_ms + detector_ms:.1f} |"
            )
        oracle = entry["systems"]["oracle|balanced"]["metrics"]["sek"]
        lines.append(f"| | *oracle ROI (GT mask, balanced) — upper bound* | *{oracle:.4f}* | "
                     f"*{oracle - metrics['sek']:+.4f}* | | | | | |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--checkpoint-set", default="best")
    arguments = parser.parse_args()
    print(table(arguments.directory, arguments.checkpoint_set))
