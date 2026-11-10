"""Analyse Research 2 pilot v1 (docs/research2/archive/v1/PILOT_SPEC.md): scratch vs LEVIR-CD initialization.

Reads ``last.pt`` histories of ``artifacts/research2/pilot/r2_pilot_<cond>_s<seed>_<version>`` (own
trusted checkpoints, CPU), uses validation F1 only, and writes a JSON + Markdown summary. Usage:

    python src/scripts/analyze_r2_pilot.py [version=v1]
"""

import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "artifacts/research2/pilot"
SEEDS = (42, 123, 777)
LEVELS = (0.90, 0.95)


def load_history(search: Path) -> list[dict]:
    files = sorted(search.glob("runs/*/fc_siam_diff/seed_*/last.pt"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one last.pt under {search}, found {len(files)}")
    history: list[dict] = torch.load(files[0], map_location="cpu", weights_only=False)["history"]
    return history


def first_epoch_reaching(curve: list[float], target: float) -> int | None:
    return next((i + 1 for i, value in enumerate(curve) if value >= target), None)


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    runs = {
        (cond, seed): load_history(PILOT / f"r2_pilot_{cond}_s{seed}_{version}")
        for cond in ("scratch", "levir") for seed in SEEDS
    }
    common = min(len(history) for history in runs.values())
    curves = {k: [row["validation_f1"] for row in h[:common]] for k, h in runs.items()}
    seconds = {k: [row["epoch_seconds"] for row in h[:common]] for k, h in runs.items()}
    scratch_final = statistics.median(curves[("scratch", s)][-1] for s in SEEDS)
    report: dict = {
        "version": version, "common_epochs": common, "scratch_median_final_f1": scratch_final,
        "runs": {}, "levels": {},
    }
    for (cond, seed), curve in curves.items():
        report["runs"][f"{cond}_s{seed}"] = {
            "final_f1": curve[-1], "best_f1": max(curve), "best_epoch": curve.index(max(curve)) + 1,
            "f1_epoch1": curve[0], "f1_epoch5": curve[min(4, common - 1)],
            "train_seconds": sum(seconds[(cond, seed)]),
        }
    for level in LEVELS:
        target = level * scratch_final
        rows = {}
        for cond in ("scratch", "levir"):
            per_seed = {}
            for seed in SEEDS:
                epoch = first_epoch_reaching(curves[(cond, seed)], target)
                per_seed[seed] = {
                    "epochs_to_target": epoch,
                    "seconds_to_target": None if epoch is None
                    else sum(seconds[(cond, seed)][:epoch]),
                }
            rows[cond] = per_seed
        wins = 0
        for seed in SEEDS:
            s = rows["scratch"][seed]["epochs_to_target"]
            v = rows["levir"][seed]["epochs_to_target"]
            wins += int((v is not None) and (s is None or v < s))
        gaps = [
            (rows["scratch"][seed]["epochs_to_target"] or common + 1)
            - (rows["levir"][seed]["epochs_to_target"] or common + 1)
            for seed in SEEDS
        ]
        report["levels"][str(level)] = {
            "target_f1": target, "per_condition": rows, "seeds_where_levir_faster": wins,
            "median_epoch_gap_scratch_minus_levir": statistics.median(gaps),
        }
    out = PILOT / f"analysis_{version}.json"
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"common epochs: {common}; scratch median final val F1: {scratch_final:.4f}")
    for key, run in report["runs"].items():
        print(f"{key:12s} f1@1={run['f1_epoch1']:.3f} f1@5={run['f1_epoch5']:.3f} "
              f"best={run['best_f1']:.4f}@{run['best_epoch']} final={run['final_f1']:.4f}")
    for level, row in report["levels"].items():
        gap = row["median_epoch_gap_scratch_minus_levir"]
        print(f"tau={level} x scratch-final ({row['target_f1']:.4f}): levir faster in "
              f"{row['seeds_where_levir_faster']}/3 seeds; median gap {gap}")
    print("wrote", out)


if __name__ == "__main__":
    main()
