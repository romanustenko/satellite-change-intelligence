"""lr probe (B/4) -> best-validation-F1 learning rate -> 3 seeds for one detector variant (V9).

Same rule as the detector screening (D-060/D-062): lr grid {1e-4, 3e-4, 1e-3}, seed 42, final
budget B = 3e16 FLOPs, seeds 42/123/777, selection by validation F1. A finished run (``run_record``
present) is never rerun. Used for the case-3 detector (Medium trained on SECOND only) and the
case-4 partner candidate (Light Resized fine-tuned on SECOND), D-074.

Usage: python src/scripts/run_r2_v2_detector_chain.py --detector light_resized --init research1 \
    --tag light_resized --device cuda:0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/scripts/run_r2_v2_detector.py"
RUNS = ROOT / "artifacts/research2/v2/runs"
LRS = (1e-4, 3e-4, 1e-3)
SEEDS = (42, 123, 777)
BUDGET = 3e16


def run(detector: str, init: str, group: str, name: str, lr: float, seed: int, budget: float,
        device: str, workers: int) -> dict:
    record_path = RUNS / group / detector / name / "run_record.json"
    if not record_path.exists():
        command = [
            sys.executable, str(RUNNER), "--detector", detector, "--init", init, "--flop-budget",
            str(budget), "--lr", str(lr), "--seed", str(seed), "--device", device, "--group", group,
            "--name", name, "--num-workers", str(workers),
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
    return json.loads(record_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--init", choices=("research1", "scratch"), required=True)
    parser.add_argument("--tag", required=True, help="short name used in run names and groups")
    parser.add_argument("--device", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    probe = {}
    for lr in LRS:
        record = run(args.detector, args.init, f"v9_lrprobe_{args.tag}", f"{args.tag}_s42_lr{lr:g}",
                     lr, 42, BUDGET / 4, args.device, args.num_workers)
        probe[lr] = record["best_validation_f1"]
        print(f"lr probe {lr:g}: val F1 {probe[lr]:.4f}", flush=True)
    best_lr = max(probe, key=lambda lr: probe[lr])
    folder = RUNS / f"v9_{args.tag}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "lr_probe.json").write_text(json.dumps(
        {"detector": args.detector, "init": args.init, "budget": BUDGET / 4,
         "val_f1": {str(k): v for k, v in probe.items()}, "selected_lr": best_lr}, indent=2))
    print(f"selected lr {best_lr:g}", flush=True)
    for seed in SEEDS:
        record = run(args.detector, args.init, f"v9_{args.tag}", f"{args.tag}_s{seed}", best_lr,
                     seed, BUDGET, args.device, args.num_workers)
        print(f"final seed {seed}: val F1 {record['best_validation_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
