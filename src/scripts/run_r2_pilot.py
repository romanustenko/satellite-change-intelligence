"""Research 2 pilot launcher (H-D, D-024a): FC-Siam-Diff on SECOND binary change, scratch vs LEVIR-CD init.

Thin wrapper over ``run_automated_parameter_search`` (no new trainer). Exactly one search per
(condition, seed); the conditions of one seed differ ONLY in ``initial_weights``. Validation only,
no test split is read. Example:

    python src/scripts/run_r2_pilot.py --condition scratch --seed 42 --gpu 1
"""

import argparse
import hashlib
import json
from pathlib import Path

from satchange.automated_hpo import run_automated_parameter_search

ROOT = Path(__file__).resolve().parents[2]
LEVIR_DIR = ROOT / "artifacts/final_models/light_full"
# Each SECOND seed is paired with a distinct LEVIR-CD seed checkpoint (all three get used once, so
# LEVIR checkpoint variance is part of the transfer condition, not hidden behind one lucky seed).
LEVIR_SEED_FOR = {42: 2024, 123: 123, 777: 777}
EPOCHS = 40
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
BASE = {
    "model_parameters": {"in_channels": 3, "out_channels": 1, "base_channels": 32},
    "image_size": None, "batch_size": 16, "num_workers": 8, "learning_rate": 0.001,
    "weight_decay": 0.0001, "scheduler": "none", "positive_weight": 10,
    "bce_weight": 0.3, "dice_weight": 0.7, "precision": "fp32",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("scratch", "levir"), required=True)
    parser.add_argument("--seed", type=int, choices=sorted(LEVIR_SEED_FOR), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--gpu-hour-budget", type=float, default=2.5)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    parameters = dict(BASE, seed=args.seed)
    if args.condition == "levir":
        checkpoint = LEVIR_DIR / f"seed_{LEVIR_SEED_FOR[args.seed]}_best.pt"
        parameters["initial_weights"] = {
            "path": str(checkpoint), "exclude": [],
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    report = run_automated_parameter_search(
        stages=[("pilot", {"learning_rate": [BASE["learning_rate"]]})],
        base_parameters=parameters, model_name="fc_siam_diff",
        dataset_root=ROOT / "data/SECOND", thresholds=THRESHOLDS,
        stage_epochs=[args.epochs] * 3, additional_seeds=False,
        output_root=ROOT / "artifacts/research2/pilot",
        search_name=f"r2_pilot_{args.condition}_s{args.seed}_{args.version}",
        gpu_ids=[args.gpu], checkpoint_interval=5, patience=None,
        gpu_hour_budget=args.gpu_hour_budget, dry_run=args.dry_run, verbose=True,
    )
    print(json.dumps({k: report[k] for k in ("complete", "paused") if k in report}))


if __name__ == "__main__":
    main()
