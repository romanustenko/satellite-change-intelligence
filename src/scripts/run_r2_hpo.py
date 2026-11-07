"""Research 2 WHERE-detector HPO on SECOND binary change (D-035), LEVIR-CD initialized.

Thin wrapper over ``run_automated_parameter_search`` (no new trainer). Phases ``hpo`` -> ``seeds`` ->
``final`` are run one at a time so the budget and the surviving configuration can be inspected. Validation
only; the SECOND test split is never read. Example:

    python src/scripts/run_r2_hpo.py --phase hpo --gpus 1 2 3
"""

import argparse
import hashlib
import json
from pathlib import Path

from satchange.automated_hpo import run_automated_parameter_search

ROOT = Path(__file__).resolve().parents[2]
LEVIR_CHECKPOINT = ROOT / "artifacts/final_models/light_full/seed_123_best.pt"
STAGE_EPOCHS = (30, 40, 60)
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
BASE = {
    "model_parameters": {"in_channels": 3, "out_channels": 1, "base_channels": 32},
    "image_size": None, "batch_size": 16, "num_workers": 8, "learning_rate": 0.001,
    "weight_decay": 0.0001, "scheduler": "none", "positive_weight": 10,
    "bce_weight": 0.3, "dice_weight": 0.7, "precision": "fp32", "seed": 42,
}
BLOCKS = [
    ("01_learning_rate", {"learning_rate": [0.0003, 0.001, 0.003]}),
    ("02_positive_weight", {"positive_weight": [3, 10, 30]}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("hpo", "seeds", "final"), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--gpu-hour-budget", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    parameters = dict(BASE)
    parameters["initial_weights"] = {
        "path": str(LEVIR_CHECKPOINT), "exclude": [],
        "sha256": hashlib.sha256(LEVIR_CHECKPOINT.read_bytes()).hexdigest(),
    }
    report = run_automated_parameter_search(
        stages=BLOCKS, base_parameters=parameters, model_name="fc_siam_diff",
        dataset_root=ROOT / "data/SECOND", thresholds=THRESHOLDS, stage_epochs=STAGE_EPOCHS,
        output_root=ROOT / "artifacts/research2/where_hpo",
        search_name=f"r2_where_light_{args.version}", stage=args.phase,
        gpu_ids=args.gpus, checkpoint_interval=5, patience=None,
        gpu_hour_budget=args.gpu_hour_budget, dry_run=args.dry_run, verbose=True,
    )
    print(json.dumps({k: report[k] for k in ("complete", "paused") if k in report}))


if __name__ == "__main__":
    main()
