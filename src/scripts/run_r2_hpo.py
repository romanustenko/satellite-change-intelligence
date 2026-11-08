"""Research 2 WHERE-detector HPO on SECOND binary change (D-035), LEVIR-CD initialized.

Thin wrapper over ``run_automated_parameter_search`` (no new trainer). Phases ``hpo`` ->
``seeds`` -> ``final`` are run one at a time so the budget and the surviving configuration can
be inspected. Validation only; the SECOND test split is never read. Example:

    python src/scripts/run_r2_hpo.py --model light --phase hpo --gpus 1 2 3
"""

import argparse
import hashlib
import json
from pathlib import Path

from satchange.automated_hpo import run_automated_parameter_search

ROOT = Path(__file__).resolve().parents[2]
LEVIR_CHECKPOINTS = {
    "light": ROOT / "artifacts/final_models/light_full/seed_123_best.pt",
    "medium": ROOT / "artifacts/final_models/medium_full/seed_123_best.pt",
}
MODEL_NAMES = {"light": "fc_siam_diff", "medium": "bit"}
MODEL_PARAMETERS = {
    "light": {"in_channels": 3, "out_channels": 1, "base_channels": 32},
    "medium": {
        "in_channels": 3, "out_channels": 1, "embedding_dim": 64, "token_length": 4, "num_heads": 8,
        "encoder_head_dim": 64, "decoder_head_dim": 64, "encoder_depth": 1, "decoder_depth": 8,
        "dropout": 0.0, "pretrained_backbone": False,
    },
}
STAGE_EPOCHS = {"light": (30, 40, 60), "medium": (20, 30, 40)}
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
BASE = {
    "image_size": None, "batch_size": 16, "num_workers": 8, "learning_rate": 0.001,
    "weight_decay": 0.0001, "scheduler": "none", "positive_weight": 10,
    "bce_weight": 0.3, "dice_weight": 0.7, "precision": "fp32", "seed": 42,
}
BLOCKS = {
    "light": [
        ("01_learning_rate", {"learning_rate": [0.0003, 0.001, 0.003]}),
        ("02_positive_weight", {"positive_weight": [3, 10, 30]}),
    ],
    # Medium (D-037): learning-rate block only; positive weight fixed to the Light winner (D-036).
    "medium": [("01_learning_rate", {"learning_rate": [0.0001, 0.0003, 0.001]})],
}
BASE_OVERRIDES = {"light": {}, "medium": {"learning_rate": 0.0003, "positive_weight": 3}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), default="light")
    parser.add_argument("--phase", choices=("hpo", "seeds", "final"), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--gpu-hour-budget", type=float, default=8.0)
    parser.add_argument("--f1-gap", type=float, default=0.04, help="seeds phase: survivor F1 gap")
    parser.add_argument("--iou-gap", type=float, default=0.05, help="seeds phase: survivor IoU gap")
    parser.add_argument("--reuse-from", type=Path, default=None, help="earlier search directory")
    parser.add_argument("--stage-epochs", type=int, nargs=3, default=None,
                        help="absolute epoch targets for hpo/seeds/final (default per model)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = LEVIR_CHECKPOINTS[args.model]
    parameters = {**BASE, **BASE_OVERRIDES[args.model]}
    parameters["model_parameters"] = MODEL_PARAMETERS[args.model]
    parameters["initial_weights"] = {
        "path": str(checkpoint), "exclude": [],
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    report = run_automated_parameter_search(
        stages=BLOCKS[args.model], base_parameters=parameters, model_name=MODEL_NAMES[args.model],
        dataset_root=ROOT / "data/SECOND", thresholds=THRESHOLDS,
        stage_epochs=tuple(args.stage_epochs or STAGE_EPOCHS[args.model]),
        output_root=ROOT / "artifacts/research2/where_hpo",
        search_name=f"r2_where_{args.model}_{args.version}", stage=args.phase,
        gpu_ids=args.gpus, checkpoint_interval=5, patience=None,
        gpu_hour_budget=args.gpu_hour_budget, f1_gap=args.f1_gap, iou_gap=args.iou_gap,
        reuse_from=args.reuse_from, dry_run=args.dry_run, verbose=True,
    )
    print(json.dumps({k: report[k] for k in ("complete", "paused") if k in report}))


if __name__ == "__main__":
    main()
