"""Research 2 system A: single-stage dense semantic change detection on SECOND (D-038).

Thin driver over the shared engine (``fit_binary_change_model(task="semantic_change_detection")``);
no new trainer. FC-Siam-Diff with 14 output channels, optional LEVIR-CD initialization (head
excluded). Validation only; the SECOND test split is never read. Direct runs are recorded in
``<output>/direct_runs.jsonl`` and must be added to ``docs/research2/GPU_LEDGER.md``. Example:

    python src/scripts/run_r2_scd.py --gpu 1 --seed 42 --lr 0.001 --epochs 40 --init levir
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.data.transforms import RandomFlipRotate
from satchange.execution import resolve_device, seed_everything
from satchange.hpo_backend import _load_initial_weights
from satchange.models.factory import SEMANTIC_OUTPUT_CHANNELS, SEMANTIC_TASK, create_task_model
from satchange.semantic.scd_training import NUM_CLASSES, SemanticSCDLoss
from satchange.semantic.second import decode_label
from satchange.training import fit_binary_change_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "satellite-change-intelligence" / "data" / "SECOND"
if not DATA.is_dir():
    DATA = ROOT / "data" / "SECOND"
LEVIR_CHECKPOINT = (
    DATA.parents[1] / "artifacts/final_models/light_full/seed_123_best.pt"
)
OUTPUT = DATA.parents[1] / "artifacts/research2/scd"
MODEL_PARAMETERS = {"in_channels": 3, "out_channels": SEMANTIC_OUTPUT_CHANNELS, "base_channels": 32}


def class_frequencies(dataset: object) -> list[float]:
    """Pixel frequency of each class over T1+T2 train labels; cached (labels are frozen)."""
    cache = OUTPUT / "class_frequency_train.json"
    if cache.is_file():
        return list(json.loads(cache.read_text())["frequency"])
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for files in dataset.files.values():  # type: ignore[attr-defined]
        for key in ("before_semantic", "after_semantic"):
            counts += np.bincount(decode_label(files[key]).ravel(), minlength=NUM_CLASSES)
    frequency = (counts / counts.sum()).tolist()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"counts": counts.tolist(), "frequency": frequency}, indent=2))
    return [float(x) for x in frequency]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--init", choices=("scratch", "levir"), default="levir")
    parser.add_argument("--class-weights", choices=("none", "sqrt_inverse"),
                        default="sqrt_inverse")
    parser.add_argument("--name", required=True, help="run directory under artifacts/research2/scd")
    args = parser.parse_args()

    device = resolve_device(f"cuda:{args.gpu}")
    seed_everything(args.seed)
    transform = RandomFlipRotate(extra_keys=("before_label", "after_label"))
    train_data, train_loader = create_dataset_dataloader(
        DATA, "train", None, args.batch_size, args.num_workers, transform=transform,
        shuffle=True, pin_memory=True, seed=args.seed, semantic_labels=True,
    )
    _, validation_loader = create_dataset_dataloader(
        DATA, "val", None, args.batch_size, args.num_workers, pin_memory=True,
        semantic_labels=True,
    )
    weights = None
    if args.class_weights == "sqrt_inverse":
        frequency = np.maximum(np.array(class_frequencies(train_data)), 1e-6)
        raw = frequency ** -0.5
        weights = (raw / raw.mean()).tolist()
    model = create_task_model("fc_siam_diff", MODEL_PARAMETERS, SEMANTIC_TASK)
    initialization = None
    if args.init == "levir":
        spec = {
            "path": str(LEVIR_CHECKPOINT), "exclude": ["output_layer"],
            "sha256": hashlib.sha256(LEVIR_CHECKPOINT.read_bytes()).hexdigest(),
        }
        initialization = _load_initial_weights(model, spec)
    model.to(device)
    loss_function = SemanticSCDLoss(weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    config = {
        "task": SEMANTIC_TASK, "model_parameters": MODEL_PARAMETERS, "seed": args.seed,
        "learning_rate": args.lr, "batch_size": args.batch_size, "init": args.init,
        "class_weights": weights, "epochs": args.epochs, "initial_weights": initialization,
    }
    started = time.perf_counter()
    history = fit_binary_change_model(
        model=model, train_loader=train_loader, validation_loader=validation_loader,
        loss_function=loss_function, optimizer=optimizer, device=device,
        number_of_epochs=args.epochs, thresholds=[], output_root=OUTPUT, model_name="fc_siam_diff",
        run_name=args.name, checkpoint_interval=5, config=config, task=SEMANTIC_TASK,
    )
    seconds = time.perf_counter() - started
    best = max(history, key=lambda row: row["selection_score"])
    record = {
        "name": args.name, "gpu": args.gpu, "seed": args.seed, "lr": args.lr, "init": args.init,
        "class_weights": args.class_weights, "epochs": len(history), "wall_hours": seconds / 3600,
        "gpu_hours": seconds / 3600, "best_epoch": best["epoch"], "best_sek": best["sek"],
        "best_miou": best["miou"], "best_fscd": best["fscd"],
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    with (OUTPUT / "direct_runs.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
