"""Research 2 semantic change detection on SECOND (D-038): system A and the WHAT stage of B.

``--stage full`` trains system A on whole scenes; ``--stage what`` trains the WHAT classifier on
fixed windows inside ground-truth ROI crops (``semantic.what_stage``) and then reports scene-level
SeK/mIoU/Fscd with oracle ROIs on the validation split. ``--model`` selects any model registered
for the semantic task.

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
from torch.utils.data import DataLoader

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.data.transforms import RandomFlipRotate
from satchange.execution import resolve_device, seed_everything
from satchange.hpo_backend import _load_initial_weights
from satchange.models.factory import (
    SEMANTIC_MODELS,
    SEMANTIC_OUTPUT_CHANNELS,
    SEMANTIC_TASK,
    create_task_model,
)
from satchange.semantic.scd_training import NUM_CLASSES, SemanticSCDLoss
from satchange.semantic.second import decode_label
from satchange.semantic.what_stage import (
    ROICropDataset,
    evaluate_what_pipeline,
    ground_truth_roi_index,
)
from satchange.training import _best_weights, fit_binary_change_model

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
    parser.add_argument("--model", choices=SEMANTIC_MODELS, default="fc_siam_diff")
    parser.add_argument("--stage", choices=("full", "what"), default="full")
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
        DATA, "train", None, args.batch_size, args.num_workers,
        transform=None if args.stage == "what" else transform,
        shuffle=True, pin_memory=True, seed=args.seed, semantic_labels=True,
    )
    validation_data, validation_loader = create_dataset_dataloader(
        DATA, "val", None, args.batch_size, args.num_workers, pin_memory=True,
        semantic_labels=True,
    )
    if args.stage == "what":
        train_crops = ROICropDataset(
            train_data, ground_truth_roi_index(train_data), random_window=True,
            transform=transform,
        )
        validation_crops = ROICropDataset(
            validation_data, ground_truth_roi_index(validation_data),
        )
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_crops, args.batch_size, shuffle=True, num_workers=args.num_workers,
            pin_memory=True, generator=generator, drop_last=False,
        )
        validation_loader = DataLoader(
            validation_crops, args.batch_size, num_workers=args.num_workers, pin_memory=True,
        )
    weights = None
    if args.class_weights == "sqrt_inverse":
        frequency = np.maximum(np.array(class_frequencies(train_data)), 1e-6)
        raw = frequency ** -0.5
        weights = (raw / raw.mean()).tolist()
    model = create_task_model(args.model, MODEL_PARAMETERS, SEMANTIC_TASK)
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
        "task": SEMANTIC_TASK, "stage": args.stage, "model": args.model,
        "model_parameters": MODEL_PARAMETERS, "seed": args.seed,
        "learning_rate": args.lr, "batch_size": args.batch_size, "init": args.init,
        "class_weights": weights, "epochs": args.epochs, "initial_weights": initialization,
    }
    started = time.perf_counter()
    history = fit_binary_change_model(
        model=model, train_loader=train_loader, validation_loader=validation_loader,
        loss_function=loss_function, optimizer=optimizer, device=device,
        number_of_epochs=args.epochs, thresholds=[], output_root=OUTPUT, model_name=args.model,
        run_name=args.name, checkpoint_interval=5, config=config, task=SEMANTIC_TASK,
    )
    seconds = time.perf_counter() - started
    best = max(history, key=lambda row: row["selection_score"])
    scene_metrics = None
    if args.stage == "what":
        last = OUTPUT / args.model / args.name / "last.pt"
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(_best_weights(state, last))  # best epoch, never the last
        scene_metrics = evaluate_what_pipeline(model, validation_data, device)
    record = {
        "name": args.name, "stage": args.stage, "model": args.model,
        "oracle_scene_metrics": scene_metrics, "gpu": args.gpu, "seed": args.seed,
        "lr": args.lr, "init": args.init,
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
