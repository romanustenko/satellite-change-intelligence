"""Detector screening (plan VS): fine-tune one Research 1 detector on SECOND under a FLOP budget.

The detector starts from its Research 1 deployment checkpoint (strict load, SHA-256 checked) and
is fine-tuned on the SECOND *train* split as binary change detection (``label > 0``); selection is
by validation Pixel F1 over thresholds 0.05..0.95 (the engine's shared binary criterion). Shared
recipe for all candidates (docs/research2/DETECTOR_SCREENING_PLAN.md §2): native 512x512, batch 8,
flips/rot90, [0,1] input without mean/std (as in Research 1), AdamW, 5 % warm-up + cosine per step,
BCE+Dice 0.5/0.5 with pos_weight 5, fp32 with TF32 off; training stops on the step count that
spends the FLOP budget. The test split is never read. Every run writes ``run_record.json`` and
appends to ``ledger.jsonl``.

Usage: python src/scripts/run_r2_v2_detector.py --detector light_full --flop-budget 3e16 \
    --lr 3e-4 --seed 42 --device cuda:0 --group screening --name light_full_s42
"""

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.data.transforms import RandomFlipRotate
from satchange.execution import resolve_device, seed_everything
from satchange.losses import BCEDiceLoss
from satchange.lr_schedule import WARMUP_FRACTION, warmup_cosine_factor
from satchange.models.factory import BINARY_TASK, create_task_model
from satchange.semantic.compute import count_training_step_flops, steps_for_budget
from satchange.training import fit_binary_change_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/SECOND"
FINAL_MODELS = ROOT / "artifacts/final_models"
RUNS = ROOT / "artifacts/research2/v2/runs"
LEDGER = ROOT / "artifacts/research2/v2/ledger.jsonl"
BATCH_SIZE = 8
THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 20)]
LOSS = {"bce_weight": 0.5, "dice_weight": 0.5, "pos_weight": 5.0, "smooth": 1.0}
# Research 1 deployment seed and SHA-256 (= git-LFS oid) of each candidate's checkpoint.
DETECTORS = {
    "light_full": (2024, "12fd0449554cdbddb04258dd0f312a3479768d23fd51d83ca55c9429f13f306c"),
    "medium_full": (123, "8680db04cea68e588217906ba93d78d3422cba9f6b8e11b031b4ce2992bd922a"),
    "heavy_full": (42, "738c757b644174501fcf8584de05db94bcb702d43ff8b2523cbba1b477bd6b42"),
}


class WarmupCosineAdamW(torch.optim.AdamW):
    """AdamW whose learning rate follows the D-057 warm-up + cosine multiplier per optimizer step.

    The shared engine applies a per-step ``lr_factor`` only to the semantic task, so the binary
    screening runs carry the schedule in the optimizer itself (no change to ``training.py``). The
    step counter is not checkpointed: screening runs are never resumed.
    """

    def __init__(self, params: Any, *, total_steps: int, lr: float, weight_decay: float) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.total_steps = total_steps
        self.peak_lrs = [group["lr"] for group in self.param_groups]
        self.step_count = 0

    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        factor = warmup_cosine_factor(self.step_count, total_steps=self.total_steps)
        for group, peak in zip(self.param_groups, self.peak_lrs, strict=True):
            group["lr"] = peak * factor
        self.step_count += 1
        return super().step(closure)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def build_detector(name: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Detector architecture from its Research 1 metadata, weights loaded strictly (CPU)."""
    directory = FINAL_MODELS / name
    metadata = json.loads((directory / "metadata.json").read_text())
    seed, expected = DETECTORS[name]
    checkpoint = directory / f"seed_{seed}_best.pt"
    digest = _sha256(checkpoint)
    if digest != expected:
        raise ValueError(f"{checkpoint} SHA-256 {digest} differs from the recorded {expected}.")
    model = create_task_model(metadata["model_name"], metadata["model_parameters"], BINARY_TASK)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # own, trusted file
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model, {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "sha256": digest,
        "research1_seed": seed,
        "model_name": metadata["model_name"],
        "model_parameters": metadata["model_parameters"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    parser.add_argument("--flop-budget", type=float, required=True, help="training FLOPs")
    parser.add_argument("--lr", type=float, required=True, help="peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", required=True, help="torch device, e.g. cuda:1")
    parser.add_argument("--group", required=True, help="subdirectory of runs/")
    parser.add_argument("--name", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(args.seed)
    _, train_loader = create_dataset_dataloader(
        DATA, "train", None, BATCH_SIZE, args.num_workers, transform=RandomFlipRotate(),
        shuffle=True, pin_memory=True, drop_last=True, seed=args.seed,
    )
    _, validation_loader = create_dataset_dataloader(
        DATA, "val", None, BATCH_SIZE, args.num_workers, pin_memory=True
    )
    model, init_report = build_detector(args.detector)
    model = model.to(device)
    loss_function = BCEDiceLoss(**LOSS).to(device)

    batch = next(iter(train_loader))
    before, after = batch["before"].to(device), batch["after"].to(device)
    mask = batch["mask"].to(device)
    step_flops = count_training_step_flops(
        model, [before, after], lambda out: loss_function(out.float(), mask)
    )
    steps = steps_for_budget(args.flop_budget, step_flops.total)
    epochs = math.ceil(steps / len(train_loader))
    del batch
    torch.cuda.empty_cache()

    optimizer = WarmupCosineAdamW(
        model.parameters(), total_steps=steps, lr=args.lr, weight_decay=args.weight_decay
    )
    config = {
        "task": BINARY_TASK,
        "model_parameters": init_report["model_parameters"],
        "init": init_report,
        "recipe": {
            "input_size": 512, "batch_size": BATCH_SIZE, "input_normalization": "none ([0,1])",
            "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
            "schedule": f"warmup {WARMUP_FRACTION} + cosine per step", "loss": LOSS,
            "precision": "fp32 (TF32 off)", "thresholds": THRESHOLDS,
            "augmentation": "flips + rot90 (train only)",
        },
        "flop_budget": args.flop_budget,
        "optimizer_steps": steps,
        "seed": args.seed,
        "step_flops": step_flops.to_record(),
    }
    started = time.perf_counter()
    history = fit_binary_change_model(
        model=model, train_loader=train_loader, validation_loader=validation_loader,
        loss_function=loss_function, optimizer=optimizer, device=device,
        number_of_epochs=epochs, thresholds=THRESHOLDS, output_root=RUNS / args.group,
        model_name=args.detector, run_name=args.name, config=config,
        max_optimizer_steps=steps, flops_per_sample=step_flops.per_sample,
    )
    seconds = time.perf_counter() - started
    best = max(history, key=lambda row: row["validation_f1"])
    last = history[-1]
    record = {
        "group": args.group, "name": args.name, "model": args.detector, "seed": args.seed,
        "lr": args.lr, "weight_decay": args.weight_decay, "flop_budget": args.flop_budget,
        "optimizer_steps": last["optimizer_steps"], "epochs_run": len(history),
        "samples_seen": last["samples_seen"], "training_flops": last["training_flops"],
        "step_flops": step_flops.total, "best_epoch": best["epoch"],
        "best_validation_f1": best["validation_f1"], "best_threshold": best["best_threshold"],
        "final_validation_f1": last["validation_f1"],
        "curve": [
            {k: r.get(k) for k in ("epoch", "optimizer_steps", "training_flops", "validation_f1",
                                   "best_threshold", "train_loss")}
            for r in history
        ],
        "wall_seconds": seconds, "gpu": torch.cuda.get_device_name(device), "device": args.device,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "tests")),
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    run_dir = RUNS / args.group / args.detector / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_record.json").write_text(json.dumps({**record, "config": config}, indent=2))
    with LEDGER.open("a") as handle:
        handle.write(json.dumps({k: v for k, v in record.items() if k != "curve"}) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
