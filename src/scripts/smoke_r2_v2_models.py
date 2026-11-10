"""R2-V4-T06: GPU smoke test of the three SCD models in the shared engine path.

For each model: ImageNet-initialized encoder, SECOND train/val loaders at 512x512 with ImageNet
input normalization, the model's own loss, ``run_epoch`` with a step budget (SCanNet: pseudo
labels from half the budget on), a short validation pass, FLOPs per training step. Every model is
trained twice from the same seed to measure run-to-run determinism. Training data only (train and
validation splits; the test split is never read). Smoke test: lr and steps are NOT the recipe.

Usage: python src/scripts/smoke_r2_v2_models.py --device cuda:1 --steps 12 --output <dir>
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.data.transforms import RandomFlipRotate
from satchange.execution import resolve_device, seed_everything
from satchange.models.factory import SEMANTIC_TASK, create_task_model
from satchange.models.mamba_scd import load_vmamba_encoder_weights
from satchange.models.scannet import load_resnet34_encoder_weights
from satchange.models.siam_unet_resnet import load_resnet_encoder_weights
from satchange.semantic.compute import count_training_step_flops
from satchange.semantic.scd_losses import (
    SCanNetPseudoLabeler,
    create_scd_loss,
    pseudo_label_start_step,
)
from satchange.semantic.scd_training import run_epoch
from satchange.step_budget import StepBudget

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/SECOND"
PRETRAINED = ROOT / "artifacts/research2/v2/pretrained"
MODELS: dict[str, dict[str, Any]] = {
    "siam_unet_resnet18": {
        "parameters": {"head_layout": "multi_head"},
        "weights": (
            PRETRAINED / "torchvision/resnet18-f37072fd.pth",
            "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            load_resnet_encoder_weights,
        ),
    },
    "scannet": {
        "parameters": {},
        "weights": (
            PRETRAINED / "torchvision/resnet34-b627a593.pth",
            "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
            load_resnet34_encoder_weights,
        ),
    },
    "mamba_scd": {
        "parameters": {},
        "weights": (
            PRETRAINED / "vmamba/vssm_tiny_0230_ckpt_epoch_262.pth",
            "09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201",
            load_vmamba_encoder_weights,
        ),
    },
}


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()


def _build(name: str, device: torch.device, seed: int) -> torch.nn.Module:
    seed_everything(seed)
    spec = MODELS[name]
    model = create_task_model(name, spec["parameters"], SEMANTIC_TASK)
    path, digest, loader = spec["weights"]
    loader(model, path, expected_sha256=digest)
    return model.to(device)


def _loaders(batch_size: int, workers: int, seed: int, validation_tiles: int) -> tuple[Any, Any]:
    _, train = create_dataset_dataloader(
        DATA,
        "train",
        None,
        batch_size,
        workers,
        transform=RandomFlipRotate(extra_keys=("before_label", "after_label")),
        shuffle=True,
        pin_memory=True,
        seed=seed,
        semantic_labels=True,
        input_normalization="imagenet",
        drop_last=True,
    )
    validation_data, _ = create_dataset_dataloader(
        DATA,
        "val",
        None,
        batch_size,
        workers,
        semantic_labels=True,
        input_normalization="imagenet",
    )
    subset = Subset(validation_data, range(validation_tiles))
    return train, DataLoader(subset, batch_size=batch_size, num_workers=workers, pin_memory=True)


def _train_once(name: str, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    model = _build(name, device, args.seed)
    train, validation = _loaders(args.batch_size, args.workers, args.seed, args.validation_tiles)
    loss_function = create_scd_loss(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    labeler = (
        SCanNetPseudoLabeler(start_step=pseudo_label_start_step(args.steps))
        if name == "scannet"
        else None
    )
    budget = StepBudget(args.steps)
    step_seconds: list[float] = []

    torch.cuda.reset_peak_memory_stats(device)
    last = time.perf_counter()
    original_take = budget.take

    def take(samples: int) -> None:
        nonlocal last
        torch.cuda.synchronize(device)
        now = time.perf_counter()
        step_seconds.append(now - last)
        last = now
        original_take(samples)

    budget.take = take  # type: ignore[method-assign, assignment]
    started = time.perf_counter()
    result = run_epoch(
        model,
        train,
        loss_function,
        device,
        optimizer=optimizer,
        budget=budget,
        semantic_argmax="land_cover",
        pseudo_labeler=labeler,
    )
    train_seconds = time.perf_counter() - started
    peak_train = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    evaluation = run_epoch(model, validation, loss_function, device, semantic_argmax="land_cover")
    peak_eval = torch.cuda.max_memory_allocated(device)
    weights = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    return {
        "mean_train_loss": result["loss"],
        "optimizer_steps": budget.optimizer_steps,
        "step_seconds": step_seconds,
        "train_seconds": train_seconds,
        "peak_train_memory_gib": peak_train / 2**30,
        "peak_eval_memory_gib": peak_eval / 2**30,
        "validation": {k: evaluation[k] for k in ("loss", "sek", "miou", "fscd", "samples")},
        "pseudo_labels_active": None if labeler is None else labeler.active,
        "weights": weights,
    }


def _flops(name: str, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    model = _build(name, device, args.seed)
    train, _ = _loaders(args.batch_size, args.workers, args.seed, args.batch_size)
    batch = next(iter(train))
    before, after = batch["before"].to(device), batch["after"].to(device)
    labels = batch["before_label"].to(device), batch["after_label"].to(device)
    loss_function = create_scd_loss(name).to(device)
    record = count_training_step_flops(
        model, [before, after], lambda out: loss_function(out, *labels)
    ).to_record()
    del model
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-tiles", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    # fp32 everywhere; TF32 explicitly off (the training precision is a protocol item).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        runs = [_train_once(name, device, args) for _ in range(2)]
        first, second = runs
        difference = (first["weights"] - second["weights"]).abs()
        timings = first["step_seconds"][2:] or first["step_seconds"]
        record = {
            "task": "R2-V4-T06",
            "model": name,
            "git_revision": _git_revision(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "device": args.device,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "input_size": 512,
            "input_normalization": "imagenet",
            "precision": "fp32 (TF32 off)",
            "lr": args.lr,
            "seed": args.seed,
            "step_flops": _flops(name, device, args),
            "median_step_seconds": statistics.median(timings),
            "run_a": {k: v for k, v in first.items() if k != "weights"},
            "run_b": {k: v for k, v in second.items() if k != "weights"},
            "determinism": {
                "max_abs_weight_difference": float(difference.max()),
                "bitwise_equal_weights": bool(difference.max() == 0),
                "loss_a": first["mean_train_loss"],
                "loss_b": second["mean_train_loss"],
            },
            "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
        }
        path = args.output / f"smoke_{name}.json"
        path.write_text(json.dumps(record, indent=2))
        print(
            json.dumps(
                {k: record[k] for k in ("model", "median_step_seconds", "determinism")}
                | {
                    "peak_train_gib": first["peak_train_memory_gib"],
                    "peak_eval_gib": first["peak_eval_memory_gib"],
                    "val": first["validation"],
                }
            ),
            flush=True,
        )
        del runs, first, second
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
