"""Research 2 v2: train one SCD model under the shared recipe and a training-FLOP budget.

Recipe (DECISION_LOG D-049, D-054..D-057): SECOND 512x512 tiles, batch 8, flips/rot90, ImageNet
input normalization and ImageNet encoder init, AdamW with 5 % warm-up + cosine per optimizer step,
the model's own loss, ``land_cover`` decoding, fp32 with TF32 off. The FLOP budget is converted to
an optimizer-step count from a real counted step (SCanNet: teacher passes charged, D-056);
training stops on that count, selection on validation SeK. Train and validation splits only; the
test split is never read. Every run writes ``run_record.json`` and appends to ``ledger.jsonl``.

Usage: python src/scripts/run_r2_v2_scd.py --model scannet --flop-budget 1e16 --lr 3e-4 \
    --weight-decay 0.01 --seed 42 --device cuda:0 --group pilot --name scannet_1e16_lr3e-4
"""

import argparse
import functools
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
from satchange.lr_schedule import WARMUP_FRACTION, warmup_cosine_factor
from satchange.models.factory import SEMANTIC_TASK, create_task_model
from satchange.models.mamba_scd import load_vmamba_encoder_weights
from satchange.models.scannet import load_resnet34_encoder_weights
from satchange.models.siam_unet_resnet import load_resnet_encoder_weights
from satchange.semantic.compute import count_training_step_flops, steps_for_budget
from satchange.semantic.scd_losses import (
    SCanNetPseudoLabeler,
    create_scd_loss,
    pseudo_label_budget_steps,
    pseudo_label_start_step,
)
from satchange.training import fit_binary_change_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/SECOND"
PRETRAINED = ROOT / "artifacts/research2/v2/pretrained"
RUNS = ROOT / "artifacts/research2/v2/runs"
LEDGER = ROOT / "artifacts/research2/v2/ledger.jsonl"
BATCH_SIZE = 8
SEMANTIC_ARGMAX = "land_cover"
MODELS: dict[str, dict[str, Any]] = {
    "siam_unet_resnet18": {
        "parameters": {"head_layout": "multi_head"},
        "encoder": (
            PRETRAINED / "torchvision/resnet18-f37072fd.pth",
            "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            load_resnet_encoder_weights,
        ),
    },
    "scannet": {
        "parameters": {},
        "encoder": (
            PRETRAINED / "torchvision/resnet34-b627a593.pth",
            "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
            load_resnet34_encoder_weights,
        ),
    },
    "mamba_scd": {
        "parameters": {},
        "encoder": (
            PRETRAINED / "vmamba/vssm_tiny_0230_ckpt_epoch_262.pth",
            "09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201",
            load_vmamba_encoder_weights,
        ),
    },
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.strip()


def build_model(name: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Model with its ImageNet encoder loaded (report returned for the run record)."""
    spec = MODELS[name]
    model = create_task_model(name, spec["parameters"], SEMANTIC_TASK)
    path, digest, loader = spec["encoder"]
    report = loader(model, path, expected_sha256=digest)
    return model.to(device), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument(
        "--flop-budget", type=float, required=True, help="training FLOPs, e.g. 1e16"
    )
    parser.add_argument("--lr", type=float, required=True, help="peak learning rate")
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", required=True, help="torch device, e.g. cuda:1")
    parser.add_argument(
        "--group", required=True, help="directory under artifacts/research2/v2/runs"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = resolve_device(args.device)
    # fp32 everywhere, TF32 explicitly off (training precision is a protocol item).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(args.seed)
    transform = RandomFlipRotate(extra_keys=("before_label", "after_label"))
    _, train_loader = create_dataset_dataloader(
        DATA,
        "train",
        None,
        BATCH_SIZE,
        args.num_workers,
        transform=transform,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        seed=args.seed,
        semantic_labels=True,
        input_normalization="imagenet",
    )
    _, validation_loader = create_dataset_dataloader(
        DATA,
        "val",
        None,
        BATCH_SIZE,
        args.num_workers,
        pin_memory=True,
        semantic_labels=True,
        input_normalization="imagenet",
    )
    model, encoder_report = build_model(args.model, device)
    loss_function = create_scd_loss(args.model).to(device)

    # Counted step: a real batch, the model's own loss (D-048); before any optimizer step.
    batch = next(iter(train_loader))
    before, after = batch["before"].to(device), batch["after"].to(device)
    labels = batch["before_label"].to(device), batch["after_label"].to(device)
    step_flops = count_training_step_flops(
        model, [before, after], lambda out: loss_function(out, *labels)
    )
    labeler: SCanNetPseudoLabeler | None = None
    teacher_step_flops = 0.0
    if args.model == "scannet":
        probe = SCanNetPseudoLabeler(start_step=1)
        teacher_step_flops = probe.count_flops(model, before, after) * BATCH_SIZE
        steps = pseudo_label_budget_steps(
            args.flop_budget, step_flops=step_flops.total, teacher_step_flops=teacher_step_flops
        )
        labeler = SCanNetPseudoLabeler(
            start_step=pseudo_label_start_step(steps), flops_per_sample=probe.flops_per_sample
        )
    else:
        steps = steps_for_budget(args.flop_budget, step_flops.total)
    steps_per_epoch = len(train_loader)
    epochs = math.ceil(steps / steps_per_epoch)
    del batch
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        "task": SEMANTIC_TASK,
        "model_parameters": MODELS[args.model]["parameters"],
        "recipe": {
            "input_size": 512,
            "batch_size": BATCH_SIZE,
            "input_normalization": "imagenet",
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "schedule": f"warmup {WARMUP_FRACTION} + cosine per step",
            "precision": "fp32 (TF32 off)",
            "semantic_argmax": SEMANTIC_ARGMAX,
            "augmentation": "flips + rot90 (train only)",
        },
        "flop_budget": args.flop_budget,
        "optimizer_steps": steps,
        "seed": args.seed,
        "step_flops": step_flops.to_record(),
        "teacher_step_flops": teacher_step_flops,
        "pseudo_label_start_step": None if labeler is None else labeler.start_step,
        "encoder_init": {k: encoder_report[k] for k in ("path", "sha256") if k in encoder_report},
    }
    started = time.perf_counter()
    history = fit_binary_change_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        number_of_epochs=epochs,
        thresholds=[],
        output_root=RUNS / args.group,
        model_name=args.model,
        run_name=args.name,
        config=config,
        task=SEMANTIC_TASK,
        max_optimizer_steps=steps,
        flops_per_sample=step_flops.per_sample,
        semantic_argmax=SEMANTIC_ARGMAX,
        pseudo_labeler=labeler,
        lr_factor=functools.partial(warmup_cosine_factor, total_steps=steps),
    )
    seconds = time.perf_counter() - started
    best = max(history, key=lambda row: row["selection_score"])
    last = history[-1]
    record = {
        "group": args.group,
        "name": args.name,
        "model": args.model,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "flop_budget": args.flop_budget,
        "optimizer_steps": last["optimizer_steps"],
        "epochs_run": len(history),
        "samples_seen": last["samples_seen"],
        "training_flops": last["training_flops"],
        "teacher_flops": last.get("teacher_flops", 0.0),
        "step_flops": step_flops.total,
        "best_epoch": best["epoch"],
        "best_sek": best["sek"],
        "best_miou": best["miou"],
        "best_fscd": best["fscd"],
        "final_sek": last["sek"],
        "curve": [
            {
                "epoch": r["epoch"],
                "steps": r["optimizer_steps"],
                "training_flops": r["training_flops"],
                "sek": r["sek"],
                "miou": r["miou"],
                "fscd": r["fscd"],
                "train_loss": r["train_loss"],
            }
            for r in history
        ],
        "wall_seconds": seconds,
        "gpu": torch.cuda.get_device_name(device),
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "tests")),
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    run_dir = RUNS / args.group / args.model / args.name
    (run_dir / "run_record.json").write_text(json.dumps({**record, "config": config}, indent=2))
    with LEDGER.open("a") as handle:
        handle.write(json.dumps({k: v for k, v in record.items() if k != "curve"}) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
