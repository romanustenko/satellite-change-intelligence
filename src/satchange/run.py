"""Generic config-driven training, torchrun, independent HPO, and bounded CUDA probe.

No action runs on import. Dataset preparation is deliberately separate.
"""

import argparse
import json
import os
import subprocess
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from satchange.data import RandomFlipRotate, create_dataset_dataloader
from satchange.execution import (
    on_rank_zero,
    precision_context,
    rank,
    seed_everything,
    world_size,
)
from satchange.losses import BCEDiceLoss
from satchange.models import count_model_parameters, create_change_model
from satchange.optimization import _create_scheduler, run_parameter_search
from satchange.training import fit_binary_change_model, train_one_epoch
from satchange.utils import load_config


def resolve_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve standard YAML into the existing flat HPO/checkpoint parameter schema."""
    training, data = config["training"], config["data"]
    loss = training["loss"]
    parameters = {
        "model_parameters": deepcopy(config["model"]["parameters"]),
        "seed": config["project"]["seed"],
        "image_size": data["image_size"],
        "batch_size": data["batch_size"],
        "num_workers": data["num_workers"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "scheduler": training.get("scheduler", "none"),
        "bce_weight": loss["bce_weight"],
        "dice_weight": loss["dice_weight"],
        "positive_weight": loss.get("pos_weight"),
        "precision": config.get("runtime", {}).get("precision", "fp32"),
    }
    return parameters


def reconstruct_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly reconstruct a trusted project checkpoint, without an existing model object."""
    state = torch.load(path, map_location=device, weights_only=False)
    for key in ("model_name", "config", "model_state_dict"):
        if key not in state:
            raise ValueError(f"Checkpoint is missing {key}.")
    config = state["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint must contain resolved architecture metadata.")
    parameters = config.get("model_parameters", config.get("model", {}).get("parameters"))
    if not isinstance(parameters, dict):
        raise ValueError("Checkpoint has no model_parameters or model.parameters mapping.")
    if "model" in config and config["model"]["name"] != state["model_name"]:
        raise ValueError("Checkpoint model identity disagrees with config.")
    model = create_change_model(state["model_name"], parameters).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model, state


def run_training(
    config: dict[str, Any],
    project_root: str | Path,
    device: torch.device,
    *,
    run_name: str | None = None,
    resume_from: str | Path | None = None,
    tiny_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Construct a fresh standard experiment and call the common fit pipeline.

    A preinitialized process group enables DDP. Batch size is per process.
    Tiny overfit deliberately validates on the same 2–8 training samples.
    """
    config = deepcopy(config)
    parameters = resolve_parameters(config)
    precision_context(device, parameters["precision"])
    seed = parameters["seed"]
    seed_everything(seed)
    root = Path(project_root)
    model_name = config["model"]["name"]
    if resume_from is not None and run_name is None:
        raise ValueError("Resume must explicitly name the existing run.")
    run_name = on_rank_zero(lambda: run_name or f"run_{uuid4().hex[:12]}")
    output_root = root / config["output"]["directory"]
    if tiny_samples is not None:
        if not 2 <= tiny_samples <= 8:
            raise ValueError("tiny_samples must be between 2 and 8.")
        output_root = output_root / "tiny_overfit"
    run_directory = output_root / model_name / run_name

    def reserve_run() -> None:
        if resume_from is None:
            run_directory.mkdir(parents=True, exist_ok=False)
        elif not run_directory.is_dir():
            raise ValueError("Resume requires the original run directory.")

    on_rank_zero(reserve_run)
    dataset_root = root / config["dataset"]["prepared_directory"]
    train_dataset, _ = create_dataset_dataloader(
        dataset_root,
        "train",
        parameters["image_size"],
        parameters["batch_size"],
        parameters["num_workers"],
        transform=None if tiny_samples else RandomFlipRotate(),
    )
    training_data: Any = train_dataset
    if tiny_samples is not None:
        if len(train_dataset) < tiny_samples:
            raise ValueError("Not enough training samples for the tiny-overfit request.")
        training_data = Subset(train_dataset, range(tiny_samples))
        validation_loader = DataLoader(training_data, batch_size=parameters["batch_size"])
    else:
        _, validation_loader = create_dataset_dataloader(
            dataset_root,
            "val",
            parameters["image_size"],
            parameters["batch_size"],
            parameters["num_workers"],
            pin_memory=device.type == "cuda",
        )
    sampler: DistributedSampler | None = (
        DistributedSampler(training_data, shuffle=True, seed=seed, drop_last=False)
        if dist.is_initialized()
        else None
    )
    train_loader = DataLoader(
        training_data,
        batch_size=parameters["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=parameters["num_workers"],
        generator=torch.Generator().manual_seed(seed + rank()),
        pin_memory=device.type == "cuda",
    )
    model = create_change_model(model_name, parameters["model_parameters"]).to(device)
    if dist.is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    seed_everything(seed + rank())
    loss = BCEDiceLoss(
        bce_weight=parameters["bce_weight"],
        dice_weight=parameters["dice_weight"],
        pos_weight=parameters["positive_weight"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"]
    )
    scheduler = _create_scheduler(optimizer, parameters["scheduler"])
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()
    config["execution"] = {
        "mode": "ddp" if dist.is_initialized() else "single",
        "world_size": world_size(),
        "per_process_batch_size": parameters["batch_size"],
        "global_batch_size": parameters["batch_size"] * world_size(),
        "precision": parameters["precision"],
        "code_revision": revision,
        "input_regime": "native" if parameters["image_size"] is None else "resized",
        "tiny_samples": tiny_samples,
        "architecture_variant": (
            "depth_ablation"
            if "encoder_depths" in parameters["model_parameters"]
            and tuple(parameters["model_parameters"]["encoder_depths"]) != (3, 4, 18, 3)
            else "default"
        ),
    }
    return fit_binary_change_model(
        model,
        train_loader,
        validation_loader,
        loss,
        optimizer,
        device,
        config["training"]["epochs"],
        config["training"]["thresholds"],
        output_root,
        model_name,
        run_name,
        config=config,
        resume_from=resume_from,
        scheduler=scheduler,
        precision=parameters["precision"],
    )


def probe_training_step(
    config: dict[str, Any],
    device: torch.device,
    *,
    enabled: bool = False,
    allow_native: bool = False,
) -> dict[str, Any]:
    """Measure one fresh real CUDA training step, including AdamW state allocation."""
    if not enabled:
        raise ValueError("The training probe requires enabled=True.")
    if dist.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("This bounded probe is single-device; use training history for DDP peaks.")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("VRAM measurement requires an explicitly selected CUDA device.")
    parameters = resolve_parameters(config)
    size = parameters["image_size"]
    batch_size = parameters["batch_size"]
    if size is None:
        if not allow_native or batch_size != 1:
            raise ValueError("Native probing requires allow_native=True and batch_size=1.")
        size = 1024
    elif size != 256 or batch_size not in (1, 2, 4):
        raise ValueError("Resized probing is bounded to 256x256 and batch sizes 1, 2, or 4.")
    precision_context(device, parameters["precision"])
    seed_everything(parameters["seed"])
    result: dict[str, Any] = {
        "config": deepcopy(config),
        "height": size,
        "width": size,
        "pairs": batch_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "precision": parameters["precision"],
        "use_checkpointing": parameters["model_parameters"].get("use_checkpointing", False),
    }
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started: float | None = None
        try:
            model = create_change_model(config["model"]["name"], parameters["model_parameters"])
            result.update(count_model_parameters(model))
            model = model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=parameters["learning_rate"],
                weight_decay=parameters["weight_decay"],
            )
            loss = BCEDiceLoss(
                bce_weight=parameters["bce_weight"],
                dice_weight=parameters["dice_weight"],
                pos_weight=parameters["positive_weight"],
            ).to(device)
            samples = [
                {
                    "before": torch.rand(3, size, size),
                    "after": torch.rand(3, size, size),
                    "mask": torch.randint(0, 2, (1, size, size)).float(),
                }
                for _ in range(batch_size)
            ]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result["loss"] = train_one_epoch(
                model,
                DataLoader(samples, batch_size=batch_size),  # type: ignore[arg-type]
                loss,
                optimizer,
                device,
                parameters["precision"],
            )
            torch.cuda.synchronize(device)
            result["status"] = "completed"
        except torch.cuda.OutOfMemoryError as error:
            result.update(status="out_of_memory", error=str(error))
        result.update(
            step_seconds=time.perf_counter() - started if started is not None else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
        )
        return result


def main() -> None:
    """Explicit terminal entry point; supports torchrun's rank environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "hpo", "probe"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tiny-samples", type=int)
    parser.add_argument("--gpu-ids", type=int, nargs="+")
    parser.add_argument("--search-space", type=Path)
    parser.add_argument("--enable-probe", action="store_true")
    parser.add_argument("--allow-native-probe", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.precision:
        config.setdefault("runtime", {})["precision"] = args.precision
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if (args.ddp and args.action != "train") or (
        not args.ddp and int(os.environ.get("WORLD_SIZE", "1")) > 1
    ):
        parser.error("torchrun requires --ddp train; HPO/probe cannot be nested in DDP.")
    device_name = args.device or config.get("runtime", {}).get("device", "cpu")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    try:
        if args.ddp:
            if "LOCAL_RANK" not in os.environ:
                parser.error("--ddp must be launched with torchrun.")
            if device.type == "cuda":
                device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
                torch.cuda.set_device(device)
            dist.init_process_group(
                "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30)
            )
        if args.action == "train":
            run_training(
                config,
                args.project_root,
                device,
                run_name=args.run_name,
                resume_from=args.resume,
                tiny_samples=args.tiny_samples,
            )
        elif args.action == "hpo":
            if not args.gpu_ids or not args.search_space:
                parser.error("hpo requires --gpu-ids and --search-space JSON.")
            search = json.loads(args.search_space.read_text(encoding="utf-8"))
            run_parameter_search(
                search,
                resolve_parameters(config),
                config["model"]["name"],
                args.project_root / config["dataset"]["prepared_directory"],
                config["training"]["thresholds"],
                config["training"]["epochs"],
                args.project_root / "artifacts" / "hyperparameter_search",
                args.run_name or f"heavy_{uuid4().hex[:12]}",
                args.gpu_ids,
            )
        else:
            result = probe_training_step(
                config, device, enabled=args.enable_probe, allow_native=args.allow_native_probe
            )
            destination = args.project_root / "artifacts" / "profiling" / f"{uuid4().hex}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
    finally:
        if args.ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
