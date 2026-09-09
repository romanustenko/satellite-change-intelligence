"""Reusable model training and validation pipeline."""

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from satchange.execution import (
    capture_rng,
    gather_objects,
    on_rank_zero,
    precision_context,
    rank,
    restore_rng,
    unwrap,
    world_size,
)
from satchange.metrics import BinaryChangeMetrics


def _save_checkpoint(
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
) -> None:
    """Save one training checkpoint."""

    if rank() != 0:
        return

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint_state,
        checkpoint_path,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    precision: str = "fp32",
) -> float:
    """Train the model for one complete epoch."""

    precision_context(device, precision)
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        before = batch["before"].to(
            device,
            non_blocking=True,
        )

        after = batch["after"].to(
            device,
            non_blocking=True,
        )

        mask = batch["mask"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        with precision_context(device, precision):
            logits = model(before, after)
        # FP32 loss/probability reductions also under BF16 autocast execution.
        logits = logits.float()
        loss = loss_function(logits, mask)

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training loss is not finite: "
                f"{loss.item()}."
            )

        loss.backward()
        optimizer.step()

        current_batch_size = before.shape[0]

        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_samples += current_batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Training data loader is empty."
        )

    return total_loss / total_samples


@torch.no_grad()
def evaluate_thresholds(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    thresholds: list[float],
    precision: str = "fp32",
) -> dict[str, Any]:
    """Evaluate one fixed model at multiple probability thresholds."""

    if not thresholds:
        raise ValueError(
            "At least one threshold is required."
        )

    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "Every threshold must be between 0 and 1. "
                f"Received {threshold}."
            )

    precision_context(device, precision)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    threshold_metrics = {
        threshold: BinaryChangeMetrics(threshold=threshold)
        for threshold in thresholds
    }

    changed_probability_sum = 0.0
    unchanged_probability_sum = 0.0

    changed_pixel_count = 0
    unchanged_pixel_count = 0

    probability_minimum = float("inf")
    probability_maximum = float("-inf")

    changed_probability_maximum = float("-inf")
    unchanged_probability_maximum = float("-inf")

    for batch in data_loader:
        before = batch["before"].to(
            device,
            non_blocking=True,
        )

        after = batch["after"].to(
            device,
            non_blocking=True,
        )

        mask = batch["mask"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        with precision_context(device, precision):
            logits = model(before, after)
        # FP32 loss/probability reductions also under BF16 autocast execution.
        logits = logits.float()
        loss = loss_function(logits, mask)

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Evaluation loss is not finite: "
                f"{loss.item()}."
            )

        current_batch_size = before.shape[0]

        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_samples += current_batch_size

        probabilities = torch.sigmoid(
            logits
        )

        target_mask = mask >= 0.5

        changed_probabilities = probabilities[
            target_mask
        ]

        unchanged_probabilities = probabilities[
            ~target_mask
        ]

        changed_probability_sum += (
            changed_probabilities.sum().item()
        )

        unchanged_probability_sum += (
            unchanged_probabilities.sum().item()
        )

        changed_pixel_count += (
            changed_probabilities.numel()
        )

        unchanged_pixel_count += (
            unchanged_probabilities.numel()
        )

        probability_minimum = min(
            probability_minimum,
            probabilities.min().item(),
        )

        probability_maximum = max(
            probability_maximum,
            probabilities.max().item(),
        )

        if changed_probabilities.numel() > 0:
            changed_probability_maximum = max(
                changed_probability_maximum,
                changed_probabilities.max().item(),
            )

        if unchanged_probabilities.numel() > 0:
            unchanged_probability_maximum = max(
                unchanged_probability_maximum,
                unchanged_probabilities.max().item(),
            )

        for metrics in threshold_metrics.values():
            metrics.update_probabilities(probabilities, mask)

    if total_samples == 0:
        raise RuntimeError(
            "Evaluation data loader is empty."
        )

    threshold_results = []

    for threshold in thresholds:
        result = threshold_metrics[threshold].compute()
        tp = int(result["true_positive"])
        fp = int(result["false_positive"])
        fn = int(result["false_negative"])
        tn = int(result["true_negative"])

        threshold_results.append(
            {
                "threshold": threshold,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "iou": result["iou"],
                "accuracy": result["accuracy"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "predicted_changed_pixels": tp + fp,
            }
        )

    mean_changed_probability = (
        changed_probability_sum
        / changed_pixel_count
        if changed_pixel_count > 0
        else 0.0
    )

    mean_unchanged_probability = (
        unchanged_probability_sum
        / unchanged_pixel_count
        if unchanged_pixel_count > 0
        else 0.0
    )

    return {
        "loss": total_loss / total_samples,
        "probability_minimum": probability_minimum,
        "probability_maximum": probability_maximum,
        "mean_changed_probability": (
            mean_changed_probability
        ),
        "mean_unchanged_probability": (
            mean_unchanged_probability
        ),
        "maximum_changed_probability": (
            changed_probability_maximum
        ),
        "maximum_unchanged_probability": (
            unchanged_probability_maximum
        ),
        "changed_pixel_count": changed_pixel_count,
        "unchanged_pixel_count": unchanged_pixel_count,
        "threshold_results": threshold_results,
    }


def fit_binary_change_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    number_of_epochs: int,
    thresholds: list[float],
    output_root: str | Path,
    model_name: str,
    run_name: str,
    checkpoint_interval: int = 5,
    config: dict[str, Any] | None = None,
    resume_from: str | Path | None = None,
    scheduler: Any | None = None,
    log_prefix: str = "",
    precision: str = "fp32",
) -> list[dict[str, Any]]:
    """Train or resume a binary change-detection model."""

    if (
        not isinstance(number_of_epochs, int)
        or isinstance(number_of_epochs, bool)
        or number_of_epochs <= 0
    ):
        raise ValueError(
            "number_of_epochs must be a positive integer."
        )

    if (
        not isinstance(checkpoint_interval, int)
        or isinstance(checkpoint_interval, bool)
        or checkpoint_interval <= 0
    ):
        raise ValueError(
            "checkpoint_interval must be a positive integer."
        )

    precision_context(device, precision)
    bare_model = unwrap(model)
    model_name = model_name.strip()
    run_name = run_name.strip()

    if not model_name:
        raise ValueError(
            "model_name cannot be empty."
        )

    if not run_name:
        raise ValueError(
            "run_name cannot be empty."
        )

    output_root = Path(
        output_root
    )

    run_directory = (
        output_root
        / model_name
        / run_name
    )

    checkpoints_directory = (
        run_directory
        / "checkpoints"
    )

    checkpoints_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_history: list[
        dict[str, Any]
    ] = []

    start_epoch = 1

    best_metrics: dict[
        str,
        Any,
    ] | None = None

    best_f1 = -1.0

    if resume_from is not None:
        resume_path = Path(
            resume_from
        )

        if not resume_path.is_file():
            raise FileNotFoundError(
                "Checkpoint was not found: "
                f"{resume_path}"
            )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        required_keys = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
        }

        missing_keys = (
            required_keys
            - checkpoint.keys()
        )

        if missing_keys:
            raise KeyError(
                "Checkpoint is missing required keys: "
                f"{sorted(missing_keys)}."
            )

        saved_model_name = checkpoint.get(
            "model_name"
        )

        if (
            saved_model_name is not None
            and saved_model_name != model_name
        ):
            raise ValueError(
                "Checkpoint model mismatch. "
                f"Expected '{model_name}', "
                f"received '{saved_model_name}'."
            )

        saved_run_name = checkpoint.get(
            "run_name"
        )

        if (
            saved_run_name is not None
            and saved_run_name != run_name
        ):
            raise ValueError(
                "Checkpoint run mismatch. "
                f"Expected '{run_name}', "
                f"received '{saved_run_name}'."
            )

        if checkpoint.get("world_size", 1) != world_size():
            raise ValueError("Exact resume requires the same world size.")
        if checkpoint.get("precision", "fp32") != precision:
            raise ValueError("Resume precision differs from the saved experiment.")
        saved_config = checkpoint.get("config")
        if config is not None and saved_config is not None:
            def architecture(value: dict[str, Any]) -> Any:
                return value.get("model_parameters", value.get("model", {}).get("parameters"))
            if architecture(config) != architecture(saved_config):
                raise ValueError("Resume architecture parameters differ from checkpoint metadata.")
        if "rng_states" in checkpoint:
            restore_rng(checkpoint["rng_states"][rank()], train_loader)
        bare_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        saved_scheduler_state = checkpoint.get(
            "scheduler_state_dict"
        )

        if saved_scheduler_state is not None:
            if scheduler is None:
                raise ValueError(
                    "The checkpoint contains scheduler state, "
                    "but no scheduler was provided."
                )

            scheduler.load_state_dict(
                saved_scheduler_state
            )

        saved_epoch = int(
            checkpoint["epoch"]
        )

        start_epoch = saved_epoch + 1

        metric_history = list(
            checkpoint.get(
                "history",
                [],
            )
        )

        best_metrics = checkpoint.get(
            "best_metrics"
        )

        if best_metrics is not None:
            best_f1 = float(
                best_metrics["validation_f1"]
            )

        elif metric_history:
            best_metrics = max(
                metric_history,
                key=lambda result: (
                    result["validation_f1"]
                ),
            )

            best_f1 = float(
                best_metrics["validation_f1"]
            )

        print(
            "Resumed from checkpoint:",
            resume_path,
        )

        print(
            "Saved epoch:",
            saved_epoch,
        )

        print(
            "Next epoch:",
            start_epoch,
        )

    final_epoch = (
        start_epoch
        + number_of_epochs
        - 1
    )

    for epoch in range(
        start_epoch,
        final_epoch + 1,
    ):
        # Synchronize previous CUDA work before starting the timer.
        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

            # Reset peak-memory statistics for this epoch.
            torch.cuda.reset_peak_memory_stats(
                device
            )

        epoch_started_at = (
            time.perf_counter()
        )

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            precision=precision,
        )

        train_loss = sum(gather_objects(train_loss)) / world_size()
        validation_results = on_rank_zero(lambda: evaluate_thresholds(
            model=bare_model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
            thresholds=thresholds,
            precision=precision,
        ))

        best_threshold_result = max(
            validation_results[
                "threshold_results"
            ],
            key=lambda result: result["f1"],
        )

        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        if scheduler is not None:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(
                    best_threshold_result["f1"]
                )
            else:
                scheduler.step()

        next_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

        epoch_seconds = (
            time.perf_counter()
            - epoch_started_at
        )

        if device.type == "cuda":
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(
                    device
                )
                / 1024 ** 2
            )
        else:
            peak_gpu_memory_mb = 0.0


        threshold_05_result = next(
            (
                result
                for result
                in validation_results[
                    "threshold_results"
                ]
                if result["threshold"] == 0.50
            ),
            None,
        )

        mean_changed_probability = (
            validation_results[
                "mean_changed_probability"
            ]
        )

        mean_unchanged_probability = (
            validation_results[
                "mean_unchanged_probability"
            ]
        )

        probability_gap = (
            mean_changed_probability
            - mean_unchanged_probability
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": (
                validation_results["loss"]
            ),
            "best_threshold": (
                best_threshold_result["threshold"]
            ),
            "validation_precision": (
                best_threshold_result["precision"]
            ),
            "validation_recall": (
                best_threshold_result["recall"]
            ),
            "validation_f1": (
                best_threshold_result["f1"]
            ),
            "validation_iou": (
                best_threshold_result["iou"]
            ),
            "validation_accuracy": (
                best_threshold_result["accuracy"]
            ),
            "true_positive": (
                best_threshold_result["tp"]
            ),
            "false_positive": (
                best_threshold_result["fp"]
            ),
            "false_negative": (
                best_threshold_result["fn"]
            ),
            "true_negative": (
                best_threshold_result["tn"]
            ),
            "f1_at_05": (
                threshold_05_result["f1"]
                if threshold_05_result is not None
                else None
            ),
            "mean_changed_probability": (
                mean_changed_probability
            ),
            "mean_unchanged_probability": (
                mean_unchanged_probability
            ),
            "probability_gap": probability_gap,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        }

        metric_history.append(
            epoch_metrics
        )

        current_f1 = float(
            epoch_metrics["validation_f1"]
        )

        is_best_epoch = (
            current_f1 > best_f1
        )

        if is_best_epoch:
            best_f1 = current_f1
            best_metrics = dict(
                epoch_metrics
            )

        memory_by_rank = gather_objects({
            "rank": rank(), "peak_allocated_mib": peak_gpu_memory_mb,
            "peak_reserved_mib": (torch.cuda.max_memory_reserved(device) / 1024 ** 2
                                  if device.type == "cuda" else 0.0),
        })
        epoch_metrics["memory_by_rank"] = memory_by_rank
        epoch_metrics["max_device_peak_allocated_mib"] = max(
            item["peak_allocated_mib"] for item in memory_by_rank)
        epoch_metrics["gpu_hours"] = max(gather_objects(epoch_seconds)) * (
            world_size() if device.type == "cuda" else 0) / 3600
        rng_states = gather_objects(capture_rng(train_loader))
        checkpoint_state = {
            "world_size": world_size(), "precision": precision, "rng_states": rng_states,
            "model_name": model_name,
            "run_name": run_name,
            "epoch": epoch,
            "model_state_dict": (
                bare_model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "current_metrics": (
                epoch_metrics
            ),
            "best_metrics": (
                best_metrics
            ),
            "history": (
                metric_history
            ),
            "config": config,
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
        }

        def save_epoch(
            checkpoint_state: dict[str, Any] = checkpoint_state,
            epoch: int = epoch, is_best_epoch: bool = is_best_epoch,
        ) -> None:
            _save_checkpoint(
                checkpoint_path=(
                    run_directory
                    / "last.pt"
                ),
                checkpoint_state=checkpoint_state,
            )

            if (
                epoch % checkpoint_interval == 0
                or epoch == final_epoch
            ):
                _save_checkpoint(
                    checkpoint_path=(
                        checkpoints_directory
                        / (
                            f"{model_name}_"
                            f"epoch_{epoch:04d}.pt"
                        )
                    ),
                    checkpoint_state=checkpoint_state,
                )

            if is_best_epoch:
                _save_checkpoint(
                    checkpoint_path=(
                        run_directory
                        / "best.pt"
                    ),
                    checkpoint_state=checkpoint_state,
                )

        on_rank_zero(save_epoch)
        if rank() != 0:
            continue

        print()
        print("=" * 80)
        print(
            f"{log_prefix}Epoch {epoch}/{final_epoch}"
        )
        print("=" * 80)

        print(
            f"Train loss:       "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss:  "
            f"{validation_results['loss']:.6f}"
        )

        print(
            f"Best threshold:   "
            f"{best_threshold_result['threshold']:.2f}"
        )

        print(
            f"Precision:        "
            f"{best_threshold_result['precision']:.6f}"
        )

        print(
            f"Recall:           "
            f"{best_threshold_result['recall']:.6f}"
        )

        print(
            f"F1:               "
            f"{best_threshold_result['f1']:.6f}"
        )

        print(
            f"IoU:              "
            f"{best_threshold_result['iou']:.6f}"
        )

    return metric_history
