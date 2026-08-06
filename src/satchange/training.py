"""Reusable model training and validation pipeline."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def _save_checkpoint(
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
) -> None:
    """Save one training checkpoint."""

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint_state,
        checkpoint_path,
    )


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one complete epoch."""
    
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

        logits = model(
            before,
            after,
        )

        loss = loss_function(
            logits,
            mask,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Training loss is not finite: {loss.item()}"
            )

        loss.backward()
        optimizer.step()

        current_batch_size = before.shape[0]

        total_loss += (
            loss.item() * current_batch_size
        )
        total_samples += current_batch_size

    if total_samples == 0:
        raise RuntimeError("Training data loader is empty.")    

    return total_loss / total_samples


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
) -> list[dict[str, Any]]:
    """Train or resume a binary change-detection model."""

    if (
        not isinstance(number_of_epochs, int)
        or isinstance(number_of_epochs, bool)
        or number_of_epochs <= 0
    ):
        raise ValueError(
            "number_of_epochs must be a "
            "positive integer."
        )

    if (
        not isinstance(checkpoint_interval, int)
        or isinstance(
            checkpoint_interval,
            bool,
        )
        or checkpoint_interval <= 0
    ):
        raise ValueError(
            "checkpoint_interval must be a "
            "positive integer."
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
    best_metrics = None
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

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain "
                "'optimizer_state_dict'."
            )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        saved_model_name = (
            checkpoint.get(
                "model_name"
            )
        )

        if (
            saved_model_name is not None
            and saved_model_name
            != model_name
        ):
            raise ValueError(
                "Checkpoint model mismatch: "
                f"{saved_model_name}."
            )

        saved_run_name = (
            checkpoint.get(
                "run_name"
            )
        )

        if (
            saved_run_name is not None
            and saved_run_name
            != run_name
        ):
            raise ValueError(
                "Checkpoint run mismatch: "
                f"{saved_run_name}."
            )

        saved_epoch = int(
            checkpoint["epoch"]
        )

        start_epoch = (
            saved_epoch + 1
        )

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
                best_metrics[
                    "validation_f1"
                ]
            )

    final_epoch = (
        start_epoch
        + number_of_epochs
        - 1
    )
