"""Reusable model training and validation pipeline."""

import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

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


@dataclass
class EarlyStopping:
    """Track significant validation-F1 improvements, independently of best.pt.

    A gain must exceed min_delta relative to the last significant best score.
    Small gains therefore accumulate. Patience is counted from the first epoch;
    min_epochs only prevents stopping too early. None disables early stopping.
    """

    patience: int | None = None
    min_epochs: int = 1
    min_delta: float = 0.0
    monitor: str = "validation_f1"
    best_score: float | None = field(default=None, init=False)
    bad_epochs: int = field(default=0, init=False)
    last_epoch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.monitor not in ("validation_f1", "selection_score"):
            raise ValueError("Unsupported early-stopping monitor.")
        if self.patience is not None and (
            type(self.patience) is not int or self.patience < 1
        ):
            raise ValueError("patience must be None or a positive integer.")
        if type(self.min_epochs) is not int or self.min_epochs < 1:
            raise ValueError("min_epochs must be a positive integer.")
        if (
            isinstance(self.min_delta, bool)
            or not isinstance(self.min_delta, (int, float))
            or not math.isfinite(self.min_delta)
            or self.min_delta < 0
        ):
            raise ValueError("min_delta must be a finite, nonnegative number.")
        self.min_delta = float(self.min_delta)

    @property
    def policy(self) -> dict[str, Any] | None:
        """Canonical metadata used by checkpoints, HPO manifests, and cache keys."""
        if self.patience is None:
            return None
        return {
            "monitor": self.monitor,
            "patience": self.patience,
            "min_epochs": self.min_epochs,
            "min_delta": self.min_delta,
        }

    def validate_budget(self, maximum_epoch: int) -> None:
        if type(maximum_epoch) is not int or maximum_epoch < 1:
            raise ValueError("The epoch budget must be a positive integer.")
        if self.patience is not None and self.min_epochs > maximum_epoch:
            raise ValueError("min_epochs cannot exceed the maximum epoch budget.")

    @property
    def should_stop(self) -> bool:
        return (
            self.patience is not None
            and self.last_epoch >= self.min_epochs
            and self.bad_epochs >= self.patience
        )

    def step(self, score: float, epoch: int) -> bool:
        self.last_epoch = epoch
        if self.patience is None:
            return False
        if not math.isfinite(score):
            raise ValueError("Early stopping requires a finite validation F1.")
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.should_stop

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "best_score": self.best_score,
            "bad_epochs": self.bad_epochs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state: dict[str, Any] | None, saved_epoch: int) -> None:
        """Restore the same stopping policy; do not silently reset patience."""
        if state is None:
            if self.patience is not None:
                raise ValueError(
                    "Checkpoint has no early-stopping state. Start a new run "
                    "instead of enabling patience during an exact resume."
                )
            self.last_epoch = saved_epoch
            return
        if state.get("policy") != self.policy:
            raise ValueError("Resume early-stopping policy differs from checkpoint.")
        bad_epochs = state.get("bad_epochs")
        best_score = state.get("best_score")
        if (
            state.get("last_epoch") != saved_epoch
            or type(bad_epochs) is not int
            or not 0 <= bad_epochs <= saved_epoch
            or (self.patience is not None and best_score is None)
            or (best_score is not None and (
                isinstance(best_score, bool)
                or not isinstance(best_score, (int, float))
                or not math.isfinite(best_score)
                or not 0 <= best_score <= 1
            ))
        ):
            raise ValueError("Invalid early-stopping checkpoint state.")
        self.best_score = best_score
        self.bad_epochs = bad_epochs
        self.last_epoch = saved_epoch


def _save_checkpoint(checkpoint_path: Path, checkpoint_state: dict[str, Any]) -> None:
    """Commit a checkpoint atomically; a failed write leaves the previous file intact."""
    if rank() != 0:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as file:
            torch.save(checkpoint_state, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a locally produced full checkpoint on CPU, not into scarce GPU memory.

    This includes Python/NumPy RNG state and therefore uses weights_only=False.
    Only supply checkpoints from your own training or another trusted source.
    Inference-only exports are intentionally not accepted as resumable state.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"epoch", "model_state_dict", "optimizer_state_dict"}
    if not isinstance(state, dict) or required - state.keys():
        raise ValueError(f"Not a complete training checkpoint: {path}")
    if type(state["epoch"]) is not int or state["epoch"] < 1:
        raise ValueError(f"Invalid checkpoint epoch: {path}")
    return state


def find_training_checkpoint(directory: str | Path) -> tuple[Path, int] | None:
    """Prefer committed last.pt; fall back to numbered epoch files if it is corrupt.

    Never silently restart from zero when checkpoint files exist but none load.
    Orphaned temporary files from an interrupted write are not candidates.
    """
    directory = Path(directory)
    candidates = [directory / "last.pt"] + sorted(
        (directory / "checkpoints").glob("*_epoch_*.pt"), reverse=True
    )
    existing = [path for path in candidates if path.is_file()]
    for path in existing:
        try:
            state = load_training_checkpoint(path)
        except (
            OSError, EOFError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError
        ) as error:
            warnings.warn(f"Cannot resume {path}: {error}", RuntimeWarning, stacklevel=2)
            continue
        return path, state["epoch"]
    if existing:
        raise RuntimeError(f"No readable training checkpoint in {directory}; files were preserved.")
    return None


def _checkpoint_run_directory(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def _best_weights(state: dict[str, Any], source: Path) -> dict[str, torch.Tensor]:
    """Recover the best weights at/before this snapshot, never later weights."""
    best_epoch = state["best_metrics"]["epoch"]
    if best_epoch == state["epoch"]:
        return state["model_state_dict"]
    if state.get("best_model_state_dict") is not None:
        return state["best_model_state_dict"]
    # Compatibility with older checkpoints, which depend on a matching best file.
    root = _checkpoint_run_directory(source)
    candidates = [root / "best.pt", root / "checkpoints" /
                  f"{state['model_name']}_epoch_{best_epoch:04d}.pt"]
    for path in candidates:
        if not path.is_file():
            continue
        best = torch.load(path, map_location="cpu", weights_only=False)
        if (best.get("epoch") == best_epoch and best.get("model_name") == state.get("model_name")
                and best.get("config") == state.get("config")):
            return best["model_state_dict"]
    raise ValueError(
        f"Checkpoint {source} needs best weights from epoch {best_epoch}, which were not found. "
        "Use another full snapshot; do not substitute weights from a later epoch."
    )


def restore_best_checkpoint(state: dict[str, Any], source: Path, destination: Path) -> None:
    """Restore selection weights alongside a resumed run, including an explicit rollback.

    If the historical full best snapshot is unavailable, best.pt is a labelled
    inference-only export. last.pt and numbered checkpoints remain full snapshots.
    """
    weights = _best_weights(state, source)
    best_epoch = state["best_metrics"]["epoch"]
    if best_epoch == state["epoch"]:
        best_state = dict(state)
    else:
        best_state = {
            "checkpoint_kind": "selection_weights", "epoch": best_epoch,
            "model_name": state["model_name"], "run_name": state["run_name"],
            "model_state_dict": weights, "best_metrics": state["best_metrics"],
            "current_metrics": state["best_metrics"], "config": state.get("config"),
            "precision": state.get("precision", "fp32"), "world_size": state.get("world_size", 1),
            "resumed_from": str(source),
        }
    if best_epoch != state["epoch"]:
        # Keep the existing full best snapshot when it corresponds to this prefix.
        original = _checkpoint_run_directory(source) / "best.pt"
        if original.is_file():
            try:
                candidate = load_training_checkpoint(original)
            except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError):
                candidate = None
            if (candidate is not None and candidate["epoch"] == best_epoch
                    and candidate.get("model_name") == state.get("model_name")
                    and candidate.get("best_metrics", {}).get("selection_score",
                        candidate.get("best_metrics", {}).get("validation_f1"))
                        == state["best_metrics"].get("selection_score",
                           state["best_metrics"].get("validation_f1"))
                    and candidate["model_state_dict"].keys() == weights.keys()
                    and all(torch.equal(candidate["model_state_dict"][key].cpu(), tensor.cpu())
                            for key, tensor in weights.items())):
                best_state = dict(candidate, run_name=state["run_name"], config=state.get("config"))
    if "task" in state:
        best_state["task"] = state["task"]
        best_state["runtime"] = state.get("runtime", {})
        best_state["runtime_history"] = state.get("runtime_history", [])
    _save_checkpoint(destination, best_state)


def _copy_weights_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Keep one independent historical best on CPU, never a second model on GPU."""
    return {name: tensor.detach().to("cpu", copy=True)
            for name, tensor in model.state_dict().items()}


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
    *,
    patience: int | None = None,
    min_epochs: int = 1,
    min_delta: float = 0.0,
    task: str = "binary_change_detection",
    reset_early_stopping: bool = False,
    stop_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Train or resume, optionally stopping after patience epochs without F1 gain.

    From scratch, number_of_epochs is a maximum. On explicit checkpoint resume,
    it retains its existing meaning: the maximum number of additional epochs.
    min_epochs refers to the absolute epoch number. Every F1 maximum still saves
    best.pt, even when its gain is smaller than the stopping min_delta.
    """
    from satchange.models.factory import BINARY_TASK, SEMANTIC_TASK
    if task not in (BINARY_TASK, SEMANTIC_TASK):
        raise ValueError(f"Unsupported training task: {task}")
    semantic = task == SEMANTIC_TASK
    score_key = "selection_score" if semantic else "validation_f1"
    if semantic and (thresholds or world_size() != 1):
        raise ValueError("Semantic ROI trials use no thresholds and one worker per GPU.")
    if semantic and config is None:
        raise ValueError("Semantic training requires resolved HPO configuration.")
    tf32 = bool((config or {}).get("tf32", False))
    early_stopping = EarlyStopping(patience, min_epochs, min_delta, monitor=score_key)
    if resume_from is None:
        early_stopping.validate_budget(number_of_epochs)

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
    best_weights: dict[str, torch.Tensor] | None = None
    runtime_history: list[dict[str, Any]] = []

    if resume_from is not None:
        resume_path = Path(
            resume_from
        )

        if not resume_path.is_file():
            raise FileNotFoundError(
                "Checkpoint was not found: "
                f"{resume_path}"
            )

        checkpoint = load_training_checkpoint(resume_path)

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
        if checkpoint.get("task", BINARY_TASK) != task:
            raise ValueError("Resume task differs from the saved checkpoint.")
        saved_config = checkpoint.get("config")
        if semantic:
            from satchange.optimization import _resume_parameters
            if (config is None
                    or _resume_parameters(config) != _resume_parameters(saved_config or {})):
                raise ValueError("Resume semantic taxonomy/data/training protocol differs.")
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
                best_metrics[score_key]
            )

        elif metric_history:
            best_metrics = max(
                metric_history,
                key=lambda result: (
                    result[score_key]
                ),
            )

            best_f1 = float(
                best_metrics[score_key]
            )

        if semantic:
            runtime_history = checkpoint.get("runtime_history", [])
            previous_runtime = checkpoint.get("runtime", {})
            current_runtime = {"device": str(device), "torch": str(torch.__version__),
                               "cuda": torch.version.cuda, "tf32": tf32}
            if previous_runtime != current_runtime:
                runtime_history = [*runtime_history, {"resumed_epoch": saved_epoch,
                                   "previous": previous_runtime, "current": current_runtime}]
        if reset_early_stopping:
            early_stopping.last_epoch = saved_epoch
        else:
            early_stopping.load_state_dict(checkpoint.get("early_stopping"), saved_epoch)
        best_weights = _best_weights(checkpoint, resume_path)
        on_rank_zero(lambda: restore_best_checkpoint(
            checkpoint, resume_path, run_directory / "best.pt"
        ))
        # Discard loaded optimizer tensors on CPU after loading into the optimizer.
        del checkpoint

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

    early_stopping.validate_budget(final_epoch)
    if early_stopping.should_stop:
        # An already early-stopped checkpoint is a completed run, not a restart.
        return metric_history

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
        if semantic:
            from satchange.models import TinySiameseChangeClassifier
            from satchange.semantic.training import SemanticCELoss, run_epoch
            if not isinstance(bare_model, TinySiameseChangeClassifier) or not isinstance(
                loss_function, SemanticCELoss
            ):
                raise ValueError(
                    "Semantic task requires Tiny classifier and SemanticCELoss adapters."
                )
            weights = loss_function.weights
            assert weights is None or isinstance(weights, torch.Tensor)
            training_results = run_epoch(bare_model, train_loader, device, weights=weights,
                                         precision=precision, optimizer=optimizer)
            train_loss = training_results["loss"]
            validation_results = run_epoch(bare_model, validation_loader, device,
                                           weights=weights, precision=precision)
            epoch_score = validation_results["selection_score"]
        else:
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

            epoch_score = best_threshold_result["f1"]

        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        if scheduler is not None:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(
                    epoch_score
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


        if semantic:
            epoch_metrics = {
                "epoch": epoch, "task": task, "train_loss": train_loss,
                "validation_loss": validation_results["loss"],
                **{k: v for k, v in validation_results.items() if k not in ("loss", "loss_totals")},
                "learning_rate": learning_rate, "next_learning_rate": next_learning_rate,
                "epoch_seconds": epoch_seconds, "peak_gpu_memory_mb": peak_gpu_memory_mb,
            }
        else:
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
            epoch_metrics[score_key]
        )

        should_stop = early_stopping.step(current_f1, epoch)
        stop_reason = (
            "early_stopping" if should_stop and epoch < final_epoch
            else "max_epochs" if epoch == final_epoch
            else None
        )
        epoch_metrics["early_stopping_bad_epochs"] = early_stopping.bad_epochs
        epoch_metrics["stop_reason"] = stop_reason

        is_best_epoch = (
            current_f1 > best_f1
        )

        if is_best_epoch:
            best_weights = _copy_weights_to_cpu(bare_model)
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
            "checkpoint_schema_version": 2,
            "thresholds": list(thresholds),
            "best_model_state_dict": best_weights,
            "early_stopping": early_stopping.state_dict(),
            "stop_reason": stop_reason,
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

        if semantic:
            checkpoint_state.pop("thresholds")
            checkpoint_state["task"] = task
            checkpoint_state["runtime_history"] = runtime_history
            checkpoint_state["runtime"] = {"device": str(device), "torch": str(torch.__version__),
                                           "cuda": torch.version.cuda, "tf32": tf32}

        def save_epoch(
            checkpoint_state: dict[str, Any] = checkpoint_state,
            epoch: int = epoch, is_best_epoch: bool = is_best_epoch,
            should_stop: bool = should_stop,
        ) -> None:
            if (
                epoch % checkpoint_interval == 0
                or epoch == final_epoch
                or should_stop
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

            _save_checkpoint(run_directory / "last.pt", checkpoint_state)

        on_rank_zero(save_epoch)
        pause_requested = any(gather_objects(
            stop_file is not None and Path(stop_file).exists()
        ))
        if pause_requested:
            if rank() == 0:
                print(f"{log_prefix}Paused after saving epoch {epoch}.", flush=True)
            break
        if rank() != 0:
            if should_stop:
                break
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

        if semantic:
            print(f"Validation selection score: {epoch_score:.6f}; "
                  f"joint accuracy: {validation_results['joint_accuracy']:.6f}")
        else:
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

        if should_stop:
            if stop_reason == "early_stopping":
                print(
                    f"{log_prefix}Early stopping at epoch {epoch}: "
                    f"{early_stopping.bad_epochs} epochs without {score_key} "
                    f"gain > {early_stopping.min_delta:g}. "
                    f"Best checkpoint: epoch {(best_metrics or {})['epoch']}, "
                    f"{score_key}={best_f1:.6f}."
                )
            break

    return metric_history


HPO_TRAINER_PATCH_VERSION = 1
