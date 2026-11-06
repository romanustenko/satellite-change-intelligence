"""Hyperparameter-search utilities."""

import gc
import inspect
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import time
import warnings
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from satchange.data import (
    RandomFlipRotate,
    create_dataset_dataloader,
)
from satchange.execution import resolve_device
from satchange.losses import BCEDiceLoss
from satchange.models.factory import (
    BINARY_TASK,
    count_model_parameters,
    create_task_model,
    model_class_for_task,
)
from satchange.training import (
    EarlyStopping,
    _best_weights,
    _save_checkpoint,
    find_training_checkpoint,
    fit_binary_change_model,
    load_training_checkpoint,
    restore_best_checkpoint,
)


def _expand_search_space(
    search_space: dict[str, list[Any]] | list[dict[str, list[Any]]],
) -> list[dict[str, Any]]:
    """Create all combinations from a search space."""

    # BEGIN EXACT-CANDIDATE GRIDS
    # A list is a union of independent grids, not their Cartesian product.
    # The controller sends singleton grids to preserve an exact candidate list.
    if isinstance(search_space, list):
        if not search_space:
            raise ValueError("search_space cannot be empty.")
        trials = []
        for grid in search_space:
            if not isinstance(grid, dict):
                raise TypeError("Each search grid must be a dictionary.")
            trials.extend(_expand_search_space(grid))
        return trials
    if not isinstance(search_space, dict):
        raise TypeError("search_space must be a dictionary or a list of grids.")
    # END EXACT-CANDIDATE GRIDS
    if not search_space:
        raise ValueError(
            "search_space cannot be empty."
        )

    parameter_names = list(
        search_space
    )

    parameter_values = []

    for name in parameter_names:
        values = search_space[name]

        if not values:
            raise ValueError(
                f"Search values for '{name}' "
                "cannot be empty."
            )

        parameter_values.append(
            values
        )

    trials = []

    for combination in product(
        *parameter_values
    ):
        trials.append(
            dict(
                zip(
                    parameter_names,
                    combination,
                    strict=True,
                )
            )
        )

    return trials


def _build_trial_parameters(
    fixed_parameters: dict[str, Any],
    search_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Merge one trial without mutating the fixed or searched parameter mappings."""

    parameters = deepcopy(fixed_parameters)
    nested_parameters = parameters.get("model_parameters", {})
    if not isinstance(nested_parameters, dict):
        raise ValueError("model_parameters must be a dictionary.")

    for name in tuple(parameters):
        if name.startswith("model__"):
            model_parameter_name = name.removeprefix("model__")
            if not model_parameter_name:
                raise ValueError("model__ search parameters must name a model parameter.")
            nested_parameters[model_parameter_name] = parameters.pop(name)

    if "model_parameters" in search_parameters:
        raise ValueError("Use model__parameter candidates instead of replacing model_parameters.")
    if "loss_weights" in search_parameters and any(
        name in search_parameters for name in ("bce_weight", "dice_weight")
    ):
        raise ValueError("Do not combine loss_weights with separate searched loss weights.")
    for name, value in search_parameters.items():
        if name.startswith("model__"):
            model_parameter_name = name.removeprefix("model__")
            if not model_parameter_name:
                raise ValueError("model__ search parameters must name a model parameter.")
            nested_parameters[model_parameter_name] = deepcopy(value)
        else:
            parameters[name] = deepcopy(value)

    parameters["model_parameters"] = nested_parameters

    if "loss_weights" in parameters:
        loss_weights = parameters.pop("loss_weights")
        if not isinstance(loss_weights, (list, tuple)) or len(loss_weights) != 2:
            raise ValueError("loss_weights must contain (bce_weight, dice_weight).")
        parameters["bce_weight"], parameters["dice_weight"] = loss_weights

    return parameters

def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
) -> Any | None:
    """Create the requested learning-rate scheduler."""

    scheduler_name = (
        scheduler_name
        .strip()
        .lower()
    )

    if scheduler_name == "none":
        return None

    if scheduler_name == "reduce_on_plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
            )
        )

    raise ValueError(
        "Unknown scheduler: "
        f"'{scheduler_name}'."
    )


def _format_trial_parameters(parameters: dict[str, Any]) -> str:
    """Return training and resolved architecture parameters as one compact line."""

    return " | ".join(
        [
            f"seed={parameters.get('seed')}",
            f"lr={parameters.get('learning_rate')}",
            f"wd={parameters.get('weight_decay')}",
            f"pw={parameters.get('positive_weight')}",
            f"BCE/Dice={parameters.get('bce_weight')}/{parameters.get('dice_weight')}",
            f"batch={parameters.get('batch_size')}",
            f"image_size={parameters.get('image_size')}",
            *[
                f"{name}={value}"
                for name, value in parameters.get("model_parameters", {}).items()
            ],
            f"scheduler={parameters.get('scheduler')}",
        ]
    )


def _summarize_training_completion(
    history: list[dict[str, Any]], budget: int, early_stopping: EarlyStopping,
) -> dict[str, Any]:
    """Distinguish a valid early stop from an incomplete/failed HPO trial."""
    completed = len(history)
    if not history or not 1 <= completed <= budget or history[-1]["epoch"] != completed:
        raise RuntimeError("HPO training returned an incomplete or inconsistent history.")
    reason = history[-1].get("stop_reason")
    if reason is None and completed == budget and early_stopping.policy is None:
        reason = "max_epochs"  # Compatibility with fixed-budget trainers.
    if reason == "early_stopping":
        if (
            early_stopping.patience is None
            or not early_stopping.min_epochs <= completed < budget
            or history[-1].get("early_stopping_bad_epochs", 0) < early_stopping.patience
        ):
            raise RuntimeError("Training returned inconsistent early-stopping metadata.")
    elif reason != "max_epochs" or completed != budget:
        raise RuntimeError("Training ended without reaching its budget or a valid early stop.")
    return {
        "budget_epochs": budget,
        "epochs_completed": completed,
        "stopped_early": reason == "early_stopping",
        "stop_reason": reason,
        "early_stopping": early_stopping.policy,
    }


# Checkpoint recovery is separate from trial construction and GPU scheduling.
_RESUME_TRAINING_KEYS = {
    "model_parameters", "seed", "image_size", "batch_size", "num_workers",
    "learning_rate", "weight_decay", "scheduler", "positive_weight",
    "bce_weight", "dice_weight", "precision",
}


def _resume_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    keys = _RESUME_TRAINING_KEYS
    values = {key: parameters[key] for key in keys if key in parameters}
    values.setdefault("precision", "fp32")
    # JSON normalizes tuple/list architecture and loss parameters consistently.
    return json.loads(json.dumps(values, sort_keys=True, allow_nan=False))


def read_hpo_checkpoint(
    path: str | Path, *, model_name: str, dataset_root: str | Path,
    thresholds: list[float], budget: int, early_stopping: EarlyStopping,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a full checkpoint before reusing its trajectory, not just its weights."""
    state = load_training_checkpoint(path)
    saved = state.get("config")
    if not isinstance(saved, dict) or (_RESUME_TRAINING_KEYS - {"precision"}) - saved.keys():
        raise ValueError(f"Checkpoint has no complete HPO parameters: {path}")
    if state.get("task", BINARY_TASK) != BINARY_TASK:
        raise ValueError(f"Unsupported checkpoint task {state.get('task')!r}: {path}")
    if state.get("model_name") != model_name or state.get("world_size", 1) != 1:
        raise ValueError(f"Checkpoint model/world size differs from this independent HPO: {path}")
    if parameters is not None and _resume_parameters(saved) != _resume_parameters(parameters):
        raise ValueError(f"Checkpoint training parameters differ from the requested trial: {path}")
    if state.get("precision", "fp32") != saved.get("precision", "fp32"):
        raise ValueError(f"Checkpoint precision metadata is inconsistent: {path}")
    if saved.get("thresholds", state.get("thresholds")) != list(thresholds):
        raise ValueError(f"Checkpoint validation thresholds differ: {path}")
    if "dataset_root" in saved:
        if Path(saved["dataset_root"]).resolve() != Path(dataset_root).resolve():
            raise ValueError(f"Checkpoint dataset path differs: {path}")
    else:
        warnings.warn(
            f"Legacy checkpoint {path} has no dataset path: verify it uses the same dataset.",
            RuntimeWarning, stacklevel=2,
        )
    epoch = state["epoch"]
    history = state.get("history", [])
    if epoch > budget or [row.get("epoch") for row in history] != list(range(1, epoch + 1)):
        raise ValueError(f"Checkpoint exceeds the target budget or has incomplete history: {path}")
    if not state.get("rng_states"):
        raise ValueError(f"Checkpoint has no RNG/loader state for continuation: {path}")
    expected_best = max(history, key=lambda row: row["validation_f1"])
    if state.get("best_metrics", {}).get("epoch") != expected_best["epoch"]:
        raise ValueError(f"Checkpoint best epoch disagrees with history: {path}")
    stopper = EarlyStopping(
        early_stopping.patience,
        early_stopping.min_epochs,
        early_stopping.min_delta,
        monitor=early_stopping.monitor,
    )
    stopper.load_state_dict(state.get("early_stopping"), epoch)
    if (saved["scheduler"] == "none") != (state.get("scheduler_state_dict") is None):
        raise ValueError(f"Checkpoint scheduler state is inconsistent: {path}")
    return state


def _prepare_trial_resume(
    source: Path | None, directory: Path, parameters: dict[str, Any],
    model_name: str, dataset_root: str | Path, thresholds: list[float],
    budget: int, early_stopping: EarlyStopping,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Fork a full snapshot into this attempt; source files are never changed."""
    if source is None:
        return None, []
    state = read_hpo_checkpoint(
        source, parameters=parameters, model_name=model_name, dataset_root=dataset_root,
        thresholds=thresholds, budget=budget, early_stopping=early_stopping,
    )
    state["best_model_state_dict"] = _best_weights(state, source)
    state["run_name"] = directory.name
    state["config"] = deepcopy(parameters)
    state["resumed_from"] = str(source.resolve())
    # A previously running epoch can itself be the new absolute target.
    restored_stopper = EarlyStopping(
        early_stopping.patience,
        early_stopping.min_epochs,
        early_stopping.min_delta,
        monitor=early_stopping.monitor,
    )
    restored_stopper.load_state_dict(state.get("early_stopping"), state["epoch"])
    reason = (
        "max_epochs" if state["epoch"] == budget
        else "early_stopping" if restored_stopper.should_stop
        else None
    )
    state["stop_reason"] = reason
    state["history"][-1]["stop_reason"] = reason
    state["current_metrics"]["stop_reason"] = reason
    destination = directory / "last.pt"
    restore_best_checkpoint(state, source, directory / "best.pt")
    _save_checkpoint(destination, state)
    return destination, state["history"]


def _select_trial_resume(
    directory: Path, explicit: str | Path | None, resume_incomplete: bool,
) -> Path | None:
    if resume_incomplete:
        local = find_training_checkpoint(directory)
        if local is not None:
            return local[0]  # OOM retry uses new progress, not the original external snapshot.
    return Path(explicit) if explicit is not None else None


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_single_trial(
    trial_number: int,
    parameters: dict[str, Any],
    model_name: str,
    dataset_root: str | Path,
    thresholds: list[float],
    number_of_epochs: int,
    search_output_root: str | Path,
    gpu_id: int,
    checkpoint_interval: int,
    *,
    patience: int | None = None,
    min_epochs: int = 1,
    min_delta: float = 0.0,
    resume_incomplete: bool = True,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run one independent training trial, optionally with early stopping."""
    task = parameters.get("task", BINARY_TASK)
    if task != BINARY_TASK:
        raise ValueError(f"Unsupported trial task: {task}")
    score_key = "validation_f1"
    early_stopping = EarlyStopping(patience, min_epochs, min_delta, monitor=score_key)
    early_stopping.validate_budget(number_of_epochs)


    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = resolve_device(f"cuda:{gpu_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(
        f"[Trial {trial_number:03d} | GPU {gpu_id}] "
        f"deterministic_algorithms="
        f"{torch.are_deterministic_algorithms_enabled()} | "
        f"cudnn_deterministic="
        f"{torch.backends.cudnn.deterministic} | "
        f"cudnn_benchmark="
        f"{torch.backends.cudnn.benchmark}"
    )

    seed = int(
        parameters["seed"]
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    training_transform = RandomFlipRotate(
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.5,
    )

    _, train_loader = (
        create_dataset_dataloader(
            dataset_root=dataset_root,
            split="train",
            image_size=parameters[
                "image_size"
            ],
            batch_size=parameters[
                "batch_size"
            ],
            num_workers=parameters[
                "num_workers"
            ],
            transform=training_transform,
            shuffle=True,
            pin_memory=True,
            seed=seed,
        )
    )

    _, validation_loader = (
        create_dataset_dataloader(
            dataset_root=dataset_root,
            split="val",
            image_size=parameters[
                "image_size"
            ],
            batch_size=parameters[
                "batch_size"
            ],
            num_workers=parameters[
                "num_workers"
            ],
            pin_memory=True,
        )
    )

    model = create_task_model(
        model_name=model_name, task=task,
        parameters=parameters[
            "model_parameters"
        ],
    ).to(device)

    parameters["code_revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    parameters["torch_version"] = str(torch.__version__)
    parameters["cuda_version"] = torch.version.cuda
    parameters["thresholds"] = list(thresholds)
    parameters["dataset_root"] = str(Path(dataset_root).resolve())
    parameters["budget_epochs"] = number_of_epochs
    parameter_counts = count_model_parameters(model)

    loss_function: torch.nn.Module
    loss_function = BCEDiceLoss(
        bce_weight=parameters[
            "bce_weight"
        ],
        dice_weight=parameters[
            "dice_weight"
        ],
        pos_weight=parameters[
            "positive_weight"
        ],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parameters[
            "learning_rate"
        ],
        weight_decay=parameters[
            "weight_decay"
        ],
    )

    scheduler = _create_scheduler(
        optimizer=optimizer,
        scheduler_name=parameters[
            "scheduler"
        ],
    )

    run_name = (
        f"trial_{trial_number:03d}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.perf_counter()

    directory = _trial_directory(search_output_root, model_name, trial_number)
    source = _select_trial_resume(directory, resume_from, resume_incomplete)
    local_checkpoint, previous_history = _prepare_trial_resume(
        source, directory, parameters, model_name, dataset_root, thresholds,
        number_of_epochs, early_stopping,
    )
    saved_epoch = len(previous_history)
    already_stopped = bool(
        previous_history and previous_history[-1].get("stop_reason") == "early_stopping"
    )
    if saved_epoch == number_of_epochs or already_stopped:
        history = previous_history
    else:
        history = fit_binary_change_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            number_of_epochs=number_of_epochs - saved_epoch,
            thresholds=thresholds,
            output_root=search_output_root,
            model_name=model_name,
            run_name=run_name,
            checkpoint_interval=(
                checkpoint_interval
            ),
            config=parameters,
            precision=parameters.get("precision", "fp32"), task=task,
            resume_from=local_checkpoint,
            scheduler=scheduler,
            patience=patience,
            min_epochs=min_epochs,
            min_delta=min_delta,
            log_prefix=(
                f"[Trial {trial_number:03d} | "
                f"GPU {gpu_id}] "
            ),
        )
    # Pause the code and wait until GPU processes end
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    training_seconds = (
        time.perf_counter()
        - started_at
    )

    last_attempt_seconds = training_seconds
    if saved_epoch:
        training_seconds += sum(row.get("epoch_seconds", 0.0) for row in previous_history)

    best_result = max(
        history,
        key=lambda result: (
            result[score_key]
        ),
    )

    peak_gpu_memory_mb = max(
        result[
            "peak_gpu_memory_mb"
        ]
        for result in history
    )


    result = {
        "trial": trial_number,
        "status": "completed",
        "resumed_from": str(source.resolve()) if source is not None else None,
        "resumed_epoch": saved_epoch,
        "new_epochs_completed": len(history) - saved_epoch,
        "last_attempt_seconds": last_attempt_seconds,
        "timing_scope": "retained_epoch_compute_plus_last_attempt_wall"
        if saved_epoch
        else "attempt_wall",
        **_summarize_training_completion(history, number_of_epochs, early_stopping),
        "gpu_id": gpu_id,
        **parameter_counts,
        "parameters": parameters,
        "best_epoch": (best_result["epoch"]),
        **(
            {
                "best_threshold": (best_result["best_threshold"]),
                "best_validation_loss": (best_result["validation_loss"]),
                "best_precision": (best_result["validation_precision"]),
                "best_recall": (best_result["validation_recall"]),
                "best_f1": (best_result["validation_f1"]),
                "best_iou": (best_result["validation_iou"]),
            }
        ),
        "training_seconds": (training_seconds),
        "seconds_per_epoch": (training_seconds / len(history)),
        "gpu_hours": (training_seconds * (device.type == "cuda") / 3600.0),
        "peak_gpu_memory_mb": (peak_gpu_memory_mb),
    }

    trial_directory = (
        Path(search_output_root)
        / model_name
        / run_name
    )

    _write_json_atomic(trial_directory / "summary.json", result)

    del model
    del loss_function
    del optimizer
    del scheduler

    torch.cuda.empty_cache()

    return result


class ParameterSearchMemoryError(RuntimeError):
    """Raised after HPO finishes when a trial cannot fit on one GPU."""


_MEMORY_VALUE_PATTERN = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(GiB|MiB|KiB|GB|MB|KB|bytes?)",
    flags=re.IGNORECASE,
)


def _memory_value_to_mb(value: float, unit: str) -> float:
    """Convert a memory value to MiB-like megabytes used by CUDA reporting."""

    normalized_unit = unit.lower()

    if normalized_unit in {"gib", "gb"}:
        return value * 1024.0
    if normalized_unit in {"mib", "mb"}:
        return value
    if normalized_unit in {"kib", "kb"}:
        return value / 1024.0
    if normalized_unit in {"byte", "bytes"}:
        return value / (1024.0 ** 2)

    raise ValueError(f"Unsupported memory unit: {unit}")


def _extract_memory_mb(
    message: str,
    prefix: str,
    suffix: str = "",
) -> float | None:
    """Extract one CUDA-memory quantity from a PyTorch OOM message."""

    pattern = re.compile(
        re.escape(prefix)
        + r"\s*"
        + _MEMORY_VALUE_PATTERN.pattern
        + (r"\s*" + re.escape(suffix) if suffix else ""),
        flags=re.IGNORECASE,
    )
    match = pattern.search(message)
    if match is None:
        return None

    return _memory_value_to_mb(
        float(match.group(1)),
        match.group(2),
    )


def _parse_cuda_oom_message(message: str) -> dict[str, float | None]:
    """Extract useful memory values from the standard PyTorch CUDA OOM text."""

    tried_to_allocate_mb = _extract_memory_mb(
        message,
        "Tried to allocate",
    )

    allocated_match = re.search(
        _MEMORY_VALUE_PATTERN.pattern
        + r"\s+is allocated by PyTorch",
        message,
        flags=re.IGNORECASE,
    )
    allocated_by_pytorch_mb = None
    if allocated_match is not None:
        allocated_by_pytorch_mb = _memory_value_to_mb(
            float(allocated_match.group(1)),
            allocated_match.group(2),
        )

    process_match = re.search(
        r"this process has\s+"
        + _MEMORY_VALUE_PATTERN.pattern
        + r"\s+memory in use",
        message,
        flags=re.IGNORECASE,
    )
    process_memory_in_use_mb = None
    if process_match is not None:
        process_memory_in_use_mb = _memory_value_to_mb(
            float(process_match.group(1)),
            process_match.group(2),
        )

    return {
        "tried_to_allocate_mb": tried_to_allocate_mb,
        "allocated_by_pytorch_mb": allocated_by_pytorch_mb,
        "process_memory_in_use_mb": process_memory_in_use_mb,
    }


def _is_cuda_oom_error(error: BaseException) -> bool:
    """Return whether an exception represents a CUDA out-of-memory failure."""

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True

    message = str(error).lower()
    return (
        "cuda out of memory" in message
        or "cuda error: out of memory" in message
    )


def _cuda_memory_snapshot_mb(
    device: torch.device | int,
) -> dict[str, float | None]:
    """Return current CUDA memory information without failing the HPO scheduler."""

    snapshot: dict[str, float | None] = {
        "free_mb": None,
        "total_mb": None,
        "allocated_mb": None,
        "reserved_mb": None,
        "peak_allocated_mb": None,
        "peak_reserved_mb": None,
    }

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot["free_mb"] = free_bytes / (1024.0 ** 2)
        snapshot["total_mb"] = total_bytes / (1024.0 ** 2)
    except Exception:
        pass

    try:
        snapshot["allocated_mb"] = (
            torch.cuda.memory_allocated(device) / (1024.0 ** 2)
        )
        snapshot["reserved_mb"] = (
            torch.cuda.memory_reserved(device) / (1024.0 ** 2)
        )
        snapshot["peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        )
        snapshot["peak_reserved_mb"] = (
            torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
        )
    except Exception:
        pass

    return snapshot


def _estimate_oom_requirement_mb(
    error_message: str,
    snapshot: dict[str, float | None],
) -> tuple[float | None, str]:
    """Estimate the single-GPU memory required at the point of failure."""

    parsed = _parse_cuda_oom_message(error_message)
    tried_to_allocate_mb = parsed["tried_to_allocate_mb"]
    allocated_by_pytorch_mb = parsed["allocated_by_pytorch_mb"]
    process_memory_in_use_mb = parsed["process_memory_in_use_mb"]

    if (
        allocated_by_pytorch_mb is not None
        and tried_to_allocate_mb is not None
    ):
        return (
            allocated_by_pytorch_mb + tried_to_allocate_mb,
            "oom_message_allocated_plus_request",
        )

    if (
        process_memory_in_use_mb is not None
        and tried_to_allocate_mb is not None
    ):
        return (
            process_memory_in_use_mb + tried_to_allocate_mb,
            "oom_message_process_plus_request",
        )

    peak_allocated_mb = snapshot.get("peak_allocated_mb")
    if peak_allocated_mb is not None:
        return (
            peak_allocated_mb,
            "cuda_peak_allocated",
        )

    return None, "unknown"


def _run_single_trial_with_oom_handling(
    trial_number: int,
    parameters: dict[str, Any],
    model_name: str,
    dataset_root: str | Path,
    thresholds: list[float],
    number_of_epochs: int,
    search_output_root: str | Path,
    gpu_id: int,
    checkpoint_interval: int,
    max_trial_memory_mb: float,
    free_memory_before_mb: float | None,
    *,
    patience: int | None = None,
    min_epochs: int = 1,
    min_delta: float = 0.0,
    resume_incomplete: bool = True,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run one trial and convert CUDA OOM into a scheduler-readable result.

    Retries resume the last saved epoch by default, preserving the stopping policy.
    Before the first saved epoch, an OOM retry necessarily starts from scratch.
    A normal early stop remains status="completed", not an OOM or a failure.
    """

    # Do not touch CUDA before _run_single_trial. It sets deterministic CUDA
    # environment variables before creating the CUDA context.
    device = torch.device(f"cuda:{gpu_id}")
    started_at = time.perf_counter()
    oom_result: dict[str, Any] | None = None

    try:
        return _run_single_trial(
            trial_number=trial_number,
            parameters=parameters,
            model_name=model_name,
            dataset_root=dataset_root,
            thresholds=thresholds,
            number_of_epochs=number_of_epochs,
            search_output_root=search_output_root,
            gpu_id=gpu_id,
            checkpoint_interval=checkpoint_interval,
            patience=patience,
            min_epochs=min_epochs,
            min_delta=min_delta,
            resume_incomplete=resume_incomplete,
            resume_from=resume_from,
        )
    except Exception as error:
        if not _is_cuda_oom_error(error):
            raise

        error_message = str(error)
        snapshot = _cuda_memory_snapshot_mb(device)
        estimated_required_memory_mb, estimate_source = (
            _estimate_oom_requirement_mb(
                error_message=error_message,
                snapshot=snapshot,
            )
        )

        capacity_exceeded = (
            estimated_required_memory_mb is not None
            and estimated_required_memory_mb > max_trial_memory_mb
        )

        oom_result = {
            "trial": trial_number,
            "status": "cuda_oom",
            "gpu_id": gpu_id,
            "parameters": parameters,
            "error": error_message,
            "attempt_seconds": time.perf_counter() - started_at,
            "free_memory_before_mb": free_memory_before_mb,
            "total_memory_mb": snapshot.get("total_mb"),
            "allocated_memory_mb": snapshot.get("allocated_mb"),
            "reserved_memory_mb": snapshot.get("reserved_mb"),
            "peak_allocated_memory_mb": snapshot.get("peak_allocated_mb"),
            "peak_reserved_memory_mb": snapshot.get("peak_reserved_mb"),
            "estimated_required_memory_mb": estimated_required_memory_mb,
            "memory_estimate_source": estimate_source,
            "max_trial_memory_mb": max_trial_memory_mb,
            "capacity_exceeded": capacity_exceeded,
        }

        print(
            f"[OOM] Trial {trial_number:03d} on GPU {gpu_id} | "
            f"estimated_required_memory_mb={estimated_required_memory_mb} | "
            f"limit_mb={max_trial_memory_mb}",
            flush=True,
        )

    # The exception traceback is no longer live here, so model tensors from the
    # failed frame can be released before the worker returns to the executor.
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    if oom_result is None:
        raise RuntimeError("CUDA OOM handling ended without an OOM result.")

    return oom_result


def _available_gpu_memory_mb(gpu_id: int) -> float | None:
    """Return currently free memory for one visible GPU."""

    try:
        with torch.cuda.device(gpu_id):
            free_bytes, _ = torch.cuda.mem_get_info()
    except Exception:
        return None

    return free_bytes / (1024.0 ** 2)


def _trial_directory(
    search_output_root: str | Path,
    model_name: str,
    trial_number: int,
) -> Path:
    """Return the output directory used by one HPO trial."""

    return (
        Path(search_output_root)
        / model_name
        / f"trial_{trial_number:03d}"
    )


def _archive_partial_trial_output(
    search_output_root: str | Path, model_name: str, trial_number: int
) -> None:
    """Preserve an abandoned attempt when recovery was explicitly disabled."""
    directory = _trial_directory(search_output_root, model_name, trial_number)
    if directory.exists():
        archive = directory.parent / "abandoned_attempts"
        archive.mkdir(exist_ok=True)
        directory.rename(archive / f"{directory.name}_{uuid4().hex[:12]}")


def _write_trial_summary(
    search_output_root: str | Path,
    model_name: str,
    trial_number: int,
    result: dict[str, Any],
) -> None:
    """Persist a completed or failed trial summary."""

    directory = _trial_directory(
        search_output_root=search_output_root,
        model_name=model_name,
        trial_number=trial_number,
    )
    directory.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(directory / "summary.json", result)


def run_parameter_search(
    search_space: dict[str, list[Any]] | list[dict[str, list[Any]]],
    fixed_parameters: dict[str, Any],
    model_name: str,
    dataset_root: str | Path,
    thresholds: list[float],
    number_of_epochs: int,
    output_root: str | Path,
    search_name: str,
    gpu_ids: list[int],
    checkpoint_interval: int = 5,
    max_trial_memory_mb: float = 80_000.0,
    oom_retry_delay_seconds: float = 30.0,
    oom_retry_memory_margin_mb: float = 512.0,
    *,
    patience: int | None = None,
    min_epochs: int = 1,
    min_delta: float = 0.0,
    resume_incomplete: bool = True,
    resume_from: dict[int, str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Run HPO trials with optional early stopping and transient CUDA OOM retries.

    number_of_epochs is the maximum per trial. patience=None preserves the
    fixed-budget behavior. Stopping settings are shared across trials, not
    search-space parameters. number_of_epochs is an ABSOLUTE target even on resume.
    resume_from maps local trial indices to full snapshots; all parameters must match.
    Interrupted groups are recovered by the automated controller in new directories.
    Only transient CUDA OOM is retried automatically; other errors propagate with
    checkpoints preserved. Never load checkpoints from an untrusted source.

    CUDA OOM handling has two paths:
    - If the estimated requirement is within ``max_trial_memory_mb``, the trial
      is re-queued and retried on another allowed GPU when enough memory is free.
    - If the estimated requirement exceeds the configured single-GPU limit,
      that trial is marked failed while the remaining search-space trials keep
      running. After every other trial finishes, ``ParameterSearchMemoryError``
      is raised and the full search summary is already written to disk.
    """
    search_trials = _expand_search_space(search_space)
    task = _build_trial_parameters(fixed_parameters, search_trials[0]).get("task", BINARY_TASK)
    if task != BINARY_TASK:
        raise ValueError(f"Unsupported search task: {task}")
    early_stopping = EarlyStopping(patience, min_epochs, min_delta, monitor="validation_f1")
    early_stopping.validate_budget(number_of_epochs)
    if type(resume_incomplete) is not bool:
        raise ValueError("resume_incomplete must be a boolean.")
    resume_from = {} if resume_from is None else dict(resume_from)

    if torch.distributed.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Independent HPO cannot run inside torchrun/DDP.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must be unique.")
    if fixed_parameters.get("execution_mode", "independent_hpo") == "ddp":
        raise ValueError("HPO trials must be independent single-device runs.")
    if max_trial_memory_mb <= 0:
        raise ValueError("max_trial_memory_mb must be greater than zero.")
    if oom_retry_delay_seconds <= 0:
        raise ValueError("oom_retry_delay_seconds must be greater than zero.")
    if oom_retry_memory_margin_mb < 0:
        raise ValueError("oom_retry_memory_margin_mb cannot be negative.")

    model_name = model_name.strip().lower()
    model_class = model_class_for_task(model_name, task)

    trial_space = []

    for trial_number, search_parameters in enumerate(search_trials):
        parameters = _build_trial_parameters(
            fixed_parameters=fixed_parameters,
            search_parameters=search_parameters,
        )

        if parameters.get("task", BINARY_TASK) != task:
            raise ValueError("A queue cannot mix different training tasks.")
        allowed_training = {
            "model_parameters", "seed", "image_size", "batch_size", "num_workers",
            "learning_rate", "weight_decay", "scheduler", "positive_weight",
            "bce_weight", "dice_weight", "precision", "execution_mode",
        }
        unknown_training = parameters.keys() - allowed_training
        if unknown_training:
            raise ValueError(f"Unsupported trial parameters: {sorted(unknown_training)}")
        if parameters.get("precision", "fp32") not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16.")

        supported = inspect.signature(model_class).parameters
        unknown = parameters["model_parameters"].keys() - supported.keys()
        if unknown:
            raise ValueError(f"Unsupported architecture parameters: {sorted(unknown)}")

        parameters["execution_mode"] = "independent_hpo"
        parameters["world_size"] = 1
        parameters["precision"] = parameters.get("precision", "fp32")
        if "encoder_depths" in parameters["model_parameters"]:
            parameters["architecture_variant"] = (
                "canonical_mit_b3"
                if tuple(parameters["model_parameters"]["encoder_depths"]) == (3, 4, 18, 3)
                else "depth_ablation"
            )

        trial_space.append((trial_number, parameters))

    if any(type(index) is not int or not 0 <= index < len(trial_space) for index in resume_from):
        raise ValueError("resume_from keys must be local trial indices in this search grid.")
    for path in resume_from.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    output_root = Path(output_root)
    search_output_root = output_root / search_name

    if search_output_root.exists():
        raise FileExistsError(
            f"Choose a new search_name; output exists: {search_output_root}"
        )

    visible_gpu_count = torch.cuda.device_count()

    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")

    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= visible_gpu_count:
            raise ValueError(
                f"GPU {gpu_id} is not available. "
                f"Visible GPU count: {visible_gpu_count}."
            )

    selected_gpu_capacities_mb = {
        gpu_id: (
            torch.cuda.get_device_properties(gpu_id).total_memory
            / (1024.0 ** 2)
        )
        for gpu_id in gpu_ids
    }
    largest_selected_gpu_mb = max(selected_gpu_capacities_mb.values())
    effective_memory_limit_mb = min(
        max_trial_memory_mb,
        largest_selected_gpu_mb,
    )

    search_output_root.mkdir(parents=True, exist_ok=False)

    print("Trials:", len(trial_space))
    print("GPUs:", gpu_ids)
    print(
        "Single-GPU memory limit:",
        f"{effective_memory_limit_mb:.0f} MB",
    )

    context = mp.get_context("spawn")
    executors = {
        gpu_id: ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
        )
        for gpu_id in gpu_ids
    }

    def restart_gpu_executor(gpu_id: int) -> None:
        """Restart one worker after OOM so its CUDA context releases memory."""

        executors[gpu_id].shutdown(wait=True)
        executors[gpu_id] = ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
        )

    pending_trials = [
        {
            "trial_number": trial_number,
            "parameters": parameters,
            "blocked_gpus": set(),
            "not_before": 0.0,
            "minimum_free_memory_mb": 0.0,
            "oom_retries": 0,
            "attempts": [],
        }
        for trial_number, parameters in trial_space
    ]

    active_futures: dict[Any, tuple[int, dict[str, Any]]] = {}
    idle_gpus = set(gpu_ids)
    results: list[dict[str, Any]] = []
    failed_trials: list[dict[str, Any]] = []
    last_wait_log_at = 0.0

    def select_pending_trial(
        gpu_id: int,
        free_memory_mb: float | None,
    ) -> dict[str, Any] | None:
        now = time.monotonic()

        for index, state in enumerate(pending_trials):
            if state["not_before"] > now:
                continue
            if gpu_id in state["blocked_gpus"]:
                continue

            minimum_free_memory_mb = state["minimum_free_memory_mb"]
            if (
                free_memory_mb is not None
                and minimum_free_memory_mb > 0
                and free_memory_mb < minimum_free_memory_mb
            ):
                continue

            return pending_trials.pop(index)

        return None

    def submit_trial(
        gpu_id: int,
        state: dict[str, Any],
        free_memory_mb: float | None,
    ) -> None:
        trial_number = int(state["trial_number"])
        parameters = state["parameters"]

        if state["oom_retries"] > 0 and not resume_incomplete:
            _archive_partial_trial_output(
                search_output_root=search_output_root,
                model_name=model_name,
                trial_number=trial_number,
            )

        attempt_number = len(state["attempts"]) + 1
        state["attempts"].append(
            {
                "attempt": attempt_number,
                "gpu_id": gpu_id,
                "free_memory_before_mb": free_memory_mb,
                "status": "running",
            }
        )

        label = "RETRY" if state["oom_retries"] > 0 else "START"
        print(
            f"\n{'=' * 100}\n"
            f"[{label}] Trial {trial_number:03d} -> GPU {gpu_id} "
            f"| attempt={attempt_number}\n"
            f"{_format_trial_parameters(parameters)}\n"
            f"epochs={number_of_epochs}\n"
            f"{'=' * 100}",
            flush=True,
        )

        future = executors[gpu_id].submit(
            _run_single_trial_with_oom_handling,
            trial_number,
            parameters,
            model_name,
            dataset_root,
            thresholds,
            number_of_epochs,
            search_output_root,
            gpu_id,
            checkpoint_interval,
            effective_memory_limit_mb,
            free_memory_mb,
            patience=patience,
            min_epochs=min_epochs,
            min_delta=min_delta,
            resume_incomplete=resume_incomplete,
            resume_from=resume_from.get(trial_number),
        )

        active_futures[future] = (gpu_id, state)
        idle_gpus.discard(gpu_id)

    def schedule_idle_gpus() -> bool:
        scheduled_any = False
        gpu_free_memory = {
            gpu_id: _available_gpu_memory_mb(gpu_id)
            for gpu_id in idle_gpus
        }

        ordered_gpus = sorted(
            idle_gpus,
            key=lambda gpu_id: (
                gpu_free_memory[gpu_id]
                if gpu_free_memory[gpu_id] is not None
                else float("inf")
            ),
            reverse=True,
        )

        for gpu_id in ordered_gpus:
            free_memory_mb = gpu_free_memory[gpu_id]
            state = select_pending_trial(
                gpu_id=gpu_id,
                free_memory_mb=free_memory_mb,
            )
            if state is None:
                continue

            submit_trial(
                gpu_id=gpu_id,
                state=state,
                free_memory_mb=free_memory_mb,
            )
            scheduled_any = True

        return scheduled_any

    try:
        while pending_trials or active_futures:
            scheduled_any = schedule_idle_gpus()

            if active_futures:
                done, _ = wait(
                    active_futures,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )

                if not done:
                    continue

                for future in done:
                    gpu_id, state = active_futures.pop(future)
                    idle_gpus.add(gpu_id)

                    result = future.result()
                    attempt = state["attempts"][-1]

                    if result["status"] == "completed":
                        attempt["status"] = "completed"
                        state["blocked_gpus"].clear()

                        result["oom_retries"] = state["oom_retries"]
                        result["gpu_attempts"] = deepcopy(state["attempts"])
                        results.append(result)

                        _write_trial_summary(
                            search_output_root=search_output_root,
                            model_name=model_name,
                            trial_number=result["trial"],
                            result=result,
                        )

                        print()
                        print(
                            f"Trial {result['trial']} finished on GPU {gpu_id}"
                        )
                        print(
                            "Parameters:",
                            _format_trial_parameters(result["parameters"]),
                        )
                        print(
                            "Model parameters:",
                            f"total={result['total_parameters']:,} | "
                            f"trainable={result['trainable_parameters']:,}",
                        )
                        print("F1:", f"{result['best_f1']:.6f}")
                        print("IoU:", f"{result['best_iou']:.6f}")
                        print(
                            "Epochs:",
                            f"{result['epochs_completed']}/{result['budget_epochs']} "
                            f"({result['stop_reason']})",
                        )
                        print("GPU hours:", f"{result['gpu_hours']:.3f}")
                        continue

                    if result["status"] != "cuda_oom":
                        raise RuntimeError(
                            f"Unexpected trial status: {result['status']}"
                        )

                    # A fresh process prevents PyTorch's allocator/cache from
                    # keeping memory reserved on the GPU after an OOM.
                    restart_gpu_executor(gpu_id)

                    attempt.update(
                        {
                            "status": "cuda_oom",
                            "attempt_seconds": result["attempt_seconds"],
                            "estimated_required_memory_mb": result[
                                "estimated_required_memory_mb"
                            ],
                            "memory_estimate_source": result[
                                "memory_estimate_source"
                            ],
                        }
                    )
                    state["oom_retries"] += 1

                    if result["capacity_exceeded"]:
                        failure = {
                            "trial": result["trial"],
                            "status": "failed_cuda_oom_capacity",
                            "gpu_id": gpu_id,
                            "parameters": result["parameters"],
                            "error": result["error"],
                            "estimated_required_memory_mb": result[
                                "estimated_required_memory_mb"
                            ],
                            "memory_estimate_source": result[
                                "memory_estimate_source"
                            ],
                            "max_trial_memory_mb": effective_memory_limit_mb,
                            "oom_retries": state["oom_retries"],
                            "gpu_attempts": deepcopy(state["attempts"]),
                        }
                        failed_trials.append(failure)

                        _write_trial_summary(
                            search_output_root=search_output_root,
                            model_name=model_name,
                            trial_number=failure["trial"],
                            result=failure,
                        )

                        print(
                            f"[FAILED] Trial {failure['trial']:03d} requires "
                            f"approximately "
                            f"{failure['estimated_required_memory_mb']:.0f} MB, "
                            f"above the single-GPU limit of "
                            f"{effective_memory_limit_mb:.0f} MB. "
                            "The remaining HPO trials will continue.",
                            flush=True,
                        )
                        continue

                    estimated_required_memory_mb = result[
                        "estimated_required_memory_mb"
                    ]
                    free_before_mb = result["free_memory_before_mb"]

                    if estimated_required_memory_mb is not None:
                        minimum_free_memory_mb = (
                            estimated_required_memory_mb
                            + oom_retry_memory_margin_mb
                        )
                    elif free_before_mb is not None:
                        minimum_free_memory_mb = (
                            free_before_mb
                            + oom_retry_memory_margin_mb
                        )
                    else:
                        minimum_free_memory_mb = 0.0

                    state["minimum_free_memory_mb"] = max(
                        state["minimum_free_memory_mb"],
                        min(
                            minimum_free_memory_mb,
                            effective_memory_limit_mb,
                        ),
                    )
                    state["blocked_gpus"].add(gpu_id)

                    other_gpu_ids = [
                        candidate_gpu_id
                        for candidate_gpu_id in gpu_ids
                        if candidate_gpu_id not in state["blocked_gpus"]
                    ]

                    if other_gpu_ids:
                        state["not_before"] = time.monotonic()
                        print(
                            f"[RETRY] Trial {result['trial']:03d} hit CUDA OOM "
                            f"on GPU {gpu_id}. It will be retried on another "
                            "available GPU.",
                            flush=True,
                        )
                    else:
                        state["blocked_gpus"].clear()
                        state["not_before"] = (
                            time.monotonic()
                            + oom_retry_delay_seconds
                        )
                        print(
                            f"[WAIT] Trial {result['trial']:03d} hit CUDA OOM "
                            "on every allowed GPU during this retry cycle. "
                            f"Waiting {oom_retry_delay_seconds:.0f}s for GPU "
                            "memory to become available before trying again.",
                            flush=True,
                        )

                    pending_trials.insert(0, state)

            elif pending_trials and not scheduled_any:
                now = time.monotonic()
                if now - last_wait_log_at >= oom_retry_delay_seconds:
                    retry_state = pending_trials[0]
                    required_mb = retry_state["minimum_free_memory_mb"]
                    print(
                        f"[WAIT] Trial {retry_state['trial_number']:03d} is "
                        "waiting for a GPU with enough free CUDA memory"
                        + (
                            f" (about {required_mb:.0f} MB required)."
                            if required_mb > 0
                            else "."
                        ),
                        flush=True,
                    )
                    last_wait_log_at = now

                time.sleep(
                    min(1.0, oom_retry_delay_seconds)
                )

    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)

    results.sort(
        key=lambda result: result["best_f1"],
        reverse=True,
    )

    total_gpu_hours = sum(
        result["gpu_hours"]
        for result in results
    )

    summary = {
        "early_stopping": early_stopping.policy,
        "budget_epochs": number_of_epochs,
        "model_name": model_name,
        "search_name": search_name,
        "number_of_trials": len(trial_space),
        "number_completed": len(results),
        "number_failed": len(failed_trials),
        "max_trial_memory_mb": effective_memory_limit_mb,
        "total_gpu_hours": total_gpu_hours,
        "results": results,
        "failed_trials": failed_trials,
    }

    _write_json_atomic(search_output_root / "results.json", summary)

    if failed_trials:
        failed_descriptions = "; ".join(
            (
                f"trial {failure['trial']:03d} "
                f"(~{failure['estimated_required_memory_mb']:.0f} MB)"
            )
            for failure in failed_trials
        )
        raise ParameterSearchMemoryError(
            "Hyperparameter search finished, but one or more trials could not "
            "be trained because their estimated CUDA memory requirement "
            f"exceeded the single-GPU limit of {effective_memory_limit_mb:.0f} MB: "
            f"{failed_descriptions}. Other search-space trials were allowed to "
            f"finish. Results were saved to {search_output_root / 'results.json'}."
        )

    return results
