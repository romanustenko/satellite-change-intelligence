"""Hyperparameter-search utilities."""

import inspect
import json
import multiprocessing as mp
import os
import random
import subprocess
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from satchange.data import (
    RandomFlipRotate,
    create_dataset_dataloader,
)
from satchange.losses import BCEDiceLoss
from satchange.models.factory import (
    MODEL_REGISTRY,
    count_model_parameters,
    create_change_model,
)
from satchange.training import (
    fit_binary_change_model,
)


def _expand_search_space(
    search_space: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Create all combinations from a search space."""

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
) -> dict[str, Any]:
    """Run one independent training trial."""

    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        f"cuda:{gpu_id}"
    )

    torch.cuda.set_device(
        device
    )

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

    model = create_change_model(
        model_name=model_name,
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
    parameter_counts = count_model_parameters(model)

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

    torch.cuda.reset_peak_memory_stats(
        device
    )

    started_at = time.perf_counter()

    history = fit_binary_change_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        number_of_epochs=number_of_epochs,
        thresholds=thresholds,
        output_root=search_output_root,
        model_name=model_name,
        run_name=run_name,
        checkpoint_interval=(
            checkpoint_interval
        ),
        config=parameters,
        precision=parameters.get("precision", "fp32"),
        resume_from=None,
        scheduler=scheduler,
        log_prefix=(
            f"[Trial {trial_number:03d} | "
            f"GPU {gpu_id}] "
        ),
    )
    # Pause the code and wait until GPU processes end
    torch.cuda.synchronize(
        device
    )

    training_seconds = (
        time.perf_counter()
        - started_at
    )

    best_result = max(
        history,
        key=lambda result: (
            result["validation_f1"]
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
        "gpu_id": gpu_id,
        **parameter_counts,
        "parameters": parameters,
        "best_epoch": (
            best_result["epoch"]
        ),
        "best_threshold": (
            best_result[
                "best_threshold"
            ]
        ),
        "best_validation_loss": (
            best_result[
                "validation_loss"
            ]
        ),
        "best_precision": (
            best_result[
                "validation_precision"
            ]
        ),
        "best_recall": (
            best_result[
                "validation_recall"
            ]
        ),
        "best_f1": (
            best_result[
                "validation_f1"
            ]
        ),
        "best_iou": (
            best_result[
                "validation_iou"
            ]
        ),
        "training_seconds": (
            training_seconds
        ),
        "seconds_per_epoch": (
            training_seconds
            / len(history)
        ),
        "gpu_hours": (
            training_seconds
            / 3600.0
        ),
        "peak_gpu_memory_mb": (
            peak_gpu_memory_mb
        ),
    }

    trial_directory = (
        Path(search_output_root)
        / model_name
        / run_name
    )

    with (
        trial_directory
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            default=str,
        )

    del model
    del loss_function
    del optimizer
    del scheduler

    torch.cuda.empty_cache()

    return result


def run_parameter_search(
    search_space: dict[str, list[Any]],
    fixed_parameters: dict[str, Any],
    model_name: str,
    dataset_root: str | Path,
    thresholds: list[float],
    number_of_epochs: int,
    output_root: str | Path,
    search_name: str,
    gpu_ids: list[int],
    checkpoint_interval: int = 5,
) -> list[dict[str, Any]]:
    """Run hyperparameter trials across multiple GPUs."""

    if torch.distributed.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Independent HPO cannot run inside torchrun/DDP.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must be unique.")
    if fixed_parameters.get("execution_mode", "independent_hpo") == "ddp":
        raise ValueError("HPO trials must be independent single-device runs.")
    model_name = model_name.strip().lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    search_trials = (
        _expand_search_space(
            search_space
        )
    )

    trial_space = []

    for trial_number, search_parameters in enumerate(
    search_trials
    ):
        parameters = _build_trial_parameters(
            fixed_parameters=fixed_parameters,
            search_parameters=search_parameters,
        )

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
        supported = inspect.signature(MODEL_REGISTRY[model_name]).parameters
        unknown = parameters["model_parameters"].keys() - supported.keys()
        if unknown:
            raise ValueError(f"Unsupported architecture parameters: {sorted(unknown)}")
        parameters["execution_mode"] = "independent_hpo"
        parameters["world_size"] = 1
        parameters["precision"] = parameters.get("precision", "fp32")
        if "encoder_depths" in parameters["model_parameters"]:
            parameters["architecture_variant"] = (
                "canonical_mit_b3" if tuple(parameters["model_parameters"]["encoder_depths"])
                == (3, 4, 18, 3) else "depth_ablation")
        trial_space.append(
            (
                trial_number,
                parameters,
            )
        )

    output_root = Path(
        output_root
    )

    search_output_root = (
        output_root
        / search_name
    )

    if search_output_root.exists():
        raise FileExistsError(f"Choose a new search_name; output exists: {search_output_root}")

    visible_gpu_count = (
        torch.cuda.device_count()
    )



    for gpu_id in gpu_ids:
        if (
            gpu_id < 0
            or gpu_id
            >= visible_gpu_count
        ):
            raise ValueError(
                f"GPU {gpu_id} is not available. "
                f"Visible GPU count: "
                f"{visible_gpu_count}."
            )

    if not gpu_ids:
        raise ValueError(
            "At least one GPU ID is required."
        )

    search_output_root.mkdir(parents=True, exist_ok=False)

    print(
        "Trials:",
        len(trial_space),
    )

    print(
        "GPUs:",
        gpu_ids,
    )

    context = mp.get_context(
        "spawn"
    )

    executors = {
        gpu_id: ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
        )
        for gpu_id in gpu_ids
    }

    results = []
    active_futures = {}

    trial_iterator = iter(
        trial_space
    )

    def submit_next_trial(
        gpu_id: int,
    ) -> bool:
        try:
            (
                trial_number,
                parameters,
            ) = next(
                trial_iterator
            )
        except StopIteration:
            return False

        print(
            f"\n{'=' * 100}\n"
            f"[START] Trial {trial_number:03d} -> GPU {gpu_id}\n"
            f"{_format_trial_parameters(parameters)}\n"
            f"epochs={number_of_epochs}\n"
            f"{'=' * 100}",
            flush=True,
        )

        future = executors[
            gpu_id
        ].submit(
            _run_single_trial,
            trial_number,
            parameters,
            model_name,
            dataset_root,
            thresholds,
            number_of_epochs,
            search_output_root,
            gpu_id,
            checkpoint_interval,
        )

        active_futures[
            future
        ] = gpu_id

        return True

    try:
        for gpu_id in gpu_ids:
            submit_next_trial(
                gpu_id
            )

        while active_futures:
            done, _ = wait(
                active_futures,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                gpu_id = (
                    active_futures.pop(
                        future
                    )
                )

                result = future.result()

                results.append(
                    result
                )

                print()
                print(
                    f"Trial {result['trial']} "
                    f"finished on GPU {gpu_id}"
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

                print(
                    "F1:",
                    f"{result['best_f1']:.6f}",
                )

                print(
                    "IoU:",
                    f"{result['best_iou']:.6f}",
                )

                print(
                    "GPU hours:",
                    f"{result['gpu_hours']:.3f}",
                )

                submit_next_trial(
                    gpu_id
                )

    finally:
        for executor in (
            executors.values()
        ):
            executor.shutdown(
                wait=True
            )

    results.sort(
        key=lambda result: (
            result["best_f1"]
        ),
        reverse=True,
    )

    total_gpu_hours = sum(
        result["gpu_hours"]
        for result in results
    )

    summary = {
        "model_name": model_name,
        "search_name": search_name,
        "number_of_trials": (
            len(results)
        ),
        "total_gpu_hours": (
            total_gpu_hours
        ),
        "results": results,
    }

    with (
        search_output_root
        / "results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    return results
