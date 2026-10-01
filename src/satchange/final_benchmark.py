"""Finalize OWN trained checkpoints: validation, test, inference benchmark and export.

Uses the project's dataset, model factory, loss and evaluation pipeline.
Checkpoints are loaded with weights_only=False: do not pass untrusted files.
Nothing runs on import. No training, downloads or threshold search on test.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
import warnings
from collections import ChainMap, OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from satchange.data import create_dataset_dataloader
from satchange.execution import precision_context
from satchange.losses import BCEDiceLoss
from satchange.models import count_model_parameters, create_change_model
from satchange.models.factory import MODEL_REGISTRY
from satchange.training import evaluate_thresholds
from satchange.utils import load_config

__all__ = ["run_final_benchmark"]

_MISSING = object()
_METRICS = ("loss", "precision", "recall", "f1", "iou", "accuracy")
_TRAIN_KEYS = (
    "learning_rate", "weight_decay", "scheduler", "positive_weight",
    "bce_weight", "dice_weight", "smooth", "batch_size", "image_size",
    "precision", "epochs", "num_workers", "seed", "execution_mode", "world_size",
    "code_revision", "torch_version", "cuda_version", "thresholds",
)


# Public API


def run_final_benchmark(
    *,
    variant: str,
    checkpoints: Mapping[int, str | Path] | str | Path,
    project_root: str | Path | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
    model_name: str | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    image_size: int | None | Literal["auto"] = "auto",
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    metadata: Mapping[str, Any] | str | Path | None = None,
    training_results: Mapping[str, Any] | str | Path | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    thresholds: Mapping[int, float] | float | None = None,
    deployment_seed: int | None = None,
    evaluation_batch_size: int = 1,
    num_workers: int = 0,
    device: str | torch.device = "auto",
    precision: Literal["fp32", "bf16"] = "fp32",
    warmup: int = 20,
    repetitions: int = 100,
    strict_validation: bool = True,
    validation_atol: float = 1e-3,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate trained seeds and save artifacts/final_models/<variant>.

    Supply {seed: local best.pt path}, or a folder containing seed_*_best.pt.
    Saved checkpoint/summary/metadata values must agree; YAML fills missing values.
    Pass {} to disable automatic metadata/training_results loading.
    image_size=None keeps native scenes; 256 resizes whole scenes, not crops.

    All validation runs precede test. Thresholds must already be selected on
    validation (or fixed beforehand); no threshold search or training occurs.
    Deployment follows an explicit seed, saved metadata, or validation F1.
    Speed is measured separately at batch=1; sample std across seeds uses ddof=1.

    Only use your own trusted checkpoints: loading uses weights_only=False,
    as in the project's training/reconstruction code, and can execute pickle code.
    Seed checkpoints are copied unchanged; deployment is a CPU FP32 state_dict.
    overwrite=True backs up an existing folder only after the new bundle succeeds.
    Returns output_dir, backup_dir and the five saved JSON documents.
    """
    # 1. Resolve inputs without changing any existing results.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", variant):
        raise ValueError("variant must be a simple folder name, e.g. 'heavy_resized'.")
    for name, value, minimum in (
        ("evaluation_batch_size", evaluation_batch_size, 1), ("num_workers", num_workers, 0),
        ("warmup", warmup, 0), ("repetitions", repetitions, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not math.isfinite(validation_atol) or validation_atol < 0:
        raise ValueError("validation_atol must be finite and non-negative.")
    if torch.distributed.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Run final benchmarking in a single process, not under torchrun/DDP.")

    root = _root(project_root)
    paths, source_directory = _discover(checkpoints, root)
    device = _device(device)
    precision_context(device, precision)
    config_document = _read_mapping(config, root)
    output = _path(output_dir or f"artifacts/final_models/{variant}", root)
    prepared = config_document.get("dataset", {}).get("prepared_directory", "data/LEVIR-CD")
    dataset = _path(dataset_root or prepared, root)
    if output in (root, *root.parents, dataset, *dataset.parents):
        raise ValueError("Output must not be the project/dataset root or one of their ancestors.")
    for split in ("val", "test"):
        if not (dataset / split).is_dir():
            raise FileNotFoundError(f"Prepared dataset split is missing: {dataset / split}")
    if (dataset / "val").resolve() == (dataset / "test").resolve():
        raise ValueError("Validation and test resolve to the same directory.")

    threshold_map = ({_seed(seed): value for seed, value in thresholds.items()}
                     if isinstance(thresholds, Mapping) else dict.fromkeys(paths, thresholds))
    if threshold_map.keys() - paths.keys():
        raise ValueError("thresholds contains seeds that are absent from checkpoints.")
    explicit = _flatten(training_parameters or {})
    if model_name is not None:
        explicit["model_name"] = model_name.strip().lower()
    if model_parameters is not None:
        explicit["model_parameters"] = deepcopy(dict(model_parameters))
    if image_size != "auto":
        explicit["image_size"] = image_size

    notes: list[str] = []
    with _output_transaction(output, overwrite) as (stage, backup), _runtime(device):
        # 2. Copy checkpoints and resolve each seed's saved configuration.
        documents = {}
        for name, value in (("metadata", metadata), ("training_results", training_results)):
            default = source_directory / f"{name}.json" if source_directory else None
            documents[name] = _read_mapping(value, root, default=default)
        bundle = documents["metadata"]
        specs = _prepare_specs(
            paths, stage, bundle, documents["training_results"], _flatten(config_document),
            explicit, threshold_map, notes,
        )
        datasets = _prepare_data(dataset, specs, evaluation_batch_size, num_workers, device)
        validation_dataset, _ = datasets["validation"]
        sample = validation_dataset[0]
        shape = list(sample["before"].shape)
        print(f"\n{variant} | {specs[0].model_name} | {shape[-2]}x{shape[-1]} | "
              f"{precision.upper()} | {device} | seeds={list(paths)}", flush=True)

        # 3. Validate every seed, freeze deployment, then evaluate test.
        reports, selected, selection = _evaluate_seeds(
            specs, datasets, stage, variant=variant, device=device, precision=precision,
            batch_size=evaluation_batch_size, deployment_seed=deployment_seed, bundle=bundle,
            strict_validation=strict_validation, validation_atol=validation_atol, notes=notes,
        )
        # 4. Benchmark deployment and publish the same five JSON documents.
        benchmark = _hardware_benchmark(selected, sample, stage, device, precision, warmup, repetitions)
        reports["benchmark"] = benchmark
        reports["training_results"] = {
            **deepcopy(documents["training_results"]), "model_name": specs[0].model_name,
            "number_of_trials": len(specs), "results": [spec.training_row for spec in specs],
        }
        reports["metadata"] = _build_metadata(
            specs, reports["validation_results"], selected, selection, bundle, notes,
            variant=variant, dataset=dataset, shape=shape, precision=precision,
            batch_size=evaluation_batch_size, num_workers=num_workers,
            strict_validation=strict_validation, validation_atol=validation_atol, backup=backup,
        )
        for name in ("benchmark", "training_results", "metadata"):
            _write_json(stage / f"{name}.json", reports[name])
        for note in sorted(set(notes)):
            warnings.warn(note, stacklevel=2)

    print(f"\nSaved: {output}", flush=True)
    if backup:
        print(f"Previous folder preserved: {backup}", flush=True)
    for split, report_name in (("validation", "validation_results"), ("test", "final_test_results")):
        report = reports[report_name]
        deviation = report[f"{split}_f1_std"]
        text = f"{deviation:.6f}" if deviation is not None else "n/a (one seed)"
        print(f"{split:10s} F1 = {report[f'{split}_f1_mean']:.6f} ± {text}", flush=True)
    print(f"Deployment latency: {benchmark['latency_ms_per_pair']:.3f} ms/pair | "
          f"{benchmark['pairs_per_second']:.3f} pairs/s", flush=True)
    return {"output_dir": str(output), "backup_dir": str(backup) if backup else None, **reports}


# Input files and saved configuration


def _path(value: str | Path, root: Path) -> Path:
    if "://" in str(value):
        raise ValueError("Pass a local checkpoint path, not a URL.")
    # pathlib keeps an absolute right-hand path unchanged.
    return (root / Path(value).expanduser()).resolve()


def _root(value: str | Path | None) -> Path:
    if value is not None:
        result = Path(value).expanduser().resolve()
        if not result.is_dir():
            raise FileNotFoundError(f"Project root not found: {result}")
        return result
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "src/satchange/models/factory.py").is_file():
                return candidate
    raise ValueError("Cannot locate project root. Pass project_root= explicitly.")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Non-JSON metadata value: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_mapping(
    value: Mapping[str, Any] | str | Path | None, root: Path, *, default: Path | None = None,
) -> dict[str, Any]:
    """Read an optional document; an explicit {} disables the default sidecar."""
    if value is None and default is not None and default.is_file():
        value = default
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    path = _path(value, root)
    if path.suffix.lower() in (".yaml", ".yml"):
        return load_config(path)
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON/YAML mapping: {path}")
    return result


def _flatten(value: Mapping[str, Any]) -> dict[str, Any]:
    """Read both the project's nested YAML and its flat HPO checkpoint config."""
    result: dict[str, Any] = {}
    model = value.get("model", {})
    if isinstance(model, Mapping):
        if "name" in model:
            result["model_name"] = model["name"]
        if "parameters" in model:
            result["model_parameters"] = deepcopy(model["parameters"])
    for key in ("project", "runtime", "data", "training", "training_parameters"):
        section = value.get(key, {})
        if not isinstance(section, Mapping):
            continue
        result.update({k: deepcopy(v) for k, v in section.items() if k in _TRAIN_KEYS})
        loss = section.get("loss")
        if not isinstance(loss, Mapping):
            continue
        result.update({k: loss[k] for k in ("bce_weight", "dice_weight") if k in loss})
        # Missing fields in a saved YAML loss mean defaults, not fallback-run values.
        result["positive_weight"] = loss.get("pos_weight")
        result["smooth"] = loss.get("smooth", 1.0)
    result.update({k: deepcopy(v) for k, v in value.items()
                   if k in (*_TRAIN_KEYS, "model_name", "model_parameters")})
    if "pos_weight" in value:
        if "positive_weight" in result and result["positive_weight"] != value["pos_weight"]:
            raise ValueError("pos_weight and positive_weight disagree.")
        result["positive_weight"] = value["pos_weight"]
    return result


def _same(a: Any, b: Any) -> bool:
    """Compare saved JSON/YAML values, treating tuples and lists as equivalent."""
    return _jsonable(a) == _jsonable(b)


def _resolve(
    key: str, sources: list[tuple[str, Mapping[str, Any]]], fallback: Mapping[str, Any],
    *, default: Any = _MISSING, full_alias: bool = False,
) -> Any:
    """Require saved values to agree; only missing fields use fallback/defaults."""
    known = [(label, source[key]) for label, source in sources if key in source]
    if not known:
        value = fallback.get(key, default)
        if value is _MISSING:
            raise ValueError(f"Missing {key}. Supply it explicitly or provide saved config/metadata.")
        return deepcopy(value)
    first_label, value = known[0]
    for label, candidate in known[1:]:
        native_alias = full_alias and value in (None, 1024) and candidate in (None, 1024)
        if not native_alias and not _same(value, candidate):
            raise ValueError(
                f"Conflicting {key}: {first_label}={value!r}, {label}={candidate!r}. "
                "Use the metadata belonging to these trained checkpoints."
            )
    return deepcopy(value)


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError(f"Seed must be a non-negative integer, received {value!r}.")
    return int(value)


def _discover(
    checkpoints: Mapping[int, str | Path] | str | Path, root: Path,
) -> tuple[dict[int, Path], Path | None]:
    directory = None
    if isinstance(checkpoints, Mapping):
        entries = list(checkpoints.items())
    else:
        directory = _path(checkpoints, root)
        if not directory.is_dir():
            raise ValueError("Pass a directory, or {seed: checkpoint_path} for individual files.")
        entries = []
        for path in sorted(directory.glob("seed_*_best.pt")):
            match = re.fullmatch(r"seed_(\d+)_best\.pt", path.name)
            if match:
                entries.append((int(match[1]), path))
    resolved = {}
    for number, value in entries:
        number = _seed(number)
        path = _path(value, root)
        if path.is_dir():
            path = path / "best.pt"
        if number in resolved:
            raise ValueError(f"Duplicate seed: {number}")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint for seed {number}: {path}")
        with path.open("rb") as handle:
            if handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{path} is a Git LFS pointer, not downloaded weights.")
        resolved[number] = path
    if not resolved:
        raise ValueError("No checkpoints found. Expected seed_<integer>_best.pt files.")
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("The same checkpoint path was assigned to several seeds.")
    return dict(sorted(resolved.items())), directory


def _load_checkpoint(path: Path) -> tuple[OrderedDict, dict[str, Any]]:
    """Load OWN project checkpoints on CPU; full pickle loading is intentional."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected checkpoint/state_dict mapping, not a model object: {path}")
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        state, metadata = payload, {}
    else:
        state = payload.get("model_state_dict", payload.get("state_dict"))
        if state is None:
            raise ValueError(f"Checkpoint has no model_state_dict/state_dict: {path}")
        # Drop optimizer and RNG state immediately; inference does not need them.
        metadata = {key: deepcopy(payload[key]) for key in (
            "model_name", "config", "parameters", "model_parameters", "seed", "epoch",
            "run_name", "current_metrics", "best_metrics", "precision", "world_size",
        ) if key in payload}
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Empty or invalid state_dict: {path}")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("Only tensor state_dicts from the project models are supported.")
    weights = OrderedDict(state.items())
    if hasattr(state, "_metadata"):
        weights._metadata = deepcopy(state._metadata)
    for prefix in ("module.", "_orig_mod.", "module."):
        if all(key.startswith(prefix) for key in weights):
            consume_prefix_in_state_dict_if_present(weights, prefix)
    return weights, metadata


def _architecture(name: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {tuple(MODEL_REGISTRY)}")
    signature = inspect.signature(MODEL_REGISTRY[name])
    bound = signature.bind(**dict(parameters))
    bound.apply_defaults()
    return deepcopy(dict(bound.arguments))


def _instantiate(name: str, parameters: Mapping[str, Any], state: Mapping) -> nn.Module:
    # Initialization must not download external weights: the complete trained state follows.
    construction = deepcopy(dict(parameters))
    for key in ("pretrained_backbone", "use_checkpointing"):
        if key in construction:
            construction[key] = False
    model = create_change_model(name, construction).cpu().float()
    model.load_state_dict(state, strict=True)
    for key, value in model.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite model weights: {key}")
    return model.eval()


@dataclass
class _SeedSpec:
    seed: int
    checkpoint: Path
    source: Path
    model_name: str
    model_parameters: dict[str, Any]
    training_parameters: dict[str, Any]
    image_size: int | None
    threshold: float
    threshold_source: str
    original_threshold: float | None
    epoch: int | None
    original_f1: float | None
    original_iou: float | None
    trial: Any
    training_row: dict[str, Any]
    sha256: str
    full_alias_used: bool

    def load_model(self) -> nn.Module:
        """Rebuild on CPU so GPU memory holds only the current seed."""
        state, _ = _load_checkpoint(self.checkpoint)
        return _instantiate(self.model_name, self.model_parameters, state)


def _best_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Reject last/intermediate checkpoints that do not represent the best epoch."""
    best = checkpoint.get("best_metrics") or {}
    current = checkpoint.get("current_metrics") or {}
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return {**best, **current}

    if best.get("epoch", epoch) not in (None, epoch):
        raise ValueError(
            f"The supplied checkpoint is epoch {epoch}, but its best epoch is {best['epoch']}. "
            "Pass best.pt, not last.pt or an arbitrary intermediate checkpoint."
        )
    if current.get("epoch", epoch) not in (None, epoch):
        raise ValueError("Checkpoint current_metrics.epoch disagrees with checkpoint epoch.")
    return {**best, **current, "epoch": epoch}


def _resolve_spec(
    seed: int, source: Path, staged: Path, bundle: Mapping[str, Any],
    imported_row: Mapping[str, Any], fallback: Mapping[str, Any],
    explicit: Mapping[str, Any], threshold_override: float | None,
    notes: list[str],
) -> _SeedSpec:
    """Resolve one seed without letting a fallback YAML replace trained settings."""
    # Saved configuration and run identity.
    state, checkpoint = _load_checkpoint(staged)
    cp_config = _flatten(checkpoint.get("config") or checkpoint.get("parameters") or {})
    for key in ("model_name", "model_parameters", "seed"):
        if key not in checkpoint:
            continue
        if key in cp_config and not _same(cp_config[key], checkpoint[key]):
            raise ValueError(f"Checkpoint top-level {key} disagrees with checkpoint config.")
        cp_config[key] = checkpoint[key]
    sidecar_path = source.parent / "summary.json"
    sidecar = _read_mapping(None, source.parent, default=sidecar_path)
    summary = deepcopy(dict(imported_row or sidecar))
    saved_seed = bundle.get("seeds", {}).get(str(seed), {})
    # Per-run sources can assert the seed. Bundle metadata describes all seeds.
    run_sources = [
        ("checkpoint", cp_config),
        ("summary.json", _flatten(sidecar.get("parameters", {}))),
        ("training_results", _flatten(imported_row.get("parameters", {}))),
    ]
    saved_sources = [*run_sources, ("metadata.json", _flatten(bundle))]
    sources = [*saved_sources, ("explicit arguments", explicit)]
    for label, params in run_sources:
        if "seed" in params and _seed(params["seed"]) != seed:
            raise ValueError(f"Seed {seed} was supplied, but {label} says seed={params['seed']}.")

    # Architecture, input resolution and loss definition.
    name = str(_resolve("model_name", sources, fallback)).strip().lower()
    saved_architectures = [(label, params["model_parameters"]) for label, params in saved_sources
                           if "model_parameters" in params]
    if saved_architectures:
        first_label, first = saved_architectures[0]
        parameters = _architecture(name, first)
        for label, candidate in saved_architectures[1:]:
            if not _same(parameters, _architecture(name, candidate)):
                raise ValueError(f"Architecture conflict between {first_label} and {label}.")
        for key, value in explicit.get("model_parameters", {}).items():
            if key not in parameters or not _same(parameters[key], value):
                raise ValueError(f"Explicit model_parameters[{key!r}] conflicts with trained model.")
    else:
        parameters = _architecture(name, _resolve("model_parameters", sources, fallback))
    if fallback.get("model_parameters"):
        try:
            fallback_architecture = _architecture(name, fallback["model_parameters"])
        except (TypeError, ValueError):
            fallback_architecture = None
        if not _same(parameters, fallback_architecture):
            notes.append(f"Seed {seed}: YAML architecture ignored; trained checkpoint/metadata used.")
    image_size = _resolve("image_size", sources, fallback, full_alias=True)
    if image_size is not None and (
        isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0
    ):
        raise ValueError("image_size must be a positive integer, or None for native scenes.")
    seen_sizes = [params["image_size"] for _, params in sources if "image_size" in params]
    full_alias_used = None in seen_sizes and 1024 in seen_sizes
    # ChainMap keeps the first saved value, including None and zero.
    priority = ChainMap(*(params for _, params in sources), fallback)
    training = {key: deepcopy(priority[key]) for key in _TRAIN_KEYS if key in priority}
    for key, default in (("bce_weight", _MISSING), ("dice_weight", _MISSING),
                         ("positive_weight", None), ("smooth", 1.0)):
        training[key] = _resolve(key, sources, fallback, default=default)
    training.update(seed=seed, image_size=image_size, model_parameters=parameters)
    for key in ("precision", "world_size"):
        if key in checkpoint:
            training[key] = checkpoint[key]
    for key in ("bce_weight", "dice_weight", "smooth", "positive_weight"):
        value = training[key]
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value))):
            raise ValueError(f"Invalid loss parameter {key}: {value}")
    BCEDiceLoss(**_loss_kwargs(training))  # Validate weights before touching validation/test.

    # Epoch and threshold identify the selected checkpoint; metrics may be rounded.
    best = _best_metadata(checkpoint)
    # Translate the three saved formats to the same four field names.
    metric_fields = ("epoch", "threshold", "f1", "iou")
    summary_fields = ("best_epoch", "best_threshold", "best_f1", "best_iou")
    metric_sources = [
        ("checkpoint", best, ("epoch", "best_threshold", "validation_f1", "validation_iou")),
        ("metadata.json", saved_seed,
         ("best_epoch", "validation_threshold", "validation_f1", "validation_iou")),
        ("training_results", imported_row, summary_fields),
        ("summary.json", sidecar, summary_fields),
    ]
    available = []
    for label, row, fields in metric_sources:
        values = {key: row[field] for key, field in zip(metric_fields, fields, strict=True)
                  if row.get(field) is not None}
        available.append((label, values))
    # Epoch/threshold identify the run; saved F1/IoU may have rounding differences.
    epoch = _resolve("epoch", available, {}, default=None)
    original_threshold = _resolve("threshold", available, {}, default=None)
    saved_metrics = ChainMap(*(row for _, row in available))
    originals = {metric: saved_metrics.get(metric) for metric in ("f1", "iou")}
    if (threshold_override is not None and original_threshold is not None
            and not _same(threshold_override, original_threshold)):
        raise ValueError(
            f"Seed {seed}: supplied threshold {threshold_override} conflicts with saved "
            f"validation threshold {original_threshold}. This finalizer does not retune "
            "already selected thresholds."
        )
    threshold = threshold_override if threshold_override is not None else original_threshold
    if threshold is None:
        raise ValueError(
            f"Seed {seed} has no saved validation threshold. Pass thresholds={{seed: value}} "
            "chosen on validation (or a predeclared fixed threshold). YAML's provisional "
            "training.threshold is not silently treated as the final threshold."
        )
    if isinstance(threshold, bool) or not math.isfinite(float(threshold)) or not 0 < threshold < 1:
        raise ValueError(f"Invalid threshold for seed {seed}: {threshold!r}")
    if epoch is not None and (isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0):
        raise ValueError(f"Invalid checkpoint epoch: {epoch}")
    for key, value in originals.items():
        if value is not None and (not math.isfinite(float(value)) or not 0 <= value <= 1):
            raise ValueError(f"Invalid saved validation {key} for seed {seed}: {value}")

    # Verify the actual weights, then preserve the available training summary.
    try:
        model = _instantiate(name, parameters, state)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Cannot strictly reconstruct seed {seed} from {source}: {error}") from error
    counts = count_model_parameters(model)
    del model, state
    trial = checkpoint.get("run_name", saved_seed.get("trial", summary.get("trial")))
    if not summary:
        summary = {
            "trial": trial, "status": "checkpoint_only", **counts,
            "parameters": deepcopy(training), "best_epoch": epoch,
            "best_threshold": original_threshold, "best_f1": originals["f1"],
            "best_iou": originals["iou"], "best_validation_loss": best.get("validation_loss"),
            "best_precision": best.get("validation_precision"),
            "best_recall": best.get("validation_recall"),
            "training_seconds": None, "seconds_per_epoch": None,
            "gpu_hours": None, "peak_gpu_memory_mb": None,
            "metadata_note": "No full training summary supplied; training cost is unknown.",
        }
    summary.setdefault("parameters", deepcopy(training))
    summary["source_checkpoint"] = str(source)
    if sidecar_path.is_file():
        summary.setdefault("source_summary", str(sidecar_path))
    summary["seed"] = seed
    with staged.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return _SeedSpec(
        seed=seed,
        checkpoint=staged,
        source=source,
        model_name=name,
        model_parameters=parameters,
        training_parameters=training,
        image_size=image_size,
        threshold=float(threshold),
        threshold_source=("saved_validation" if threshold_override is None
                          else "user_supplied_before_test"),
        original_threshold=original_threshold,
        epoch=int(epoch) if epoch is not None else None,
        original_f1=originals["f1"],
        original_iou=originals["iou"],
        trial=trial,
        training_row=summary,
        sha256=digest,
        full_alias_used=full_alias_used,
    )


def _loss_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {"bce_weight": parameters["bce_weight"], "dice_weight": parameters["dice_weight"],
            "pos_weight": parameters["positive_weight"], "smooth": parameters["smooth"]}


# Evaluation runtime and inference measurement


def _device(value: str | torch.device) -> torch.device:
    if str(value) == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This benchmark supports CPU and CUDA devices only.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device is unavailable: {device}")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device is unavailable: {device}")
        device = torch.device("cuda", index)
    return device


@contextmanager
def _runtime(device: torch.device) -> Iterator[None]:
    """Temporarily disable TF32/autocast; restore caller backend flags and Torch RNG."""
    settings: list[tuple[Any, str, Any]] = []

    def set_temporarily(obj: Any, name: str, value: Any) -> None:
        settings.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    cuda_scope = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    devices = [device.index] if device.type == "cuda" else []
    with cuda_scope, torch.random.fork_rng(devices=devices):
        try:
            # Do not mix the new fp32_precision controls with legacy allow_tf32 controls.
            if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                for obj in (torch.backends, torch.backends.cuda.matmul,
                            torch.backends.cudnn, torch.backends.cudnn.conv):
                    set_temporarily(obj, "fp32_precision", "ieee")
            else:
                set_temporarily(torch.backends.cuda.matmul, "allow_tf32", False)
                set_temporarily(torch.backends.cudnn, "allow_tf32", False)
            set_temporarily(torch.backends.cudnn, "benchmark", False)
            set_temporarily(torch.backends.cudnn, "deterministic", True)
            with torch.autocast(device_type=device.type, enabled=False):
                yield
        finally:
            for obj, name, value in reversed(settings):
                setattr(obj, name, value)


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _evaluate(
    spec: _SeedSpec, loader: Any, split: str, device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model = spec.load_model()
    loss = BCEDiceLoss(**_loss_kwargs(spec.training_parameters))
    try:
        model.to(device)
        loss.to(device)
        with torch.inference_mode():
            result = evaluate_thresholds(
                model=model, data_loader=loader, loss_function=loss, device=device,
                thresholds=[spec.threshold], precision=precision,
            )
        metrics = result["threshold_results"][0]
        if not all(math.isfinite(float(result["loss"] if key == "loss" else metrics[key]))
                   for key in _METRICS):
            raise RuntimeError(f"Non-finite {split} metrics for seed {spec.seed}.")
        row = {
            "trial": spec.trial, "seed": spec.seed, "best_validation_epoch": spec.epoch,
            "validation_threshold": spec.threshold,
            "original_validation_f1": spec.original_f1,
            "original_validation_iou": spec.original_iou,
            f"{split}_loss": float(result["loss"]),
            **{f"{split}_{key}": float(metrics[key]) for key in _METRICS if key != "loss"},
            **{f"{split}_{key}": int(metrics[key]) for key in ("tp", "fp", "fn", "tn")},
            "threshold_source": spec.threshold_source,
        }
        if split == "test":
            row.update(validation_f1=spec.original_f1, validation_iou=spec.original_iou)
        print(f"{split:10s} seed={spec.seed:5d} | threshold={spec.threshold:.4g} | "
              f"F1={metrics['f1']:.6f} | IoU={metrics['iou']:.6f}", flush=True)
        return row
    finally:
        del model, loss
        _release(device)


def _hardware_benchmark(
    spec: _SeedSpec, sample: Mapping[str, Any], stage: Path, device: torch.device,
    precision: str, warmup: int, repetitions: int,
) -> dict[str, Any]:
    model = spec.load_model()
    # Portable, weights-only FP32 state_dict. Original per-seed checkpoints remain byte-exact.
    weights_path = stage / "deployment_weights.pt"
    torch.save(model.state_dict(), weights_path)
    counts = count_model_parameters(model)
    _release(device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    before = after = output = None
    try:
        model.to(device)
        before = sample["before"].unsqueeze(0).to(device=device, dtype=torch.float32)
        after = sample["after"].unsqueeze(0).to(device=device, dtype=torch.float32)
        times = []
        with torch.inference_mode(), precision_context(device, precision):
            for _ in range(warmup):
                output = model(before, after)
                output = None
            for _ in range(repetitions):
                if cuda:
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = model(before, after)
                if cuda:
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - started) * 1000.0)
                output = None
        peak = torch.cuda.max_memory_allocated(device) if cuda else None
        mean_ms = statistics.mean(times)
        return {
            "parameters": counts["total_parameters"],
            "trainable_parameters": counts["trainable_parameters"],
            "checkpoint_mb": spec.checkpoint.stat().st_size / 1024 ** 2,
            "weights_only_mb": weights_path.stat().st_size / 1024 ** 2,
            "peak_inference_vram_mb": (peak - baseline) / 1024 ** 2 if cuda else None,
            "notebook_baseline_vram_mb": baseline / 1024 ** 2 if cuda else None,
            "process_peak_allocated_vram_mb": peak / 1024 ** 2 if cuda else None,
            "process_peak_reserved_vram_mb": (
                torch.cuda.max_memory_reserved(device) / 1024 ** 2 if cuda else None
            ),
            "latency_ms_per_pair": mean_ms, "pairs_per_second": 1000.0 / mean_ms,
            "latency_median_ms": statistics.median(times), "latency_samples_ms": times,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
            "pytorch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "deployment_seed": spec.seed, "deployment_threshold": spec.threshold,
            "protocol": {
                "precision": precision, "tf32": False, "batch_size": 1,
                "input_shape_per_image": list(before.shape), "images_per_pair": 2,
                "warmup": warmup, "repetitions": repetitions,
                "timing": "synchronized forward only" if cuda else "CPU forward only",
                "input": "first validation pair, already on device; reused for all repetitions",
                "includes_loading_preprocessing_h2d_sigmoid_threshold": False,
                "memory_unit": "MiB; existing process tensor allocations subtracted",
                "memory_scope": "model + input pair + forward peak, including warmup; not total GPU usage",
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_version": torch.backends.cudnn.version(),
                "cpu_threads": torch.get_num_threads(),
                "cpu_interop_threads": torch.get_num_interop_threads(),
                "light_protocol_identical": "not assumed; compare recorded protocols",
            },
        }
    finally:
        del output, before, after, model
        _release(device)


# Publishing: preserve old results until all new artifacts are ready.


@contextmanager
def _output_transaction(output: Path, overwrite: bool) -> Iterator[tuple[Path, Path | None]]:
    """Stage all results. On ordinary failures the previous final folder is untouched."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.benchmark.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Another run/stale lock exists: {lock}. Check the running process.") from error
    stage = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\nstarted={datetime.now(UTC).isoformat()}\n")
        if output.exists() and (not output.is_dir() or not overwrite):
            raise FileExistsError(f"Output exists: {output}. Use overwrite=True to back it up and replace it.")
        backup = None
        if output.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            backup = output.parent / ".benchmark_backups" / f"{output.name}_{stamp}_{uuid4().hex[:8]}"
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.benchmark_work_", dir=output.parent))
        print(f"Working directory: {stage}", flush=True)
        yield stage, backup
        if backup is not None:
            backup.parent.mkdir(parents=True, exist_ok=True)
            output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                backup.rename(output)
            raise
    except BaseException as error:
        if stage is not None and stage.exists():
            _write_json(stage / "FAILED.json", {"error": str(error), "evaluation_complete": False})
            print(f"Benchmark failed; previous final files were not replaced. Diagnostics: {stage}",
                  flush=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


# Preparation: copy checkpoints, then verify their metadata and input geometry.


def _prepare_specs(
    paths: Mapping[int, Path], stage: Path, bundle: Mapping[str, Any],
    training_document: Mapping[str, Any], fallback: Mapping[str, Any],
    explicit: Mapping[str, Any], thresholds: Mapping[int, float | None],
    notes: list[str],
) -> list[_SeedSpec]:
    imported = {}
    for row in training_document.get("results", []):
        if not isinstance(row, Mapping):
            raise ValueError("training_results.results must contain mappings.")
        number = row.get("seed", row.get("parameters", {}).get("seed"))
        if number is None:
            continue
        number = _seed(number)
        if number in imported:
            raise ValueError(f"Multiple training rows for seed {number}; select final runs first.")
        imported[number] = row

    specs = []
    for seed, source in paths.items():
        staged = stage / f"seed_{seed}_best.pt"
        shutil.copy2(source, staged)
        specs.append(_resolve_spec(
            seed, source, staged, bundle, imported.get(seed, {}), fallback, explicit,
            thresholds.get(seed), notes,
        ))

    if len({spec.sha256 for spec in specs}) != len(specs):
        raise ValueError("Identical checkpoint bytes assigned to different seeds; not independent runs.")
    first = specs[0]
    for spec in specs[1:]:
        same_architecture = (spec.model_name == first.model_name
                             and _same(spec.model_parameters, first.model_parameters))
        if not same_architecture:
            raise ValueError("All seeds in one final folder must have the same architecture.")
        native_alias = spec.image_size in (None, 1024) and first.image_size in (None, 1024)
        if spec.image_size != first.image_size and not native_alias:
            raise ValueError("All seeds must have the same scene resolution.")
        if not _same(_loss_kwargs(spec.training_parameters), _loss_kwargs(first.training_parameters)):
            raise ValueError("Loss definitions differ across seeds; use separate benchmark folders.")
    return specs


def _prepare_data(
    root: Path, specs: list[_SeedSpec], batch_size: int, num_workers: int,
    device: torch.device,
) -> dict[str, tuple[Any, Any]]:
    """Build both loaders and check image headers before evaluating either split."""
    from PIL import Image

    image_size = specs[0].image_size
    datasets = {
        name: create_dataset_dataloader(
            dataset_root=root, split=split, image_size=image_size,
            batch_size=batch_size, num_workers=num_workers,
            transform=None, shuffle=False, pin_memory=device.type == "cuda", drop_last=False,
        )
        for name, split in (("validation", "val"), ("test", "test"))
    }
    original_shapes = set()
    for dataset, _ in datasets.values():
        for sample_id in dataset.sample_ids:
            shapes = set()
            for files in (dataset.before_files, dataset.after_files, dataset.label_files):
                with Image.open(files[sample_id]) as image:
                    shapes.add(image.size)
            if len(shapes) != 1:
                raise ValueError(f"Unaligned native image/mask dimensions: {sample_id}")
            original_shapes.update(shapes)

    sizes = {spec.image_size for spec in specs}
    alias_used = len(sizes) > 1 or any(spec.full_alias_used for spec in specs)
    if alias_used and original_shapes != {(1024, 1024)}:
        raise ValueError("None and 1024 are equivalent only for native 1024x1024 scenes.")
    processed_shapes = original_shapes if image_size is None else {(image_size, image_size)}
    if len(processed_shapes) != 1:
        raise ValueError("Native input shapes differ; this bundle requires one fixed scene shape.")
    return datasets


def _evaluate_seeds(
    specs: list[_SeedSpec], datasets: Mapping[str, tuple[Any, Any]], stage: Path,
    *, variant: str, device: torch.device, precision: str, batch_size: int,
    deployment_seed: int | None, bundle: Mapping[str, Any],
    strict_validation: bool, validation_atol: float, notes: list[str],
) -> tuple[dict[str, Any], _SeedSpec, str]:
    """Run validation, freeze deployment selection, then run test at frozen thresholds."""
    reports = {}
    validation_by_seed = {}
    first = specs[0]
    selected = None
    selection = ""

    for split, filename in (("validation", "validation_results"), ("test", "final_test_results")):
        dataset, loader = datasets[split]
        rows = []
        for spec in specs:
            row = _evaluate(spec, loader, split, device, precision)
            for metric in ("f1", "iou"):
                if split == "test":
                    row[f"recomputed_validation_{metric}"] = (
                        validation_by_seed[spec.seed][f"validation_{metric}"]
                    )
                    continue

                original = getattr(spec, f"original_{metric}")
                comparable = original is not None and spec.threshold == spec.original_threshold
                delta = row[f"validation_{metric}"] - original if comparable else None
                row[f"validation_{metric}_delta"] = delta
                if delta is None or abs(delta) <= validation_atol:
                    continue
                message = (
                    f"Seed {spec.seed}: validation {metric} differs by {delta:+.6f} "
                    f"(tolerance={validation_atol}). Check code, resolution, data and precision."
                )
                if strict_validation:
                    raise ValueError(message + " No test evaluation was started.")
                notes.append(message)
            rows.append(row)

        if split == "validation":
            validation_by_seed = {row["seed"]: row for row in rows}
            by_seed = {spec.seed: spec for spec in specs}
            saved_seed = bundle.get("deployment_seed")
            if deployment_seed is not None:
                choice = _seed(deployment_seed)
                if choice not in by_seed:
                    raise ValueError(
                        f"deployment_seed={choice} is not among input seeds {sorted(by_seed)}."
                    )
                selection = "explicitly supplied before test evaluation"
            elif saved_seed is not None and _seed(saved_seed) in by_seed:
                choice = _seed(saved_seed)
                selection = "preserved from input metadata; not selected by test F1"
            else:
                use_saved = all(spec.original_f1 is not None
                                and spec.threshold == spec.original_threshold for spec in specs)
                scores = {spec.seed: spec.original_f1 if use_saved
                          else validation_by_seed[spec.seed]["validation_f1"] for spec in specs}
                choice = max(scores, key=lambda seed: (scores[seed], -seed))
                source = "saved" if use_saved else "recomputed"
                selection = f"highest {source} validation F1; not test F1; ties: smallest seed"
            selected = by_seed[choice]
            print(f"Deployment selected BEFORE test: seed={choice}, "
                  f"threshold={selected.threshold} ({selection})", flush=True)

        report = {
            "model": first.model_name,
            "variant": variant,
            "image_size": first.image_size,
            "seeds": [row["seed"] for row in rows],
            "std_type": "sample",
            "std_ddof": 1,
            "number_of_pairs": len(dataset),
            "evaluation_batch_size": batch_size,
            "results": rows,
            "precision": precision,
            "metric_aggregation": "global pixel confusion counts within each seed; mean across seeds",
            "threshold_selection": "saved or explicitly supplied validation threshold; no test tuning",
        }
        for metric in _METRICS:
            values = [float(row[f"{split}_{metric}"]) for row in rows]
            report[f"{split}_{metric}_mean"] = statistics.mean(values)
            # A single seed has no sample standard deviation: preserve JSON null, not zero.
            report[f"{split}_{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        reports[filename] = report
        _write_json(stage / f"{filename}.json", report)

    # Validation always runs first, so selected cannot be None here.
    assert selected is not None
    return reports, selected, selection


def _build_metadata(
    specs: list[_SeedSpec], validation: Mapping[str, Any], selected: _SeedSpec,
    selection: str, bundle: Mapping[str, Any], notes: list[str],
    *, variant: str, dataset: Path, shape: list[int], precision: str,
    batch_size: int, num_workers: int, strict_validation: bool,
    validation_atol: float, backup: Path | None,
) -> dict[str, Any]:
    """Keep the legacy JSON fields and add provenance without changing metric semantics."""
    first = specs[0]
    common_training = {
        key: deepcopy(value)
        for key, value in first.training_parameters.items()
        if key not in ("seed", "model_parameters")
        and all(key in spec.training_parameters and _same(value, spec.training_parameters[key])
                for spec in specs)
    }
    seeds = {}
    for spec, row in zip(specs, validation["results"], strict=True):
        seeds[str(spec.seed)] = {
            "trial": spec.trial,
            "best_epoch": spec.epoch,
            "validation_threshold": spec.threshold,
            "source_checkpoint": str(spec.source),
            "checkpoint_file": spec.checkpoint.name,
            "checkpoint_sha256": spec.sha256,
            "training_parameters": spec.training_parameters,
            "threshold_source": spec.threshold_source,
        }
        for metric in ("f1", "iou"):
            original = getattr(spec, f"original_{metric}")
            use_saved = original is not None and spec.threshold == spec.original_threshold
            seeds[str(spec.seed)][f"validation_{metric}"] = (
                original if use_saved else row[f"validation_{metric}"]
            )
            seeds[str(spec.seed)][f"recomputed_validation_{metric}"] = row[f"validation_{metric}"]

    for key in ("learning_rate", "weight_decay", "scheduler", "batch_size", "epochs", "precision"):
        values = [spec.training_parameters.get(key, "unknown") for spec in specs]
        if any(not _same(values[0], value) for value in values[1:]):
            notes.append(f"Training {key} differs across seeds: {values}; see per-seed metadata.")
    source_metadata = deepcopy(dict(bundle))
    source_metadata.pop("source_metadata", None)  # Do not nest previous bundles on every refresh.
    return {
        "model_level": {"fc_siam_diff": "light", "bit": "medium",
                        "siam_mit_b3_diff": "heavy"}.get(first.model_name, "custom"),
        "variant": "full_resolution" if first.image_size in (None, 1024) else "resized",
        "model_name": first.model_name,
        "image_size": first.image_size,
        "model_parameters": first.model_parameters,
        "training_parameters": common_training,
        "deployment_seed": selected.seed,
        "deployment_threshold": selected.threshold,
        "deployment_best_epoch": selected.epoch,
        "deployment_selection": selection,
        "evaluation_batch_size": batch_size,
        "num_workers": num_workers,
        "std_ddof": 1,
        "evaluation_function": "satchange.training.evaluate_thresholds",
        "seeds": seeds,
        "evaluation_complete": True,
        "schema_version": 2,
        "bundle_variant": variant,
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset),
        "evaluation_precision": precision,
        "input_shape_per_image": [1, *shape],
        "preprocessing": {
            "spatial_policy": "whole scene; no crops or tiles",
            "rgb_normalization": "float32 / 255.0",
            "rgb_resize": "none" if first.image_size is None else "PIL BILINEAR",
            "mask_resize": "none" if first.image_size is None else "PIL NEAREST",
            "mask_binarization": "value > 0",
            "prediction_rule": "sigmoid(logit) >= threshold",
            "metric_resolution": "processed mask resolution; no upsampling back to native size",
        },
        "validation_check": {"strict": strict_validation, "absolute_tolerance": validation_atol},
        "deployment_weights_format": "plain CPU FP32 state_dict; no optimizer",
        "seed_checkpoints_format": "byte-exact copies of input checkpoints",
        "backup_directory": str(backup) if backup else None,
        "source_metadata": source_metadata,
        "notes": sorted(set(notes)),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON file containing run_final_benchmark arguments")
    arguments = parser.parse_args()
    spec = _read_mapping(arguments.spec.resolve(), Path.cwd())
    run_final_benchmark(**spec)


if __name__ == "__main__":
    _main()
