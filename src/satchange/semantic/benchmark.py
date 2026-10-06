"""Semantic evaluation/export adapter for the shared run_final_benchmark entry point."""

import math
import shutil
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from satchange.execution import precision_context
from satchange.hpo_backend import SatelliteBackend
from satchange.hpo_core import family
from satchange.models import count_model_parameters
from satchange.models.factory import SEMANTIC_TASK
from satchange.semantic.data import SemanticROIDataset, file_digest
from satchange.semantic.inference import evaluate_classifier, load_classifier
from satchange.training import _save_checkpoint

_METRICS = (
    "loss",
    "selection_score",
    "before_macro_f1",
    "after_macro_f1",
    "before_accuracy",
    "after_accuracy",
    "joint_accuracy",
)


def _speed(
    checkpoint: Path,
    sample: dict[str, Any],
    weights: Path,
    device: torch.device,
    precision: str,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    """Time one native ROI pair, excluding proposal extraction, I/O and device transfer."""
    from satchange.final_benchmark import _release

    _release(device)
    cuda = device.type == "cuda"
    baseline = torch.cuda.memory_allocated(device) if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model, state = load_classifier(checkpoint, device)
    torch.backends.cuda.matmul.allow_tf32 = state["config"]["tf32"]
    torch.backends.cudnn.allow_tf32 = state["config"]["tf32"]
    values = [sample[key].unsqueeze(0).to(device) for key in ("before", "after", "mask")]
    counts = count_model_parameters(model)
    times = []
    try:
        with torch.inference_mode(), precision_context(device, precision):
            for _ in range(warmup):
                model(*values)
            for _ in range(repetitions):
                if cuda:
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(*values)
                if cuda:
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 1000)
        mean = statistics.mean(times)
        return {
            **counts,
            "deployment_seed": state["config"]["seed"],
            "checkpoint_mb": checkpoint.stat().st_size / 1024**2,
            "weights_only_mb": weights.stat().st_size / 1024**2,
            "latency_ms_per_pair": mean,
            "pairs_per_second": 1000 / mean,
            "latency_median_ms": statistics.median(times),
            "latency_samples_ms": times,
            "peak_inference_vram_mb": (
                (torch.cuda.max_memory_allocated(device) - baseline) / 1024**2 if cuda else None
            ),
            "device": str(device),
            "pytorch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
            "protocol": {
                "task": SEMANTIC_TASK,
                "precision": precision,
                "tf32": state["config"]["tf32"],
                "batch_size": 1,
                "input_shape_per_image": list(values[0].shape),
                "mask_shape": list(values[2].shape),
                "warmup": warmup,
                "repetitions": repetitions,
                "input": "first eligible native validation ROI",
                "timing": "synchronized classifier forward only" if cuda else "CPU forward only",
                "includes_detector_roi_extraction_loading_h2d": False,
                "scope": "classifier only; variable ROI sizes affect latency",
                "memory_unit": "MiB; existing process allocations subtracted",
            },
        }
    finally:
        del model, state, values
        _release(device)


def finalize_semantic(
    *,
    paths: dict[int, Path],
    prepared: Path,
    output: Path,
    variant: str,
    root: Path,
    training_results: Any,
    metadata: Any,
    device: torch.device,
    precision: str,
    evaluation_batch_size: int,
    num_workers: int,
    deployment_seed: int | None,
    warmup: int,
    repetitions: int,
    strict_validation: bool,
    validation_atol: float,
    overwrite: bool,
) -> dict[str, Any]:
    """Validate all seeds, freeze deployment, evaluate untouched test, then publish a bundle."""
    from satchange.final_benchmark import (
        _output_transaction,
        _read_mapping,
        _runtime,
        _seed,
        _write_json,
    )

    prepared = prepared / "prepared.json" if prepared.is_dir() else prepared
    if not prepared.is_file():
        raise FileNotFoundError(prepared)
    training = _read_mapping(training_results, root)
    bundle = _read_mapping(metadata, root)
    if training and (not training.get("complete") or set(training.get("seeds", [])) != set(paths)):
        raise ValueError("Final training summary must be complete and cover every input seed.")
    rows = {int(row["parameters"]["seed"]): row for row in training.get("results", [])}
    if rows and set(rows) != set(paths):
        raise ValueError("Final training rows differ from the supplied checkpoint seeds.")
    reports: dict[str, Any] = {}
    notes = []
    with _output_transaction(output, overwrite) as (stage, backup), _runtime(device):
        states = {}
        copies = {}
        hashes = set()
        definition = None
        for seed, source in paths.items():
            copy = stage / f"seed_{seed}_best.pt"
            shutil.copy2(source, copy)
            model, state = load_classifier(copy)
            del model
            if state["config"]["seed"] != seed:
                raise ValueError("Checkpoint seed differs from its mapping key.")
            normalized = family(SatelliteBackend().normalize(state["config"]))
            if definition is not None and definition != normalized:
                raise ValueError(
                    "Semantic seeds must have the same data/model/training definition."
                )
            definition = normalized
            digest = file_digest(copy)
            if digest in hashes:
                raise ValueError("Identical checkpoint bytes assigned to different seeds.")
            hashes.add(digest)
            if seed in rows:
                row = rows[seed]
                if (
                    row["epoch"] != training["target_epoch"]
                    or state.get("training_completed_epoch") != row["epoch"]
                    or state.get("training_budget_epochs") != training["target_epoch"]
                    or Path(row["best_checkpoint"]).resolve() != source.resolve()
                ):
                    raise ValueError(
                        "Checkpoint completion/path disagrees with final training summary."
                    )
                if family(SatelliteBackend().normalize(row["parameters"])) != normalized:
                    raise ValueError("Final training parameters disagree with the checkpoint.")
            states[seed], copies[seed] = state, copy
        first = states[next(iter(states))]
        dataset = SemanticROIDataset(prepared, "val", first["config"]["semantic_config"])
        if not dataset.records:
            raise ValueError("Validation contains no eligible semantic ROIs.")
        selected = None
        selection = ""
        for split, filename in (("val", "validation_results"), ("test", "final_test_results")):
            values = []
            for seed, copy in copies.items():
                result = evaluate_classifier(
                    copy,
                    prepared,
                    split,
                    device,
                    precision=precision,
                    evaluation_batch_size=evaluation_batch_size,
                    num_workers=num_workers,
                )
                if split == "val":
                    saved = states[seed]["best_metrics"]
                    for metric in _METRICS:
                        expected = saved["validation_loss" if metric == "loss" else metric]
                        delta = float(result[metric]) - float(expected)
                        result[f"{metric}_delta"] = delta
                        if not math.isfinite(delta) or abs(delta) > validation_atol:
                            message = f"Seed {seed}: validation {metric} differs by {delta:+.6f}."
                            if strict_validation:
                                raise ValueError(message + " No test evaluation was started.")
                            notes.append(message)
                values.append(result)
            label = "validation" if split == "val" else "test"
            report = {
                "task": SEMANTIC_TASK,
                "variant": variant,
                "seeds": list(copies),
                "results": values,
                "std_ddof": 1,
                "std_type": "sample",
                "metric_aggregation": "global per-date ROI confusion counts; mean across seeds",
                "evaluation_batch_size": evaluation_batch_size,
                "num_workers": num_workers,
                "precision": precision,
                "number_of_rois": values[0]["samples"],
                "scope": "eligible ROI semantic classification; excludes detector accuracy",
            }
            for metric in _METRICS:
                scores = [float(value[metric]) for value in values]
                report[f"{label}_{metric}_mean"] = statistics.mean(scores)
                report[f"{label}_{metric}_std"] = (
                    statistics.stdev(scores) if len(scores) > 1 else None
                )
            reports[filename] = report
            _write_json(stage / f"{filename}.json", report)
            if split == "val":
                choice = (
                    deployment_seed
                    if deployment_seed is not None
                    else bundle.get("deployment_seed")
                )
                if choice is None:
                    choice = max(values, key=lambda row: (row["selection_score"], -row["seed"]))[
                        "seed"
                    ]
                    selection = "highest validation mean per-date macro F1; ties: smallest seed"
                else:
                    choice = _seed(choice)
                    selection = "explicit/saved deployment seed fixed before test"
                if choice not in copies:
                    raise ValueError("Deployment seed is absent from input checkpoints.")
                selected = choice
                _write_json(
                    stage / "deployment_choice.json",
                    {"seed": choice, "selection": selection, "test_used": False},
                )
        assert selected is not None
        weights = stage / "deployment_weights.pt"
        _save_checkpoint(
            weights,
            {
                key: tensor.detach().cpu().float()
                if tensor.is_floating_point()
                else tensor.detach().cpu()
                for key, tensor in states[selected]["model_state_dict"].items()
            },
        )
        reports["benchmark"] = _speed(
            copies[selected], dataset[0], weights, device, precision, warmup, repetitions
        )
        reports["training_results"] = (
            deepcopy(training)
            if training
            else {
                "results": [
                    {"seed": seed, "best_epoch": state["epoch"], "parameters": state["config"]}
                    for seed, state in states.items()
                ]
            }
        )
        reports["metadata"] = {
            "schema_version": 1,
            "task": SEMANTIC_TASK,
            "model_name": first["model_name"],
            "model_parameters": first["config"]["model_parameters"],
            "training_parameters": definition,
            "deployment_seed": selected,
            "deployment_selection": selection,
            "deployment_best_epoch": states[selected]["epoch"],
            "dataset_root": str(prepared),
            "prepared_identity": file_digest(prepared),
            "protocol": dataset.document["protocol"],
            "evaluation_complete": True,
            "evaluation_precision": precision,
            "std_ddof": 1,
            "validation_check": {
                "strict": strict_validation,
                "absolute_tolerance": validation_atol,
            },
            "seeds": {
                str(seed): {
                    "checkpoint_file": copies[seed].name,
                    "checkpoint_sha256": file_digest(copies[seed]),
                    "source_checkpoint": str(paths[seed]),
                    "best_epoch": state["epoch"],
                }
                for seed, state in states.items()
            },
            "deployment_weights_format": "plain CPU FP32 state_dict; no optimizer",
            "scope": "semantic accuracy on eligible ROIs; detector performance is not evaluated",
            "notes": notes,
            "source_metadata": bundle,
        }
        for name in ("benchmark", "training_results", "metadata"):
            _write_json(stage / f"{name}.json", reports[name])
    return {"output_dir": str(output), "backup_dir": str(backup) if backup else None, **reports}
