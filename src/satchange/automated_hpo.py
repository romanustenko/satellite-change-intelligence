"""Public three-phase HPO API: search -> seed validation -> final training.

Training/data/model implementation is injected through a backend import path.
Use stage_epochs=[40, 60, 80] for ABSOLUTE targets. Calling a cell twice does
not add epochs twice. run_final_benchmark remains a separate, read-only stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .hpo_core import (
    CodeCompatibilityError,
    Experiment,
    ParameterSearchMemoryError,
    PipelineRunError,
    PipelineSettings,
    acknowledge_code_change,
    experiment_lock,
    load_backend,
)

__all__ = [
    "run_automated_parameter_search",
    "request_hpo_pause",
    "acknowledge_code_change",
    "CodeCompatibilityError",
    "ParameterSearchMemoryError",
    "PipelineRunError",
]
PHASES = ("hpo", "seeds", "final")


def _budgets(values: Sequence[int] | Mapping[str, int]) -> dict[str, int]:
    if isinstance(values, Mapping):
        if set(values) != set(PHASES):
            raise ValueError("stage_epochs mapping needs exactly hpo, seeds and final.")
        result = dict(values)
    else:
        if len(values) != 3:
            raise ValueError("stage_epochs must contain [hpo_epochs, seed_epochs, final_epochs].")
        result = dict(zip(PHASES, values, strict=False))
    epochs = [result[name] for name in PHASES]
    if any(type(n) is not int or n < 1 for n in epochs) or epochs != sorted(epochs):
        raise ValueError("stage_epochs must be positive, nondecreasing absolute targets.")
    return result


def _paths(value: str | Path | Sequence[str | Path] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        value = [value]
    return list(dict.fromkeys(str(Path(item).expanduser().resolve()) for item in value))


def request_hpo_pause(search_directory: str | Path, *, after: str = "epoch") -> Path:
    """Call from another kernel/terminal. Works ONLY with the new v3 runner.

    after='epoch': save each active trial's epoch and pause it.
    after='trial': finish active trials, but dispatch no new ones.
    A subsequent explicit run_automated_parameter_search call clears the marker.
    """
    if after not in ("epoch", "trial"):
        raise ValueError("after must be 'epoch' or 'trial'.")
    root = Path(search_directory).expanduser().resolve()
    if not (root / "pipeline_protocol.json").exists():
        raise ValueError(
            "No v3 protocol here. The legacy process does not understand pause markers."
        )
    path = root / ("PAUSE_AFTER_EPOCH" if after == "epoch" else "STOP_AFTER_ACTIVE_TRIALS")
    path.touch()
    return path


def run_automated_parameter_search(
    *,
    stages: Sequence[tuple[str, dict[str, list[Any]]]],
    base_parameters: dict[str, Any],
    model_name: str,
    dataset_root: str | Path,
    thresholds: Sequence[float] = (),
    output_root: str | Path,
    search_name: str,
    stage: str = "hpo",
    hpo_stage: str | None = None,
    stage_epochs: Sequence[int] | Mapping[str, int] = (40, 60, 80),
    seeds: Sequence[int] = (42, 123, 777),
    additional_seeds: bool = True,
    gpu_ids: Sequence[int] | None = None,
    devices: Sequence[str] | None = None,
    reuse_from: str | Path | Sequence[str | Path] | None = None,
    resume_from: str | Path | Sequence[str | Path] | None = None,
    resume_incomplete: bool = True,
    pending_policy: str = "all",
    additional_epochs: int = 0,
    extension_id: str | None = None,
    final_parameters: dict[str, Any] | None = None,
    patience: int | None = None,
    min_epochs: int = 1,
    min_delta: float = 0.0,
    seed_patience: int | None = None,
    f1_gap: float = 0.04,
    iou_gap: float = 0.05,
    checkpoint_interval: int = 10,
    max_trial_memory_mb: float = 80_000.0,
    oom_retry_delay_seconds: float = 30.0,
    oom_retry_memory_margin_mb: float = 512.0,
    max_oom_attempts: int | None = 12,
    allow_legacy_import: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
    backend: str = "satchange.hpo_backend:SatelliteBackend",
    backend_context: Mapping[str, Any] | None = None,
    task: str = "binary_change_detection",
    gpu_hour_budget: float | None = None,
    budget_reason: str | None = None,
    unverified_reuse: str = "refuse",
) -> dict[str, Any]:
    """Execute ONLY the requested phase, safely resumable from disk.

    stage='hpo': one base seed; hpo_stage='03_loss' runs just that named block.
        Prior blocks must be complete (old completed runs may satisfy them).
        A legacy stage='03_loss' call is accepted as an alias.
    stage='seeds': all surviving configurations on the three supplied seeds.
        additional_seeds=False is an explicitly exploratory one-seed check.
    stage='final': ONE frozen winning configuration, ALWAYS three seeds;
        early stopping is disabled and every seed must reach the final target.

    stage_epochs are total targets, NOT [40 + 60 + 80]. A 40-epoch checkpoint
    goes to 60 with 20 additional epochs, then to 80 with another 20.
    additional_epochs=N adds N to the selected phase's target. extension_id
    makes that extension idempotent; repeat the same call to resume it.

    pending_policy='finish_started' freezes the subset that actually has
    checkpoints/results, not all entries in requests/. In the seed phase,
    families with ONLY the old HPO seed are excluded. Only already-started
    seed identities are continued; absent companion seeds are NOT invented.
    Families missing any required seed are reported but cannot win selection.
    stage='final' always fills all three mandatory seed runs.

    Legacy folders are READ-ONLY inputs via reuse_from. Use a NEW search_name.
    resume_from accepts explicit full checkpoints, including an explicitly
    chosen best.pt; automatic continuation always prefers last.pt.
    Existing parameters must match, including the seed. A new seed starts
    from its own initialization unless its own matching checkpoint exists.

    A policy change on promotion resets ONLY early-stopping state and is
    recorded. It does not reset optimizer, LR scheduler, RNG, or best history.
    Resume is at epoch boundaries, not from an interrupted minibatch.

    dry_run writes a frozen plan but runs no training. Results are incremental.
    CUDA OOM retries retain last.pt; exhausted retries/capacity failures raise
    AFTER other jobs in the phase finish. No later phase/benchmark runs then.

    Other projects can copy hpo_core.py and provide an importable backend class
    conforming to its Backend protocol; this wrapper supplies project defaults.

    Code compatibility (schema 4): the protocol records a per-file code manifest. Code that
    affects training or experiment control must match the accepted code, otherwise the call
    refuses; acknowledge_code_change records an evidence-backed exception. reuse_from and
    resume_from sources are verified the same way. unverified_reuse='exploratory' reuses
    COMPLETED results from unverified code, labelled exploratory_only; partial runs from
    unverified code are never continued.

    GPU-hours: every dispatch is an allocation in <search>/gpu_ledger.jsonl (dispatch to
    completion x devices). gpu_hour_budget (allocated GPU-hours for this search directory) is
    recorded in the ledger; None keeps a previously recorded budget. Trials that do not fit
    are not dispatched; reaching the budget pauses active trials at the next epoch boundary
    and any overshoot is recorded. Zero overshoot is not guaranteed.
    """
    import math

    if task != "binary_change_detection":
        raise ValueError(f"Unsupported task: {task}")
    if base_parameters.get("task", "binary_change_detection") != task:
        raise ValueError("The requested task differs from base_parameters.")

    epochs = _budgets(stage_epochs)
    names = [name for name, _ in stages]
    if stage in names:
        if hpo_stage is not None:
            raise ValueError("Pass either stage=<block> or stage='hpo', hpo_stage=<block>.")
        stage, hpo_stage = "hpo", stage
    if stage not in PHASES:
        raise ValueError("stage must be 'hpo', 'seeds', 'final', or a defined HPO block name.")
    if stage != "hpo" and hpo_stage is not None:
        raise ValueError("hpo_stage applies only to stage='hpo'.")
    if not search_name or Path(search_name).name != search_name or search_name in (".", ".."):
        raise ValueError("search_name must be a single directory name.")
    if len(names) != len(set(names)) or any(
        not name or Path(name).name != name or name in (*PHASES, ".", "..") for name in names
    ):
        raise ValueError("HPO block names must be unique directory names, not hpo/seeds/final.")
    if any("seed" in space for _, space in stages):
        raise ValueError("Use seeds= for seed validation; do not search seed inside an HPO grid.")
    if len(seeds) != 3 or any(type(n) is not int or n < 0 for n in seeds) or len(set(seeds)) != 3:
        raise ValueError("Provide exactly THREE distinct nonnegative final seeds.")
    if base_parameters.get("seed") not in seeds:
        raise ValueError("The base HPO seed must be one of the three final seeds.")
    if type(additional_seeds) is not bool:
        raise ValueError("additional_seeds must be True or False.")
    if not resume_incomplete:
        raise ValueError(
            "This API never silently restarts partial trials. Leave resume_incomplete=True."
        )
    if pending_policy not in ("all", "finish_started"):
        raise ValueError("pending_policy must be 'all' or 'finish_started'.")
    if type(additional_epochs) is not int or additional_epochs < 0:
        raise ValueError("additional_epochs must be a nonnegative integer.")
    if devices is not None and gpu_ids is not None:
        raise ValueError("Pass devices OR gpu_ids, not both.")
    if devices is None:
        if not gpu_ids or any(type(g) is not int or g < 0 for g in gpu_ids):
            raise ValueError("Specify allocated gpu_ids, or devices=['cpu'] for a CPU run.")
        devices = [f"cuda:{gpu}" for gpu in gpu_ids]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("devices must be nonempty and unique.")
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer.")
    for name, stopping_count in (("patience", patience), ("seed_patience", seed_patience)):
        if stopping_count is not None and (type(stopping_count) is not int or stopping_count < 1):
            raise ValueError(f"{name} must be None or a positive integer.")
    if (
        type(min_epochs) is not int
        or min_epochs < 1
        or (patience is not None and min_epochs > epochs["hpo"])
    ):
        raise ValueError("Invalid min_epochs for HPO.")
    if (
        not thresholds
        or any(isinstance(t, bool) or not math.isfinite(t) or not 0 < t < 1 for t in thresholds)
    ):
        raise ValueError("thresholds must be finite numbers strictly between zero and one.")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (f1_gap, iou_gap)):
        raise ValueError("f1_gap and iou_gap must be in [0, 1].")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative.")
    for name, value, positive in (
        ("max_trial_memory_mb", max_trial_memory_mb, True),
        ("oom_retry_delay_seconds", oom_retry_delay_seconds, True),
        ("oom_retry_memory_margin_mb", oom_retry_memory_margin_mb, False),
    ):
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid {name}.")
    if max_oom_attempts is not None and (type(max_oom_attempts) is not int or max_oom_attempts < 1):
        raise ValueError("max_oom_attempts must be positive or None for unlimited retries.")
    if gpu_hour_budget is not None and (
        isinstance(gpu_hour_budget, bool)
        or not isinstance(gpu_hour_budget, (int, float))
        or not math.isfinite(gpu_hour_budget)
        or gpu_hour_budget <= 0
    ):
        raise ValueError("gpu_hour_budget must be a positive finite number of GPU-hours or None.")
    if unverified_reuse not in ("refuse", "exploratory"):
        raise ValueError("unverified_reuse must be 'refuse' or 'exploratory'.")
    root = Path(output_root).expanduser().resolve() / search_name
    if (root / "automated_hpo_protocol.json").exists():
        raise ValueError("Use a NEW search_name for v3; pass the old experiment via reuse_from.")
    context = dict(backend_context or {})
    context.update(
        model_name=model_name.strip().lower(),
        dataset_root=str(Path(dataset_root).expanduser().resolve()),
        thresholds=list(thresholds),
        allow_legacy_import=allow_legacy_import,
    )
    adapter = load_backend(backend)
    base = adapter.normalize(base_parameters)
    adapter.validate(base, context)
    policies = {
        name: None
        if count is None
        else {
            "monitor": "validation_f1",
            "patience": count,
            "min_epochs": min_epochs if name == "hpo" else 1,
            "min_delta": float(min_delta),
        }
        for name, count in (("hpo", patience), ("seeds", seed_patience), ("final", None))
    }
    settings = PipelineSettings(
        root=root,
        backend=backend,
        context=context,
        base_parameters=base,
        stages=deepcopy(list(stages)),
        stage_epochs=epochs,
        policies=policies,
        seeds=list(seeds),
        devices=list(devices),
        metrics=("f1", "iou"),
        gaps={"f1": f1_gap, "iou": iou_gap},
        reuse_from=_paths(reuse_from),
        checkpoints=_paths(resume_from),
        additional_seeds=additional_seeds,
        checkpoint_interval=checkpoint_interval,
        max_trial_memory_mib=max_trial_memory_mb,
        retry_seconds=oom_retry_delay_seconds,
        memory_margin_mib=oom_retry_memory_margin_mb,
        max_oom_attempts=max_oom_attempts,
        verbose=verbose,
        gpu_hour_budget=None if gpu_hour_budget is None else float(gpu_hour_budget),
        budget_reason=budget_reason,
        unverified_reuse=unverified_reuse,
    )
    with experiment_lock(root):
        return Experiment(settings).run(
            stage,
            hpo_stage=hpo_stage,
            pending_policy=pending_policy,
            dry_run=dry_run,
            additional_epochs=additional_epochs,
            extension_id=extension_id,
            final_parameters=final_parameters,
        )
