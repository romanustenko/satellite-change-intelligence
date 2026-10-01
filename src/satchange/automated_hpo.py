
"""Model-independent, staged HPO using the project's existing GPU runner.

Public entry point: ``run_automated_parameter_search``. The controller validates
its settings, reuses completed trials, runs each search stage, and ranks the
final configurations on validation seeds. Training stays in ``optimization.py``.

All search stages are defined in ``stages``. A fresh search directory starts
from the first grid. Re-running the same search reuses completed trials and
resumes incomplete ones from their saved epochs. Explicit full checkpoints may
also seed matching trials. Test evaluation and model export remain separate.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence
from uuid import uuid4

from satchange import optimization as _optimization
from satchange.training import EarlyStopping

Parameters = dict[str, Any]
SearchSpace = dict[str, list[Any]]
Stage = tuple[str, SearchSpace]
TrialResult = dict[str, Any]
RunnerSearchSpace = SearchSpace | list[SearchSpace]

_TRAINING_KEYS = {
    "model_parameters", "seed", "image_size", "batch_size", "num_workers",
    "learning_rate", "weight_decay", "scheduler", "positive_weight",
    "bce_weight", "dice_weight", "precision",
}
_ALLOWED_PARAMETER_KEYS = _TRAINING_KEYS | {"execution_mode", "loss_weights"}
_REQUIRED_PARAMETER_KEYS = _TRAINING_KEYS - {"precision"}
_GAP_TOLERANCE = 1e-12

# Only this known scheduling-only upgrade may reuse a previous protocol.
# Epochs, training sources, checkpoints, thresholds and all HPO settings must match.
_PREVIOUS_CONTROLLER_SHA256 = "979c47a834053045ca868a9cad74150348788dc63a66db3b413246ee190ae552"
_PREVIOUS_OPTIMIZATION_SHA256 = "383b736cbce73bb94d24931c13a5d8ed6fa1fc397b0ab91cc10468d664c56dd0"
_PARALLEL_OPTIMIZATION_SHA256 = "cf57f18fa91747e22561879b2e16030049e9fe70f04199629368f40fd78f32be"
_PARALLEL_CONTROLLER_SHA256 = "7e4af7069e993fbc79925352930e927c80743c091556ebab005c866defe3a08b"
# Verified against the supplied diagnostic and the corresponding pre-resume
# trainer/runner. All non-controller package files must match this snapshot.
_LEGACY_FULL_CONTROLLER_SHA256 = "1d83c3a9b6613877896e7201671345364f759b745dce0d9b0722d3b20fbf11d4"
_LEGACY_FULL_SOURCE_SHA256 = "914eaeb140c7e57a78429128e5c9597556f144a0621ceef6098299dd8cd295b5"
_DIAGNOSTIC_SOURCE_SHA256 = "cb3c4061c4008f42eef5dc4b6f7a77816789864a50dd8c9e1c8ade83d750e49d"

@dataclass(frozen=True)
class _SearchSettings:
    """Resolved settings shared by validation, persistence, and stage execution."""

    stages: list[Stage]
    base_parameters: Parameters
    model_name: str
    dataset_root: Path
    thresholds: list[float]
    number_of_epochs: int
    early_stopping: dict[str, Any] | None
    output_root: Path
    search_name: str
    gpu_ids: list[int]
    f1_gap: float
    iou_gap: float
    evaluation_seeds: list[int]
    deployment_seed: int
    evaluate_seeds: bool
    checkpoint_interval: int
    verbose: bool
    resume_incomplete: bool
    resume_paths: list[Path]


# ---------------------------------------------------------------------------
# JSON persistence and canonical trial identity
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    """Replace derived metadata atomically; never modify trial/checkpoint files."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize_parameters(raw: Parameters) -> Parameters:
    """Keep training/architecture fields and remove runner-generated metadata."""
    parameters = {
        name: deepcopy(value)
        for name, value in raw.items()
        if name in _TRAINING_KEYS
    }
    parameters.setdefault("model_parameters", {})
    parameters.setdefault("precision", "fp32")
    # Tuples and lists must identify the same trial after JSON reload.
    return json.loads(_canonical_json(parameters))


def _merge_parameters(base: Parameters, variation: Parameters) -> Parameters:
    merged = _optimization._build_trial_parameters(base, variation)
    return _normalize_parameters(merged)


def _trial_key(
    parameters: Parameters, epochs: int, early_stopping: dict[str, Any] | None = None,
) -> str:
    identity = {"parameters": parameters, "epochs": epochs}
    if early_stopping is not None:
        identity["early_stopping"] = early_stopping
    return _canonical_json(identity)


# ---------------------------------------------------------------------------
# Input preparation and validation; no filesystem writes or GPU work
# ---------------------------------------------------------------------------


def _is_directory_name(name: str) -> bool:
    return bool(name) and Path(name).name == name and name not in (".", "..")


def _unsupported_parameters(parameters: Parameters) -> set[str]:
    return {
        name for name in parameters
        if name not in _ALLOWED_PARAMETER_KEYS and not name.startswith("model__")
    }


def _prepare_base_parameters(raw: Parameters) -> Parameters:
    if raw.get("execution_mode", "independent_hpo") != "independent_hpo":
        raise ValueError("Only independent single-GPU HPO is supported.")
    unknown = _unsupported_parameters(raw)
    if unknown:
        raise ValueError(f"Unsupported base parameters: {sorted(unknown)}")
    parameters = _merge_parameters(raw, {})
    if type(parameters.get("seed")) is not int:
        raise ValueError("base_parameters must specify an integer HPO seed.")
    return parameters


def _prepare_seeds(
    hpo_seed: int,
    seeds: Sequence[int] | None,
    deployment_seed: int | None,
) -> tuple[list[int], int]:
    evaluation_seeds = [hpo_seed] if seeds is None else list(seeds)
    invalid = (
        not evaluation_seeds
        or any(type(seed) is not int for seed in evaluation_seeds)
        or len(set(evaluation_seeds)) != len(evaluation_seeds)
    )
    if invalid:
        raise ValueError("seeds must contain unique integers.")
    if deployment_seed is None:
        deployment_seed = (
            hpo_seed if hpo_seed in evaluation_seeds else evaluation_seeds[0]
        )
    if deployment_seed not in evaluation_seeds:
        raise ValueError("deployment_seed must belong to the evaluated seeds.")
    return evaluation_seeds, deployment_seed


def _validate_settings(settings: _SearchSettings) -> None:
    if type(settings.resume_incomplete) is not bool:
        raise ValueError("resume_incomplete must be a boolean.")
    for path in settings.resume_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    if not _is_directory_name(settings.search_name):
        raise ValueError("search_name must be one nonempty directory name.")
    if type(settings.number_of_epochs) is not int or settings.number_of_epochs < 1:
        raise ValueError("number_of_epochs must be a positive integer.")
    if type(settings.checkpoint_interval) is not int or settings.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer.")

    gpu_ids = settings.gpu_ids
    if (
        not gpu_ids
        or any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("Specify unique, nonnegative allocated GPU IDs.")
    if not settings.thresholds or any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in settings.thresholds
    ):
        raise ValueError("thresholds must be finite numbers in [0, 1].")
    if any(
        not math.isfinite(gap) or not 0 <= gap <= 1
        for gap in (settings.f1_gap, settings.iou_gap)
    ):
        raise ValueError("Metric gaps must be in [0, 1].")
    if settings.model_name not in _optimization.MODEL_REGISTRY:
        raise ValueError(f"Unknown registered model: {settings.model_name}")
    _validate_stages(settings)


def _validate_stages(settings: _SearchSettings) -> None:
    names = [name for name, _ in settings.stages]
    if len(names) != len(set(names)) or any(
        not _is_directory_name(name) or name == "seeds"
        for name in names
    ):
        raise ValueError("Use unique stage directory names other than 'seeds'.")
    if not settings.stages:
        raise ValueError("Provide at least one search stage.")
    for _, space in settings.stages:
        if "seed" in space:
            raise ValueError("Use the seeds argument, not a seed search stage.")
        if "model_parameters" in space:
            raise ValueError("Search architecture fields with model__parameter.")
        invalid = _unsupported_parameters(space) | (set(space) & {"execution_mode"})
        if invalid:
            raise ValueError(f"Unsupported search parameters: {sorted(invalid)}")
        # Reuse the project's own syntax validation instead of reimplementing it.
        for variation in _optimization._expand_search_space(space):
            _merge_parameters(settings.base_parameters, variation)


def _validate_completed_trial(
    result: TrialResult, path: Path, epochs: int, thresholds: list[float]
) -> None:
    if any(
        not math.isfinite(result[metric]) or not 0 <= result[metric] <= 1
        for metric in ("best_f1", "best_iou")
    ):
        raise RuntimeError(f"Invalid validation metrics: {path}")
    if result["parameters"].get("thresholds") != thresholds:
        raise RuntimeError(f"Different threshold grid: {path}")
    if not 1 <= result["best_epoch"] <= epochs:
        raise RuntimeError(f"Inconsistent trial epoch budget: {path}")
    _validate_completion_metadata(result, path, epochs)


def _validate_completion_metadata(result: TrialResult, path: Path, epochs: int) -> None:
    """Check completion metadata before reusing a trial from this search."""
    required = {"epochs_completed", "budget_epochs", "stop_reason", "stopped_early", "early_stopping"}
    missing = required - result.keys()
    if missing:
        raise RuntimeError(f"Trial is missing completion metadata {sorted(missing)}: {path}")
    policy = result["early_stopping"]
    if policy is not None:
        if not isinstance(policy, dict):
            raise RuntimeError(f"Invalid early-stopping policy: {path}")
        try:
            stopper = EarlyStopping(**_stopping_arguments(policy))
            stopper.validate_budget(epochs)
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"Invalid early-stopping policy: {path}") from error
        if stopper.policy != policy:
            raise RuntimeError(f"Unsupported early-stopping policy: {path}")
    completed = result["epochs_completed"]
    if (
        type(completed) is not int or not 1 <= completed <= epochs
        or result.get("budget_epochs") != epochs
        or result["best_epoch"] > completed
    ):
        raise RuntimeError(f"Inconsistent completed epoch count: {path}")
    early = result.get("stop_reason") == "early_stopping"
    if result.get("stopped_early") is not early:
        raise RuntimeError(f"Inconsistent stopping status: {path}")
    if early:
        if policy is None or not policy["min_epochs"] <= completed < epochs:
            raise RuntimeError(f"Invalid early-stopped trial: {path}")
    elif result.get("stop_reason") != "max_epochs" or completed != epochs:
        raise RuntimeError(f"Trial stopped without a valid completion reason: {path}")


def _stopping_arguments(policy: dict[str, Any] | None) -> dict[str, Any]:
    """Pass only the public trainer arguments, excluding descriptive metadata."""
    if policy is None:
        return {"patience": None, "min_epochs": 1, "min_delta": 0.0}
    return {name: policy[name] for name in ("patience", "min_epochs", "min_delta")}


def _validate_selection_results(results: list[TrialResult]) -> None:
    if not results or len({result["budget_epochs"] for result in results}) != 1:
        raise RuntimeError("Selection needs completed trials at one epoch budget.")
    if len({result["parameters"]["seed"] for result in results}) != 1:
        raise RuntimeError("Search-stage selection must use one shared HPO seed.")
    if len({_canonical_json(result.get("early_stopping")) for result in results}) != 1:
        raise RuntimeError("Selection needs trials with the same early-stopping policy.")


# ---------------------------------------------------------------------------
# Protocol identity: reuse only within this search and with unchanged settings
# ---------------------------------------------------------------------------


def _training_source_hash(*, optimization_source: bytes | None = None) -> str:
    """Fingerprint training sources; the controller has its own protocol hash."""
    package = Path(_optimization.__file__).resolve().parent
    digest = sha256()
    for path in sorted(package.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        digest.update(path.relative_to(package).as_posix().encode())
        content = path.read_bytes()
        if optimization_source is not None and path.resolve() == Path(_optimization.__file__).resolve():
            content = optimization_source
        digest.update(content)
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_protocol(settings: _SearchSettings) -> dict[str, Any]:
    """Describe the model, data path, search stages, and training policy."""
    return json.loads(_canonical_json({
        "schema_version": 3,
        "resume_incomplete": settings.resume_incomplete,
        "resume_from": [
            {"path": str(path), "sha256": _file_digest(path)} for path in settings.resume_paths
        ],
        "source_sha256": _training_source_hash(),
        "controller_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": settings.model_name,
        "dataset_root": str(settings.dataset_root),
        "base": settings.base_parameters,
        "thresholds": settings.thresholds,
        "epochs": settings.number_of_epochs,
        "early_stopping": settings.early_stopping,
        "seeds": settings.evaluation_seeds,
        "deployment_seed": settings.deployment_seed,
        "evaluate_seeds": settings.evaluate_seeds,
        "gaps": [settings.f1_gap, settings.iou_gap],
        "stages": settings.stages,
    }))


def _previous_training_source_hash() -> str | None:
    """Reconstruct only the known pre-upgrade optimizer for a strict hash check.

    All other installed package files are hashed as-is. This does not waive
    source validation or allow arbitrary changes to the training code.
    """
    content = Path(_optimization.__file__).read_bytes()
    if sha256(content).hexdigest() != _PARALLEL_OPTIMIZATION_SHA256:
        return None
    text = content.decode("utf-8")
    prefix, marker, remainder = text.partition("    # BEGIN EXACT-CANDIDATE GRIDS\n")
    if not marker:
        return None
    _, marker, suffix = remainder.partition("    # END EXACT-CANDIDATE GRIDS\n")
    if not marker:
        return None
    previous = (prefix + suffix).replace(
        "search_space: dict[str, list[Any]] | list[dict[str, list[Any]]],",
        "search_space: dict[str, list[Any]],",
    ).encode("utf-8")
    if sha256(previous).hexdigest() != _PREVIOUS_OPTIMIZATION_SHA256:
        return None
    return _training_source_hash(optimization_source=previous)


def _can_upgrade_parallel_protocol(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if previous.get("controller_sha256") != _PREVIOUS_CONTROLLER_SHA256:
        return False
    source_keys = {"controller_sha256", "source_sha256"}
    if (
        {key: value for key, value in previous.items() if key not in source_keys}
        != {key: value for key, value in current.items() if key not in source_keys}
    ):
        return False
    previous_source = _previous_training_source_hash()
    return previous_source is not None and previous.get("source_sha256") == previous_source


def _can_upgrade_stage_protocol(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Allow only known releases, never ignore arbitrary source/settings changes."""
    if _can_upgrade_parallel_protocol(previous, current):
        return True

    # Last parallel release: training sources and the entire protocol are unchanged.
    if previous.get("controller_sha256") == _PARALLEL_CONTROLLER_SHA256:
        return {k: v for k, v in previous.items() if k != "controller_sha256"} == {
            k: v for k, v in current.items() if k != "controller_sha256"
        }

    # The actual Full experiment in the diagnostic predates checkpoint resume.
    # It gains automatic recovery, not a new epoch/augmentation/loss policy.
    if (
        previous.get("schema_version") != 2
        or previous.get("controller_sha256") != _LEGACY_FULL_CONTROLLER_SHA256
        or previous.get("source_sha256") != _LEGACY_FULL_SOURCE_SHA256
        or current.get("source_sha256") != _DIAGNOSTIC_SOURCE_SHA256
    ):
        return False
    upgraded = dict(previous, schema_version=3, resume_incomplete=True, resume_from=[])
    source_keys = {"controller_sha256", "source_sha256"}
    return {k: v for k, v in upgraded.items() if k not in source_keys} == {
        k: v for k, v in current.items() if k not in source_keys
    }


def _ensure_protocol(root: Path, requests: Path, settings: _SearchSettings) -> None:
    """Create a new manifest or verify a repeat call in the same search directory."""
    current = _build_protocol(settings)
    protocol_path = root / "automated_hpo_protocol.json"
    if protocol_path.exists():
        previous = _read_json(protocol_path)
        if previous == current:
            return
        if not _can_upgrade_stage_protocol(previous, current):
            raise RuntimeError(
                "Code or protocol changed; use a new search_name. Existing results are unchanged."
            )
        upgrade = "parallel" if previous.get("controller_sha256") == _PREVIOUS_CONTROLLER_SHA256 else "stage_api"
        backup = root / f"automated_hpo_protocol.before_{upgrade}.{previous['controller_sha256']}.json"
        if backup.exists() and _read_json(backup) != previous:
            raise RuntimeError("Protocol backup differs; controller upgrade stopped.")
        if not backup.exists():
            _write_json(backup, previous)
        _write_json(protocol_path, current)
        if settings.verbose:
            print("Controller upgrade: previous protocol backed up; saved trials retained.", flush=True)
        return

    if any(path != requests for path in root.iterdir()) or any(requests.iterdir()):
        raise RuntimeError(
            "The output directory contains files without a matching protocol. "
            "Use a new search_name; existing results will not be overwritten."
        )
    _write_json(protocol_path, current)


# ---------------------------------------------------------------------------
# Exact candidate expansion and grouping for the existing GPU runner
# ---------------------------------------------------------------------------


def _build_candidates(
    parents: list[Parameters], space: SearchSpace, epochs: int, *, keep_controls: bool,
    early_stopping: dict[str, Any] | None = None,
) -> dict[str, Parameters]:
    candidates: dict[str, Parameters] = {}
    for parent in parents:
        if keep_controls:
            candidates[_trial_key(parent, epochs, early_stopping)] = deepcopy(parent)
        for variation in _optimization._expand_search_space(space):
            parameters = _merge_parameters(parent, variation)
            candidates[_trial_key(parameters, epochs, early_stopping)] = parameters
    for parameters in candidates.values():
        missing = _REQUIRED_PARAMETER_KEYS - parameters.keys()
        if missing:
            raise ValueError(f"Incomplete trial parameters: {sorted(missing)}")
    return candidates


def _flatten_parameters(parameters: Parameters) -> Parameters:
    flat = {
        name: deepcopy(value)
        for name, value in parameters.items()
        if name != "model_parameters"
    }
    flat.update({
        f"model__{name}": deepcopy(value)
        for name, value in parameters["model_parameters"].items()
    })
    return flat


def _exact_trial_grids(pending: list[Parameters]) -> list[SearchSpace]:
    """Encode each candidate as one singleton grid for the existing GPU queue.

    Keeping candidates separate prevents accidental cross-products between
    unrelated parent configurations or between paired BCE/Dice weights.
    """
    return [
        {name: [value] for name, value in _flatten_parameters(parameters).items()}
        for parameters in pending
    ]


# ---------------------------------------------------------------------------
# Seed statistics and the final validation-only report
# ---------------------------------------------------------------------------


def _summarize_seed_family(
    family_key: str, results: list[TrialResult], settings: _SearchSettings
) -> dict[str, Any]:
    results.sort(key=lambda result: result["parameters"]["seed"])
    actual_seeds = [result["parameters"]["seed"] for result in results]
    if actual_seeds != sorted(settings.evaluation_seeds):
        raise RuntimeError("Missing or duplicate seeds; final ranking stopped.")
    chosen = next(
        result for result in results
        if result["parameters"]["seed"] == settings.deployment_seed
    )
    checkpoint = Path(chosen["run_directory"]) / "best.pt"
    f1_values = [result["best_f1"] for result in results]
    iou_values = [result["best_iou"] for result in results]
    return {
        "parameters": json.loads(family_key),
        "n_seeds": len(results),
        "mean_f1": mean(f1_values),
        "std_f1": stdev(f1_values) if len(results) > 1 else None,
        "mean_iou": mean(iou_values),
        "std_iou": stdev(iou_values) if len(results) > 1 else None,
        "deployment_seed": settings.deployment_seed,
        "deployment_best_checkpoint": str(checkpoint),
        "deployment_checkpoint_exists": checkpoint.is_file(),
        "deployment_validation_threshold": chosen["best_threshold"],
        "seed_results": results,
    }


def _build_final_summary(
    seed_results: list[TrialResult], settings: _SearchSettings, root: Path
) -> dict[str, Any]:
    families: dict[str, list[TrialResult]] = {}
    for result in seed_results:
        parameters = _normalize_parameters(result["parameters"])
        parameters.pop("seed")
        families.setdefault(_canonical_json(parameters), []).append(result)
    ranking = [
        _summarize_seed_family(key, results, settings)
        for key, results in families.items()
    ]
    ranking.sort(key=lambda row: (row["mean_f1"], row["mean_iou"]), reverse=True)
    best_f1 = max(row["mean_f1"] for row in ranking)
    best_iou = max(row["mean_iou"] for row in ranking)
    for row in ranking:
        row["within_soft_gap"] = (
            best_f1 - row["mean_f1"] <= settings.f1_gap + _GAP_TOLERANCE
            and best_iou - row["mean_iou"] <= settings.iou_gap + _GAP_TOLERANCE
        )
    return {
        "model_name": settings.model_name,
        "search_directory": str(root),
        "selection_split": "val",
        "test_used": False,
        "epochs_per_seed": settings.number_of_epochs,  # Budget, not actual epochs.
        "max_epochs_per_seed": settings.number_of_epochs,
        "early_stopping": settings.early_stopping,
        "seeds": settings.evaluation_seeds,
        "f1_gap": settings.f1_gap,
        "iou_gap": settings.iou_gap,
        "ranking": ranking,
        "selected_by_mean_validation_f1": ranking[0],
    }


# ---------------------------------------------------------------------------
# Stateful orchestration: disk cache, soft selection, and runner calls
# ---------------------------------------------------------------------------


class _AutomatedSearch:
    """Keep per-call state explicit instead of sharing it through nested closures."""

    def __init__(self, settings: _SearchSettings) -> None:
        self.settings = settings
        self.root = settings.output_root / settings.search_name
        self.requests = self.root / "requests"
        self.cache: dict[str, TrialResult] = {}
        self.resume_sources: dict[str, tuple[Path, int]] = {}
        self.explicit_keys: dict[str, Path] = {}
        self.used_resume_keys: set[str] = set()

    def run(self, *, stage: str | None = None) -> dict[str, Any]:
        stage_names = [name for name, _ in self.settings.stages]
        if stage is not None and stage not in [*stage_names, "seeds"]:
            raise ValueError(f"Unknown stage {stage!r}; choose from {stage_names + ['seeds']}.")
        self.requests.mkdir(parents=True, exist_ok=True)
        _ensure_protocol(self.root, self.requests, self.settings)
        self._restore_completed_requests()
        self._load_explicit_checkpoints()
        parents = [self.settings.base_parameters]

        finalists: list[TrialResult] = []
        for index, (name, space) in enumerate(self.settings.stages):
            finalists = self._run_stage(
                name, parents, space,
                keep_controls=index > 0,
                apply_selection=True,
                allow_training=stage is None or stage == name,
            )
            parents = [_normalize_parameters(result["parameters"]) for result in finalists]
            if stage == name:
                return {
                    "stage": name,
                    "search_directory": str(self.root),
                    "survivors": finalists,
                    "next_stage": stage_names[index + 1] if index + 1 < len(stage_names) else "seeds",
                    "selection_report": str(self.root / f"{name}_selection.json"),
                }

        seed_results = self._evaluate_finalists(parents, finalists)
        summary = _build_final_summary(seed_results, self.settings, self.root)
        summary["unused_resume_checkpoints"] = [
            str(path) for key, path in self.explicit_keys.items()
            if key not in self.used_resume_keys and key not in self.cache
        ]
        if summary["unused_resume_checkpoints"]:
            self._announce("Some explicit checkpoints did not match surviving stage candidates: "
                           + str(summary["unused_resume_checkpoints"]))
        path = self.root / "final_validation_summary.json"
        _write_json(path, summary)
        self._announce(f"Final validation summary: {path}")
        return summary

    def _announce(self, message: str) -> None:
        if self.settings.verbose:
            print(message, flush=True)

    def _collect_completed_trials(self, directory: Path, epochs: int) -> list[TrialResult]:
        found = []
        model_directory = directory / self.settings.model_name
        for path in sorted(model_directory.glob("trial_*/summary.json")):
            result = _read_json(path)
            if result.get("status") != "completed":
                continue
            _validate_completed_trial(result, path, epochs, self.settings.thresholds)
            result.update(budget_epochs=epochs, run_directory=str(path.parent.resolve()))
            key = _trial_key(
                _normalize_parameters(result["parameters"]), epochs,
                result.get("early_stopping"),
            )
            self.cache.setdefault(key, result)
            found.append(result)
        return found

    def _register_resume(self, key: str, path: Path, epoch: int) -> None:
        previous = self.resume_sources.get(key)
        if previous is None or epoch > previous[1]:
            self.resume_sources[key] = (path, epoch)

    def _restore_completed_requests(self) -> None:
        requests = [_read_json(path) for path in sorted(self.requests.glob("*.json"))]
        for request in requests:
            if request.get("early_stopping") != self.settings.early_stopping:
                raise RuntimeError("Saved request uses a different early-stopping policy.")
            results = self._collect_completed_trials(
                self.root / request["search_name"], request["epochs"]
            )
            self._validate_request_policy(results)
        if not self.settings.resume_incomplete:
            return
        for request in requests:
            variations = _optimization._expand_search_space(request["search_space"])
            for index, variation in enumerate(variations):
                parameters = _merge_parameters(request["fixed_parameters"], variation)
                key = _trial_key(parameters, request["epochs"], self.settings.early_stopping)
                if key in self.cache:
                    continue
                directory = self.root / request["search_name"] / self.settings.model_name / f"trial_{index:03d}"
                checkpoint = _optimization.find_training_checkpoint(directory)
                if checkpoint is not None:
                    self._register_resume(key, *checkpoint)

    def _load_explicit_checkpoints(self) -> None:
        """Associate each supplied snapshot with its actual parameters, not a trial number."""
        for path in self.settings.resume_paths:
            state = _optimization.read_hpo_checkpoint(
                path, model_name=self.settings.model_name, dataset_root=self.settings.dataset_root,
                thresholds=self.settings.thresholds, budget=self.settings.number_of_epochs,
                early_stopping=EarlyStopping(**_stopping_arguments(self.settings.early_stopping)),
            )
            parameters = _normalize_parameters(state["config"])
            key = _trial_key(parameters, self.settings.number_of_epochs, self.settings.early_stopping)
            if key in self.explicit_keys:
                raise ValueError("Provide only one explicit checkpoint for each configuration/seed.")
            self.explicit_keys[key] = path
            # A rerun must continue new progress rather than repeatedly rolling back.
            self._register_resume(key, path, state["epoch"])

    def _select_survivors(self, results: list[TrialResult], name: str) -> list[TrialResult]:
        _validate_selection_results(results)
        best_f1 = max(result["best_f1"] for result in results)
        best_iou = max(result["best_iou"] for result in results)
        report, survivors = [], []
        for result in results:
            f1_gap = best_f1 - result["best_f1"]
            iou_gap = best_iou - result["best_iou"]
            selected = (
                f1_gap <= self.settings.f1_gap + _GAP_TOLERANCE
                and iou_gap <= self.settings.iou_gap + _GAP_TOLERANCE
            )
            report.append(dict(result, selected=selected, f1_gap=f1_gap, iou_gap=iou_gap))
            if selected:
                survivors.append(result)
        _write_json(self.root / f"{name}_selection.json", report)
        if not survivors:
            raise RuntimeError("No trial meets both metric gaps; selection report saved.")
        self._announce(
            f"{name}: keep {len(survivors)}/{len(results)}; "
            f"best F1={best_f1:.6f}, IoU={best_iou:.6f}"
        )
        return survivors

    def _save_stage_plan(
        self, name: str, candidates: dict[str, Parameters], pending: list[Parameters]
    ) -> None:
        """Record new/reused trials; the stage grid is the only trial-count budget."""
        reused = len(candidates) - len(pending)
        _write_json(self.root / f"{name}_plan.json", {
            "epochs": self.settings.number_of_epochs,
            "early_stopping": self.settings.early_stopping,
            "new_trials": len(pending),
            "reused_trials": reused,
            "candidates": list(candidates.values()),
        })
        self._announce(f"{name}: new={len(pending)}, reused={reused}")

    def _execute_group(self, name: str, fixed: Parameters, space: RunnerSearchSpace) -> None:
        variations = _optimization._expand_search_space(space)
        self._announce(
            f"{name}: queue={len(variations)}, "
            f"parallel_limit={min(len(variations), len(self.settings.gpu_ids))}, "
            f"gpu_ids={self.settings.gpu_ids}"
        )
        checkpoints = {}
        for index, variation in enumerate(variations):
            parameters = _merge_parameters(fixed, variation)
            key = _trial_key(parameters, self.settings.number_of_epochs, self.settings.early_stopping)
            if key in self.resume_sources:
                path, epoch = self.resume_sources[key]
                checkpoints[index] = str(path)
                self.used_resume_keys.add(key)
                self._announce(f"{name}: resume trial {index:03d} after epoch {epoch}")
        request_name = f"{name}_{uuid4().hex[:12]}"
        _write_json(self.requests / f"{request_name}.json", {
            "search_name": request_name,
            "resume_from": checkpoints,
            "epochs": self.settings.number_of_epochs,
            "early_stopping": self.settings.early_stopping,
            "trials": len(variations),
            "fixed_parameters": fixed,
            "search_space": space,
        })
        try:
            _optimization.run_parameter_search(
                search_space=space,
                fixed_parameters=fixed,
                model_name=self.settings.model_name,
                dataset_root=self.settings.dataset_root,
                thresholds=self.settings.thresholds,
                number_of_epochs=self.settings.number_of_epochs,
                output_root=self.root,
                search_name=request_name,
                gpu_ids=self.settings.gpu_ids,
                checkpoint_interval=self.settings.checkpoint_interval,
                resume_incomplete=self.settings.resume_incomplete,
                resume_from=checkpoints,
                **_stopping_arguments(self.settings.early_stopping),
            )
        finally:
            # Keep completed siblings even if another trial fails. Propagate the error.
            results = self._collect_completed_trials(
                self.root / request_name, self.settings.number_of_epochs
            )
            self._validate_request_policy(results)

    def _validate_request_policy(self, results: list[TrialResult]) -> None:
        if any(result.get("early_stopping") != self.settings.early_stopping for result in results):
            raise RuntimeError("Runner returned a trial with a different stopping policy.")

    def _run_stage(
        self, name: str, parents: list[Parameters], space: SearchSpace,
        *, keep_controls: bool, apply_selection: bool, allow_training: bool = True,
    ) -> list[TrialResult]:
        candidates = _build_candidates(
            parents, space, self.settings.number_of_epochs, keep_controls=keep_controls,
            early_stopping=self.settings.early_stopping,
        )
        pending = [parameters for key, parameters in candidates.items() if key not in self.cache]
        if pending and not allow_training:
            raise RuntimeError(
                f"Prerequisite stage {name!r} has {len(pending)} unfinished trials. "
                f"Run stage={name!r} first. No new training was started."
            )
        self._save_stage_plan(name, candidates, pending)
        if pending:
            # One queue per stage: a free GPU takes the next pending candidate.
            self._execute_group(name, {}, _exact_trial_grids(pending))
        missing_keys = candidates.keys() - self.cache.keys()
        if missing_keys:
            raise RuntimeError(
                f"Stage {name} has {len(missing_keys)} unfinished trials; no selection made."
            )
        results = [self.cache[key] for key in candidates]
        _write_json(self.root / f"{name}_results.json", results)
        return self._select_survivors(results, name) if apply_selection else results

    def _evaluate_finalists(
        self, parents: list[Parameters], finalists: list[TrialResult]
    ) -> list[TrialResult]:
        if not self.settings.evaluate_seeds:
            return finalists
        return self._run_stage(
            "seeds", parents, {"seed": self.settings.evaluation_seeds},
            keep_controls=False, apply_selection=False,
        )


# ---------------------------------------------------------------------------
# Public API; patience=None preserves fixed-budget behavior
# ---------------------------------------------------------------------------


def run_automated_parameter_search(
    *,
    stages: Sequence[Stage],
    base_parameters: Parameters,
    model_name: str,
    dataset_root: str | Path,
    thresholds: Sequence[float],
    number_of_epochs: int,
    output_root: str | Path,
    search_name: str,
    gpu_ids: Sequence[int],
    f1_gap: float = 0.04,
    iou_gap: float = 0.05,
    seeds: Sequence[int] | None = None,
    deployment_seed: int | None = None,
    checkpoint_interval: int | None = None,
    verbose: bool = True,
    patience: int | None = None,
    min_epochs: int = 1,
    min_delta: float = 0.0,
    resume_incomplete: bool = True,
    resume_from: str | Path | Sequence[str | Path] | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Run all HPO stages, or one named stage, with validation-only soft selection.

    Pass the SAME complete ``stages`` list on every call. ``stage="03_loss"``
    runs only that named stage. Preceding stages are reconstructed from completed
    results (including their soft selection); a missing prerequisite fails before
    any GPU training. Later stages are NOT launched. The return contains
    ``stage``, ``survivors``, ``next_stage`` and the selection-report path.
    ``stage="seeds"`` requires all HPO stages complete, then performs configured
    seed evaluation and writes the final summary. With seeds=None it only writes
    the single-HPO-seed summary. ``stage=None`` preserves the original all-stage
    behavior. ``stage`` is execution control, not part of the saved protocol.

    ``stages`` contains ``(name, search_space)`` pairs in execution order.
    Every new search starts with exactly the first grid. Later stages expand
    the surviving configurations and retain each parent as a control.
    All candidates within BOTH metric gaps survive; there is no fixed top-k.
    Search syntax is inherited from optimization.py, including paired
    ``loss_weights=[(BCE, Dice), ...]`` and ``model__parameter`` fields.

    ``base_parameters`` supplies defaults; each stage overrides its searched
    fields. New configurations train from scratch using run_parameter_search.
    All pending candidates of a stage share one queue across gpu_ids. Each
    GPU runs at most one independent trial from this search, and starts another
    as soon as it becomes free. Selection waits for the complete stage.
    No additional trial-count limit is applied. Old grouped request records
    remain readable for both completed-result reuse and checkpoint recovery.

    ``patience=None`` trains for the full number_of_epochs. Otherwise, the
    trainer stops after patience epochs without a cumulative F1 gain greater
    than min_delta, but never before min_epochs. Each raw F1 maximum still
    updates best.pt. One policy applies to all stages and final seeds.

    Optional ``seeds`` run after the search stages without per-seed pruning.
    The final ranking uses mean validation F1, then mean IoU. The predetermined
    deployment_seed is not selected by its score. With seeds=None, results
    from the single HPO seed are returned without another training stage.

    Use a NEW search_name to start from scratch. A repeat call with the SAME
    name, settings, and code reuses completed trials only from this search.
    Known previous releases can migrate with a protocol backup and strict source
    checks, including the supplied legacy Full snapshot. Existing training settings
    must remain unchanged. This is not a general bypass of source validation.
    With resume_incomplete=True (default), incomplete trials use last.pt or a
    readable numbered fallback. Only unsaved work in the interrupted epoch is lost.
    resume_from optionally supplies one full checkpoint or a list: these seed
    matching candidates, do not add configurations or resurrect pruned parents.
    Use a NEW search_name for an explicit rollback/changed budget. The epoch budget
    here remains an ABSOLUTE maximum, not additional epochs. Stopping and training
    parameters must still match; a saved epoch above the target is rejected.
    Completed siblings are collected before errors propagate. Only transient
    CUDA OOM is retried automatically; after other failures fix the underlying
    problem and repeat the call. Load only your own trusted full checkpoints.
    Do not run two controllers in one directory or change the data/code during
    a search. Test evaluation, DDP, and model export are not performed here.
    """
    stopper = EarlyStopping(patience, min_epochs, min_delta)
    stopper.validate_budget(number_of_epochs)
    base = _prepare_base_parameters(base_parameters)
    evaluation_seeds, chosen_seed = _prepare_seeds(base["seed"], seeds, deployment_seed)
    supplied = [] if resume_from is None else (
        [resume_from] if isinstance(resume_from, (str, Path)) else list(resume_from)
    )
    settings = _SearchSettings(
        stages=deepcopy(list(stages)),
        base_parameters=base,
        model_name=model_name.strip().lower(),
        dataset_root=Path(dataset_root).resolve(),
        thresholds=list(thresholds),
        number_of_epochs=number_of_epochs,
        early_stopping=stopper.policy,
        output_root=Path(output_root).resolve(),
        search_name=search_name,
        gpu_ids=list(gpu_ids),
        f1_gap=f1_gap,
        iou_gap=iou_gap,
        evaluation_seeds=evaluation_seeds,
        deployment_seed=chosen_seed,
        evaluate_seeds=seeds is not None,
        checkpoint_interval=(
            number_of_epochs if checkpoint_interval is None else checkpoint_interval
        ),
        verbose=verbose,
        resume_incomplete=resume_incomplete,
        resume_paths=[Path(path).resolve() for path in supplied],
    )
    _validate_settings(settings)
    return _AutomatedSearch(settings).run(stage=stage)
