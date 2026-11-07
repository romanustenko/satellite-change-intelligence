"""Satellite Change Intelligence adapter for hpo_core.

Only this file knows the project's data/model/loss/trainer/checkpoint schema.
Legacy experiment folders are read-only. New runs live in the NEW search folder.
All .pt inputs must be your own checkpoints (the project uses full torch.load).
"""

from __future__ import annotations

import inspect
import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch

from satchange.data import RandomFlipRotate, create_dataset_dataloader
from satchange.execution import resolve_device, seed_everything
from satchange.hpo_core import canonical, read_json, write_json
from satchange.losses import BCEDiceLoss
from satchange.models.factory import (
    BINARY_TASK,
    MODEL_REGISTRY,
    count_model_parameters,
    create_task_model,
    model_class_for_task,
)
from satchange.optimization import (
    _build_trial_parameters,
    _create_scheduler,
    _is_cuda_oom_error,
    _parse_cuda_oom_message,
)
from satchange.training import EarlyStopping, _save_checkpoint, fit_binary_change_model

TRAIN_KEYS = {
    "seed",
    "model_parameters",
    "image_size",
    "batch_size",
    "num_workers",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "positive_weight",
    "bce_weight",
    "dice_weight",
    "precision",
}
# Optional identity keys: absent means the historical behaviour (scratch initialization), so
# existing trial identities are unchanged.
OPTIONAL_TRAIN_KEYS = {"initial_weights"}
GENERATED_KEYS = {
    "epochs",
    "number_of_epochs",
    "code_revision",
    "torch_version",
    "cuda_version",
    "thresholds",
    "world_size",
    "execution_mode",
    "architecture_variant",
    "model_name",
    "early_stopping",
    "patience",
    "min_epochs",
    "min_delta",
    "resume_from",
    "source_checkpoint",
    "target_epoch",
    "hpo_phase",
    "run_name",
    # Legacy run metadata: accepted on input, never part of a configuration identity.
    # dataset_root is checked against the requested data before a checkpoint is reused.
    "budget_epochs",
    "dataset_root",
    "training_data",
    "gpu_id",
}


HPO_BACKEND_METADATA_VERSION = 1


def _load(path: Path) -> dict[str, Any]:
    """mmap avoids reading every tensor while indexing project checkpoint metadata."""
    with path.open("rb") as handle:
        prefix = handle.read(100)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"Git LFS pointer, not actual weights: {path}")
    payload = torch.load(
        path, map_location="cpu", weights_only=False, mmap=prefix.startswith(b"PK")
    )
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Expected a complete project checkpoint, not a raw state_dict: {path}")
    return payload


def _policy(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if "policy" in value:
        value = value["policy"]
    if value is None or value.get("patience") is None:
        return None
    return EarlyStopping(
        **{key: value[key] for key in ("patience", "min_epochs", "min_delta")},
        monitor=value.get("monitor", "validation_f1"),
    ).policy


def _stopping_arguments(policy: dict[str, Any] | None) -> dict[str, Any]:
    return (
        {"patience": None}
        if policy is None
        else {key: policy[key] for key in ("patience", "min_epochs", "min_delta")}
    )


def _check_dataset(document: Mapping[str, Any], context: Mapping[str, Any], source: Path) -> None:
    """Validate saved dataset provenance without adding it to the parameter hash.

    Relative legacy paths refer to the source project, never the notebook's cwd.
    A relocated project may supply backend_context={"project_root": ...} explicitly.
    Path equality checks provenance, not the contents/version of the dataset.
    """
    if "dataset_root" not in document:
        return  # Older files may lack this field; do not invent provenance.
    value = document["dataset_root"]
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Invalid saved dataset_root in {source}: {value!r}")
    saved = Path(value).expanduser()
    if not saved.is_absolute():
        base: Path | None
        project_root = context.get("project_root")
        if project_root is not None:
            base = Path(project_root).expanduser().resolve()
        else:
            base = next(
                (
                    parent
                    for parent in source.resolve().parents
                    if (parent / "src/satchange").is_dir() or (parent / "pyproject.toml").is_file()
                ),
                None,
            )
        if base is None:
            raise ValueError(
                f"Cannot resolve relative dataset_root={value!r} in {source}. "
                'Supply backend_context={"project_root": <source project root>}.'
            )
        saved = base / saved
    expected = Path(context["dataset_root"]).expanduser().resolve()
    if saved.resolve() != expected:
        raise ValueError(
            f"Dataset path differs in {source}: saved={saved.resolve()}, expected={expected}"
        )


def _saved_budget(
    summary: Mapping[str, Any], checkpoint: Mapping[str, Any], fallback: int | None, source: Path
) -> int | None:
    """Read the run's total budget, not an additional-epochs training argument."""
    config = checkpoint.get("config") or {}
    # A current summary or final-budget annotation can supersede a historical
    # best.pt config. Request/protocol budgets are preferable to legacy 'epochs'.
    candidates = (
        summary.get("budget_epochs"),
        checkpoint.get("training_budget_epochs"),
        config.get("budget_epochs"),
        config.get("target_epoch"),
        fallback,
        config.get("epochs"),
    )
    value = next((value for value in candidates if value is not None), None)
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"Invalid saved epoch budget in {source}: {value!r}")
    return value


class SatelliteBackend:
    """Other models use MODEL_REGISTRY; other projects provide their own adapter."""

    def normalize(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        merged = _build_trial_parameters(dict(parameters), {})
        # Preserve None, zero and false. Do not silently discard unknown training controls.
        keys = TRAIN_KEYS | OPTIONAL_TRAIN_KEYS
        unknown = set(merged) - keys - GENERATED_KEYS
        if unknown:
            raise ValueError(f"Unsupported training fields: {sorted(unknown)}")
        result = {key: deepcopy(value) for key, value in merged.items() if key in keys}
        result.setdefault("precision", "fp32")
        result.setdefault("model_parameters", {})
        return cast(dict[str, Any], json.loads(canonical(result)))

    def merge(self, base: dict, variation: dict) -> dict:
        return self.normalize(_build_trial_parameters(base, variation))

    def validate(self, parameters: dict, context: dict) -> None:
        if torch.distributed.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Independent HPO cannot run inside torchrun/DDP.")
        missing = TRAIN_KEYS - parameters.keys()
        if missing:
            raise ValueError(f"Missing training parameters: {sorted(missing)}")
        if parameters.get("initial_weights") is not None:
            _validate_initial_weights(parameters["initial_weights"])
        name = context["model_name"]
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model {name!r}; register it in MODEL_REGISTRY.")
        inspect.signature(MODEL_REGISTRY[name]).bind(**parameters["model_parameters"])
        for key, minimum in (("seed", 0), ("batch_size", 1), ("num_workers", 0)):
            if type(parameters[key]) is not int or parameters[key] < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        size = parameters["image_size"]
        if size is not None and (type(size) is not int or size < 1):
            raise ValueError("image_size must be a positive integer or None.")
        if parameters["precision"] not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16.")
        if parameters["scheduler"] not in ("none", "reduce_on_plateau"):
            raise ValueError("Use a scheduler supported by optimization._create_scheduler.")
        for key in ("learning_rate", "weight_decay"):
            value = parameters[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Invalid {key}: {value}")
        BCEDiceLoss(
            bce_weight=parameters["bce_weight"],
            dice_weight=parameters["dice_weight"],
            pos_weight=parameters["positive_weight"],
        )

    def code_signature(self) -> str:
        package = Path(__file__).resolve().parent
        names = {
            "training.py",
            "execution.py",
            "losses.py",
            "metrics.py",
            "hpo_backend.py",
            "hpo_core.py",
            "automated_hpo.py",
            "optimization.py",
            "run.py",
        }
        files = [
            p
            for p in package.rglob("*.py")
            if p.name in names or p.parent.name in ("data", "models", "semantic")
        ]
        digest = sha256()
        for path in sorted(files):
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    # Schema-4 code identity (D-022): per-file hashes of normalized (CRLF->LF) sources, limited
    # to the files this experiment's task and model depend on. The schema-3 code_signature()
    # above is kept unchanged as the legacy algorithm for old protocols and migration evidence.
    CONTROL_FILES = ("hpo_core.py", "automated_hpo.py")
    MONITORING_FILES = ("hpo_accounting.py",)
    TRAINING_FILES = (
        "training.py",
        "execution.py",
        "losses.py",
        "metrics.py",
        "optimization.py",
        "hpo_backend.py",
        "data/__init__.py",
        "data/levir_cd.py",
        "data/second_binary.py",
        "data/transforms.py",
        "semantic/data.py",
        "semantic/second.py",
        "models/__init__.py",
        "models/factory.py",
    )

    def code_identity(self, context: Mapping[str, Any]) -> dict[str, Any]:
        package = Path(__file__).resolve().parent
        task = context.get("task") or BINARY_TASK
        categories = dict.fromkeys(self.CONTROL_FILES, "control")
        categories.update(dict.fromkeys(self.MONITORING_FILES, "monitoring"))
        categories.update(dict.fromkeys(self.TRAINING_FILES, "training"))
        source = inspect.getsourcefile(model_class_for_task(context["model_name"], task))
        if source is None:
            raise ValueError(f"Cannot locate the source of model {context['model_name']!r}.")
        categories[Path(source).resolve().relative_to(package).as_posix()] = "training"
        files = {
            name: sha256((package / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in sorted(categories)
        }
        return {
            "scheme": "satchange-code-v2",
            "normalization": "CRLF->LF",
            "files": files,
            "categories": {name: categories[name] for name in sorted(categories)},
        }

    def _inspect(
        self, directory: Path, context: dict, budget: int | None, explicit: Path | None = None
    ) -> dict | None:
        summary_path = directory / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        last_path = explicit or directory / "last.pt"
        best_path = directory / "best.pt"
        path = last_path if last_path.exists() else best_path if best_path.exists() else None
        if path is None:
            return None  # A request/empty folder alone is NOT evidence that training started.
        checkpoint = _load(path)
        if checkpoint.get("model_name") != context["model_name"]:
            if explicit:
                raise ValueError(f"Explicit checkpoint belongs to another model: {path}")
            return None
        raw = checkpoint.get("config") or summary.get("parameters")
        if not raw:
            raise ValueError(f"Checkpoint has no training configuration: {path}")
        _check_dataset(raw, context, path)
        _check_dataset(summary, context, summary_path)
        _check_dataset(summary.get("parameters") or {}, context, summary_path)
        try:
            parameters = self.normalize(raw)
            self.validate(parameters, context)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid training configuration in {path}: {error}") from error
        if summary.get("parameters") and self.normalize(summary["parameters"]) != parameters:
            raise ValueError(f"summary.json parameters disagree with checkpoint: {directory}")
        if checkpoint.get("task") is not None:
            raise ValueError(f"Checkpoint task differs: {path}")
        thresholds = raw.get("thresholds", [] if context.get("allow_legacy_import") else None)
        if thresholds != context["thresholds"]:
            raise ValueError(f"Threshold grid differs in {path}")
        if (
            checkpoint.get("precision", "fp32") != parameters["precision"]
            or checkpoint.get("world_size", 1) != 1
        ):
            raise ValueError(f"Checkpoint precision/world size is incompatible: {path}")
        epoch = checkpoint.get("epoch")
        if type(epoch) is not int or epoch < 1:
            raise ValueError(f"Invalid checkpoint epoch: {path}")
        best = checkpoint.get("best_metrics") or checkpoint.get("current_metrics")
        if not best:
            raise ValueError(f"Checkpoint is missing validation metrics: {path}")
        policy = _policy(checkpoint.get("early_stopping"))
        reason = checkpoint.get("stop_reason")
        budget = _saved_budget(summary, checkpoint, budget, path)
        complete = reason in ("max_epochs", "early_stopping")
        if summary.get("status") == "completed":
            reported_epoch = summary.get("epochs_completed", budget)
            if last_path.exists() and reported_epoch is not None and reported_epoch != epoch:
                # The last checkpoint is newer than a stale summary after a crashed resume.
                complete = reason in ("max_epochs", "early_stopping")
            elif reported_epoch is not None:
                complete = True
        if reason is None and complete:
            reason = summary.get("stop_reason", "max_epochs")
        if not last_path.exists() and complete:
            # best.pt may be from epoch 12, but a trusted summary can prove training reached 40.
            epoch = summary.get("epochs_completed", budget) or epoch
        if reason == "max_epochs" and budget is None:
            budget = epoch
        if explicit:
            # An explicitly supplied best.pt means continue FROM ITS ACTUAL epoch.
            epoch = checkpoint["epoch"]
            complete = False
        metrics = self._metrics(best)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
            raise ValueError(f"Invalid metrics in {path}")
        best_epoch = int(best["epoch"])
        if best_path.exists():
            best_payload = _load(best_path)
            _check_dataset(best_payload.get("config") or {}, context, best_path)
            if (
                best_payload.get("model_name") != context["model_name"]
                or self.normalize(best_payload.get("config", {})) != parameters
            ):
                raise ValueError(f"best.pt belongs to a different configuration/seed: {directory}")
            if best_payload["epoch"] != best_epoch:
                if best_epoch == checkpoint["epoch"]:
                    best_path = path  # Crash between atomic last.pt and best.pt commits.
                else:
                    raise ValueError(f"best.pt does not match last.pt's best epoch: {directory}")
            if best_path != path and best_payload["epoch"] == best_epoch:
                saved_best = best_payload.get("current_metrics") or best_payload.get(
                    "best_metrics", {}
                )
                for metric, value in metrics.items():
                    saved_value = saved_best.get(f"validation_{metric}")
                    if saved_value is None or abs(float(saved_value) - value) > 1e-8:
                        raise ValueError(
                            f"best.pt {metric} disagrees with last.pt metadata: {directory}"
                        )
            del best_payload
        elif best_epoch == checkpoint["epoch"]:
            best_path = path
        else:
            raise FileNotFoundError(
                f"Missing best.pt for historical epoch {best_epoch}: {directory}"
            )
        row = {
            "parameters": parameters,
            "epoch": epoch,
            "policy": policy,
            "stop_reason": reason,
            "status": "completed" if complete else "partial",
            "budget_epochs": budget,
            "run_directory": str(directory.resolve()),
            "run_name": checkpoint["run_name"],
            "last_checkpoint": str(last_path.resolve()) if last_path.exists() else None,
            "best_checkpoint": str(best_path.resolve()),
            "best_epoch": best_epoch,
            **self._summary_metrics(best),
            "metrics": metrics,
            "model_name": context["model_name"],
            "training_seconds": summary.get("training_seconds"),
            "gpu_hours": summary.get("gpu_hours"),
            "imported_from": str(path.resolve()),
            "summary_path": str(summary_path.resolve()),
        }
        del checkpoint
        return row

    @staticmethod
    def _metrics(best: Mapping[str, Any]) -> dict[str, float]:
        return {"f1": float(best["validation_f1"]), "iou": float(best["validation_iou"])}

    @classmethod
    def _summary_metrics(cls, best: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "best_f1": float(best["validation_f1"]),
            "best_iou": float(best["validation_iou"]),
            "best_threshold": float(best["best_threshold"]),
        }

    def discover(
        self, roots: Sequence[str], checkpoints: Sequence[str], context: dict
    ) -> list[dict]:
        records, seen = [], set()
        for root_value in roots:
            root = Path(root_value).resolve()
            if not root.is_dir():
                raise FileNotFoundError(root)
            protocol_path = root / "automated_hpo_protocol.json"
            protocol = read_json(protocol_path) if protocol_path.exists() else {}
            if protocol:
                if not context.get("allow_legacy_import"):
                    raise ValueError(
                        "Set allow_legacy_import=True to import your old experiment explicitly."
                    )
                if (
                    protocol.get("model") != context["model_name"]
                    or protocol.get("thresholds") != context["thresholds"]
                ):
                    raise ValueError(f"Legacy model/threshold protocol differs: {protocol_path}")
                if (
                    Path(protocol["dataset_root"]).resolve()
                    != Path(context["dataset_root"]).resolve()
                ):
                    raise ValueError(f"Legacy dataset path differs: {protocol_path}")
            # A previous v3 run can also be reused from a new search_name after a
            # backend fix. Its saved context remains binding even if individual
            # older checkpoints did not record dataset_root yet.
            pipeline_path = root / "pipeline_protocol.json"
            if pipeline_path.exists():
                saved_context = read_json(pipeline_path).get("context", {})
                if (
                    saved_context.get("model_name") != context["model_name"]
                    or saved_context.get("thresholds") != context["thresholds"]
                ):
                    raise ValueError(f"Pipeline model/threshold protocol differs: {pipeline_path}")
                if saved_context.get("task") != context.get("task"):
                    raise ValueError(f"Pipeline task differs: {pipeline_path}")
                _check_dataset(saved_context, context, pipeline_path)
            budgets = {}
            requests = root / "requests"
            if requests.exists():
                for request_path in requests.glob("*.json"):
                    request = read_json(request_path)
                    if "search_name" in request and "epochs" in request:
                        budgets[request["search_name"]] = request["epochs"]
            for current, directories, filenames in os.walk(root):
                directories[:] = [
                    d
                    for d in directories
                    if not d.startswith(".")
                    and d not in {"checkpoints", "operations", "__pycache__", "diagnostics"}
                ]
                path = Path(current)
                if path.parent.name != context["model_name"] or not (
                    {"last.pt", "best.pt"} & set(filenames)
                ):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                relative = path.relative_to(root)
                request_name = relative.parts[0] if len(relative.parts) > 2 else ""
                row = self._inspect(
                    path, context, budgets.get(request_name, protocol.get("epochs"))
                )
                if row:
                    records.append(row)
        for value in checkpoints:
            path = Path(value).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            row = self._inspect(path.parent, context, None, explicit=path)
            if row:
                records.append(row)
        return records

    def resources(self, devices: Sequence[str]) -> dict[str, dict]:
        result: dict[str, dict[str, Any]] = {}
        for name in devices:
            device = torch.device(name)
            if device.type == "cpu":
                result[name] = {"free_mib": float("inf"), "total_mib": 0.0}
                continue
            if device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError(f"Unavailable CUDA device: {name}")
            if device.index is None or device.index >= torch.cuda.device_count():
                raise ValueError(f"Use a visible logical CUDA index: {name}")
            free, total = torch.cuda.mem_get_info(device)
            result[name] = {
                "free_mib": free / 1024**2,
                "total_mib": total / 1024**2,
                "name": torch.cuda.get_device_name(device),
            }
        return result

    def train(self, job: dict, device: str, stop_file: str) -> dict:
        try:
            return self._train(job, device, stop_file)
        except Exception as error:
            if not _is_cuda_oom_error(error):
                raise
            parsed = _parse_cuda_oom_message(str(error))
            allocated, requested = parsed["allocated_by_pytorch_mb"], parsed["tried_to_allocate_mb"]
            # Do not add reserved cache or OTHER processes' memory to the requirement.
            required = (
                allocated + requested if allocated is not None and requested is not None else None
            )
            return {
                "status": "oom",
                "error": str(error),
                "required_mib": required,
                "requirement_kind": "live_tensors_plus_failed_allocation_lower_bound"
                if required
                else "unknown",
                "parameters": job["parameters"],
                "target_epoch": job["target_epoch"],
            }

    def _train(self, job: dict, device_name: str, stop_file: str) -> dict:
        trainer_parameters = inspect.signature(fit_binary_change_model).parameters
        if not {"stop_file", "reset_early_stopping"}.issubset(trainer_parameters):
            raise RuntimeError(
                "Run scripts/install_hpo_v3.py and restart the kernel; trainer patch is missing."
            )
        context = job["context"]
        parameters = deepcopy(job["parameters"])
        self.validate(parameters, context)
        if context.get("task") not in (None, BINARY_TASK):
            raise ValueError(f"Unsupported task: {context.get('task')}")
        score_key = "validation_f1"
        device = resolve_device(device_name)
        if device.type == "cpu":
            torch.set_num_threads(context.get("cpu_threads", 1))
        if device.type == "cuda":
            torch.cuda.set_device(device)
        seed_everything(parameters["seed"])
        source = job.get("source")
        run_name = source["run_name"] if source else f"seed_{parameters['seed']}"
        run_dir = Path(job["directory"]) / context["model_name"] / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(Path(job["directory"]) / "job.json", job)
        local_last = run_dir / "last.pt"
        local_best = run_dir / "best.pt"
        source_epoch = 0
        if source and not local_last.exists():
            source_path = Path(source["last_checkpoint"])
            check = _load(source_path)
            _check_dataset(check.get("config") or {}, context, source_path)
            if check["epoch"] != source["epoch"] or self.normalize(check["config"]) != parameters:
                raise ValueError(
                    "Source checkpoint changed after plan creation. "
                    "Stop the old training process first."
                )
            del check
            # Best is copied FIRST; if interrupted before last is copied, retry is harmless.
            _atomic_copy(Path(source["best_checkpoint"]), local_best)
            _atomic_copy(source_path, local_last)
        saved = _load(local_last) if local_last.exists() else None
        reset_stopper = False
        if saved is not None:
            _check_dataset(saved.get("config") or {}, context, local_last)
            source_epoch = saved["epoch"]
            required = {
                "optimizer_state_dict",
                "scheduler_state_dict",
                "rng_states",
                "history",
                "early_stopping",
            }
            missing = required - saved.keys()
            if missing:
                raise ValueError(
                    f"Cannot perform exact state resume; missing {sorted(missing)} in {local_last}"
                )
            if (
                saved.get("world_size", 1) != 1
                or saved.get("precision", "fp32") != parameters["precision"]
            ):
                raise ValueError("Resume world size or precision mismatch.")
            if self.normalize(saved["config"]) != parameters:
                raise ValueError(
                    "Resume training parameters differ; do "
                    "not transplant optimizer state across trials."
                )
            history = saved["history"]
            if [row["epoch"] for row in history] != list(range(1, source_epoch + 1)):
                raise ValueError("Resume requires a complete ordered history from epoch 1.")
            if len(saved["rng_states"]) != 1 or saved["rng_states"][0].get("loader") is None:
                raise ValueError("Checkpoint lacks the shuffled training-loader RNG state.")
            saved_best = saved.get("best_metrics") or max(history, key=lambda row: row[score_key])
            if saved_best["epoch"] == source_epoch:
                # Repair a prior pause/crash after last.pt was committed but before best.pt.
                _atomic_copy(local_last, local_best)
            elif not local_best.exists() or _load(local_best)["epoch"] != saved_best["epoch"]:
                raise ValueError(
                    "Historical best checkpoint is missing or inconsistent; refusing to lose it."
                )
            reset_stopper = _policy(saved["early_stopping"]) != job["policy"]
            del saved
        else:
            history = []
        remaining = job["target_epoch"] - source_epoch
        if remaining < 0:
            raise ValueError(
                "Local checkpoint is beyond the requested target; refusing to truncate history."
            )
        if Path(stop_file).exists() and remaining:
            return {"status": "paused", "epoch": source_epoch, "parameters": parameters}
        if remaining:
            self_message = (
                f"seed={parameters['seed']}: epoch {source_epoch} "
                f"-> {job['target_epoch']} (+{remaining})"
            )
            print(
                f"[RESUME] {self_message}" if source_epoch else f"[FRESH] {self_message}",
                flush=True,
            )
            if reset_stopper:
                print(
                    "[POLICY] New phase stopping policy; "
                    "optimizer, scheduler and RNG are retained.",
                    flush=True,
                )
            loss: torch.nn.Module
            data_options = dict(
                dataset_root=context["dataset_root"],
                image_size=parameters["image_size"],
                batch_size=parameters["batch_size"],
                num_workers=parameters["num_workers"],
                pin_memory=device.type == "cuda",
            )
            _, train_loader = create_dataset_dataloader(
                **data_options,
                split="train",
                shuffle=True,
                seed=parameters["seed"],
                transform=RandomFlipRotate(
                    horizontal_flip_probability=0.5, vertical_flip_probability=0.5
                ),
            )
            _, val_loader = create_dataset_dataloader(**data_options, split="val")
            loss = BCEDiceLoss(
                bce_weight=parameters["bce_weight"],
                dice_weight=parameters["dice_weight"],
                pos_weight=parameters["positive_weight"],
            ).to(device)
            construction = deepcopy(parameters["model_parameters"])
            if source_epoch and "pretrained_backbone" in construction:
                construction["pretrained_backbone"] = (
                    False  # Trained state is about to be restored.
                )
            model = create_task_model(
                context["model_name"], construction, context.get("task", "binary_change_detection")
            ).to(device)
            counts = count_model_parameters(model)
            initialization = None
            if not source_epoch and parameters.get("initial_weights") is not None:
                # Fresh start only; a resumed run restores its own trained state below.
                initialization = _load_initial_weights(model, parameters["initial_weights"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=parameters["learning_rate"],
                weight_decay=parameters["weight_decay"],
            )
            scheduler = _create_scheduler(optimizer, parameters["scheduler"])
            config = {
                **parameters,
                "thresholds": context["thresholds"],
                "epochs": job["target_epoch"],
                "budget_epochs": job["target_epoch"],
                "dataset_root": context["dataset_root"],
                "execution_mode": "independent_hpo",
                "world_size": 1,
                "hpo_phase": job["phase"],
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
            }
            try:
                config["code_revision"] = subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
                ).stdout.strip()
            except OSError:
                config["code_revision"] = "unknown"
            try:
                history = fit_binary_change_model(
                    model=model,
                    train_loader=train_loader,
                    validation_loader=val_loader,
                    loss_function=loss,
                    optimizer=optimizer,
                    device=device,
                    number_of_epochs=remaining,
                    thresholds=context["thresholds"],
                    output_root=job["directory"],
                    model_name=context["model_name"],
                    run_name=run_name,
                    checkpoint_interval=job["checkpoint_interval"],
                    config=config,
                    resume_from=local_last if source_epoch else None,
                    scheduler=scheduler,
                    precision=parameters["precision"],
                    log_prefix=f"[{job['id']} | {device}] ",
                    stop_file=stop_file,
                    reset_early_stopping=reset_stopper,
                    task=context.get("task", "binary_change_detection"),
                    **_stopping_arguments(job["policy"]),
                )
            finally:
                del model, loss, optimizer, scheduler, train_loader, val_loader
        else:
            counts = {}
            config = {
                **parameters,
                "thresholds": context["thresholds"],
                "epochs": job["target_epoch"],
                "budget_epochs": job["target_epoch"],
                "dataset_root": context["dataset_root"],
            }
        if not history:
            return {"status": "paused", "epoch": source_epoch, "parameters": parameters}
        epoch = history[-1]["epoch"]
        best = max(history, key=lambda row: row[score_key])
        reason = history[-1].get("stop_reason")
        complete = epoch == job["target_epoch"] or (
            reason == "early_stopping" and job["policy"] is not None
        )
        seconds = sum(float(row.get("epoch_seconds", 0)) for row in history)
        metrics = self._metrics(best)
        if complete:
            # A historical best may have been found in an earlier phase. Its weights,
            # optimizer, epoch and history remain unchanged, but the NEW copy belongs
            # to the current training budget. Keep the original config as provenance.
            best_payload = _load(local_best)
            best_payload.setdefault(
                "original_training_config", deepcopy(best_payload.get("config", {}))
            )
            best_payload["config"] = {
                **best_payload.get("config", {}),
                "epochs": job["target_epoch"],
                "hpo_phase": job["phase"],
                "budget_epochs": job["target_epoch"],
                "dataset_root": context["dataset_root"],
            }
            best_payload["training_completed_epoch"] = epoch
            best_payload["training_budget_epochs"] = job["target_epoch"]
            _save_checkpoint(local_best, best_payload)
            del best_payload
        summary = {
            "trial": run_name,
            "status": "completed" if complete else "paused",
            "epoch": epoch,
            "epochs_completed": epoch,
            "budget_epochs": job["target_epoch"],
            "stop_reason": "max_epochs" if epoch == job["target_epoch"] else reason,
            "stopped_early": complete and epoch < job["target_epoch"],
            "early_stopping": job["policy"],
            "policy": job["policy"],
            "parameters": config,
            "best_epoch": best["epoch"],
            **self._summary_metrics(best),
            "metrics": metrics,
            "best_validation_loss": best.get("validation_loss"),
            "best_precision": best.get("validation_precision"),
            "best_recall": best.get("validation_recall"),
            "training_seconds": seconds,
            "seconds_per_epoch": seconds / epoch,
            "timing_scope": "sum of saved epoch_seconds; not end-to-end controller wall time",
            "gpu_hours": sum(float(row.get("gpu_hours", 0)) for row in history),
            # Epochs inherited from an earlier phase's checkpoint are excluded here, so phase
            # totals can be summed without double counting. Allocation time: gpu_ledger.jsonl.
            "epoch_compute_gpu_hours_this_attempt": sum(
                float(row.get("gpu_hours", 0)) for row in history if row["epoch"] > source_epoch
            ),
            "peak_gpu_memory_mb": max(row.get("peak_gpu_memory_mb", 0) for row in history),
            "run_directory": str(run_dir),
            "run_name": run_name,
            "last_checkpoint": str(local_last),
            "best_checkpoint": str(local_best),
            "model_name": context["model_name"],
            "device": device_name,
            "resume_source": str(local_last) if source_epoch else None,
            "resumed_from_epoch": source_epoch,
            "initial_weights_report": initialization,
            "new_epochs_completed": epoch - source_epoch,
            "early_stopping_reset": reset_stopper,
            **counts,
        }
        write_json(run_dir / "summary.json", summary)
        return {**summary, "parameters": parameters}


def _validate_initial_weights(value: Any) -> None:
    """initial_weights = {"path": str, "sha256": 64-hex, "exclude": [state-dict key prefixes]}."""
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "exclude"}:
        raise ValueError("initial_weights needs exactly path, sha256 and exclude.")
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError("initial_weights.path must be a non-empty string.")
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ValueError("initial_weights.sha256 must be a lowercase 64-character hex digest.")
    exclude = value["exclude"]
    if not isinstance(exclude, list) or any(not isinstance(k, str) or not k for k in exclude):
        raise ValueError("initial_weights.exclude must be a list of non-empty key prefixes.")


def _load_initial_weights(model: torch.nn.Module, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Load trusted own weights (hash-checked); only explicitly excluded keys may differ."""
    from satchange.semantic.data import file_digest

    path = Path(spec["path"]).expanduser().resolve()
    if file_digest(path) != spec["sha256"]:
        raise ValueError(f"initial_weights SHA-256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"initial_weights has no state_dict: {path}")
    prefixes = tuple(spec["exclude"])
    own = model.state_dict()
    source = {k: v for k, v in state.items() if not k.startswith(prefixes)} if prefixes else state
    expected = {k for k in own if not (prefixes and k.startswith(prefixes))}
    if set(source) != expected:
        raise ValueError(
            "initial_weights keys differ beyond the excluded prefixes: "
            f"missing={sorted(expected - set(source))[:5]}, "
            f"unexpected={sorted(set(source) - expected)[:5]}"
        )
    for key, tensor in source.items():
        if tuple(tensor.shape) != tuple(own[key].shape):
            raise ValueError(f"initial_weights shape mismatch for {key}: "
                             f"{tuple(tensor.shape)} vs {tuple(own[key].shape)}")
    model.load_state_dict(source, strict=False)
    return {
        "path": str(path), "sha256": spec["sha256"], "loaded_keys": len(source),
        "excluded_keys": sorted(set(own) - expected),
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.copying")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
