"""Portable, validation-only experiment controller (standard library only).

A backend supplies parameter normalization, checkpoint discovery, device probing,
and training. No model, dataset, PyTorch, or satchange imports belong here.
Budgets are ABSOLUTE target epochs, never implicitly added on every restart.
"""

from __future__ import annotations

import importlib
import itertools
import json
import math
import multiprocessing as mp
import os
import signal
import socket
import statistics
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

Row = dict[str, Any]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def identity(value: Any) -> str:
    return sha256(canonical(value).encode()).hexdigest()[:20]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value: Any) -> None:
    """Commit metadata atomically on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def experiment_lock(root: Path) -> Iterator[None]:
    """Hold a nonblocking OS lock; kernel/process exit releases it on Windows and POSIX."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".hpo_v3.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)

            def acquire() -> None:
                return msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release() -> None:
                return msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")

            def acquire() -> None:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(handle, fcntl.LOCK_UN)

        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f"Another HPO v3 controller owns {root}") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{socket.gethostname()} pid={os.getpid()}\n".encode())
            handle.flush()
            yield
        finally:
            handle.seek(0)
            release()


class Backend(Protocol):
    """A backend class must have a no-argument constructor; use an import path."""

    def normalize(self, parameters: Mapping[str, Any]) -> Row: ...
    def merge(self, base: Row, variation: Row) -> Row: ...
    def validate(self, parameters: Row, context: Row) -> None: ...
    def code_signature(self) -> str: ...
    def discover(
        self, roots: Sequence[str], checkpoints: Sequence[str], context: Row
    ) -> list[Row]: ...
    def resources(self, devices: Sequence[str]) -> dict[str, Row]: ...
    def train(self, job: Row, device: str, stop_file: str) -> Row: ...


def load_backend(reference: str) -> Backend:
    module, name = reference.split(":", 1)
    return cast(Backend, getattr(importlib.import_module(module), name)())


def _worker(reference: str, job: Row, device: str, stop_file: str) -> Row:
    # The parent turns Ctrl+C into a cooperative epoch-boundary pause.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return load_backend(reference).train(job, device, stop_file)
    except Exception as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


class PipelineRunError(RuntimeError):
    def __init__(self, message: str, report_path: str, failed_trials: list[Row]):
        super().__init__(message)
        self.report_path = report_path
        self.failed_trials = failed_trials


class ParameterSearchMemoryError(PipelineRunError):
    """Raised AFTER runnable siblings finish and their results have been saved."""


@dataclass
class PipelineSettings:
    root: Path
    backend: str
    context: Row
    base_parameters: Row
    stages: list[tuple[str, Row]]
    stage_epochs: dict[str, int]
    policies: dict[str, Row | None]
    seeds: list[int]
    devices: list[str]
    gaps: dict[str, float] = field(default_factory=lambda: {"f1": 0.04, "iou": 0.05})
    metrics: tuple[str, ...] = ("f1", "iou")
    reuse_from: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    additional_seeds: bool = True
    checkpoint_interval: int = 10
    max_trial_memory_mib: float = 80_000
    retry_seconds: float = 30
    memory_margin_mib: float = 512
    max_oom_attempts: int | None = 12
    verbose: bool = True


def family(parameters: Row) -> Row:
    return {key: value for key, value in parameters.items() if key != "seed"}


def can_reuse(record: Row, target: int, policy: Row | None) -> bool:
    if record.get("status") != "completed":
        return False
    if record.get("epoch") == target and (record.get("policy") == policy or policy is None):
        # A run that physically reached the full target already satisfies a fixed budget.
        # Keep its original policy metadata; do not relabel or retrain it.
        return True
    return (
        record.get("stop_reason") == "early_stopping"
        and policy is not None
        and record.get("policy") == policy
        and record.get("budget_epochs") == target
    )


def summarize_families(
    records: list[Row], seeds: Sequence[int], metrics: Sequence[str]
) -> tuple[list[Row], list[Row]]:
    """Never compare a two-seed mean with a three-seed mean."""
    grouped: dict[str, list[Row]] = {}
    for row in records:
        grouped.setdefault(canonical(family(row["parameters"])), []).append(row)
    ranking, incomplete = [], []
    for key, rows in grouped.items():
        by_seed = {row["parameters"]["seed"]: row for row in rows}
        if len(by_seed) != len(rows):
            raise ValueError("Duplicate seed records in one configuration.")
        if set(by_seed) != set(seeds):
            incomplete.append(
                {
                    "parameters": json.loads(key),
                    "present_seeds": sorted(by_seed),
                    "missing_seeds": sorted(set(seeds) - set(by_seed)),
                }
            )
            continue
        ordered = [by_seed[seed] for seed in seeds]
        item = {
            "parameters": json.loads(key),
            "n_seeds": len(seeds),
            "seeds": list(seeds),
            "seed_results": ordered,
        }
        for metric in metrics:
            values = [float(row["metrics"][metric]) for row in ordered]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite {metric} in completed results.")
            item[f"mean_{metric}"] = statistics.mean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else None
        ranking.append(item)
    ranking.sort(
        key=lambda row: tuple(-row[f"mean_{m}"] for m in metrics) + (canonical(row["parameters"]),)
    )
    return ranking, incomplete


class Experiment:
    def __init__(self, settings: PipelineSettings):
        self.s = settings
        self.root = settings.root
        self.backend = load_backend(settings.backend)
        self.records: list[Row] = []

    def announce(self, message: str) -> None:
        if self.s.verbose:
            print(message, flush=True)

    def refresh(self) -> None:
        self.records = self.backend.discover(
            [*self.s.reuse_from, str(self.root)], self.s.checkpoints, self.s.context
        )

    def ensure_protocol(self) -> None:
        fixed = {
            "schema_version": 3,
            "backend": self.s.backend,
            "backend_code_signature": self.backend.code_signature(),
            "context": self.s.context,
            "base": self.s.base_parameters,
            "stages": self.s.stages,
            "seeds": self.s.seeds,
            "metrics": self.s.metrics,
            "gaps": self.s.gaps,
            "additional_seeds": self.s.additional_seeds,
            "reuse_from": self.s.reuse_from,
            "initial_checkpoints": self.s.checkpoints,
        }
        fixed = json.loads(canonical(fixed))
        path = self.root / "pipeline_protocol.json"
        if path.exists() and read_json(path) != fixed:
            raise ValueError(
                "Model/data/code/search definition changed. Use a new search_name; "
                "reuse_from can point to the old experiment."
            )
        if not path.exists():
            write_json(path, fixed)
        # Budgets are recorded per operation, so a larger target is a new explicit operation.

    def _matching(self, parameters: Row) -> list[Row]:
        key = canonical(parameters)
        return [row for row in self.records if canonical(row["parameters"]) == key]

    def _source(self, parameters: Row, target: int, policy: Row | None) -> tuple[Row | None, bool]:
        matches = [row for row in self._matching(parameters) if row["epoch"] <= target]
        reusable = [row for row in matches if can_reuse(row, target, policy)]
        if reusable:
            return sorted(reusable, key=lambda row: (-row["epoch"], row["run_directory"]))[0], True
        resumable = [row for row in matches if row.get("last_checkpoint")]
        if resumable:
            if all(row["epoch"] == target and row.get("policy") != policy for row in resumable):
                raise ValueError(
                    "Changing to a different early-stopping rule at an already-reached "
                    "target is not an exact resume. Use a larger target or a new fresh run."
                )
            return sorted(resumable, key=lambda row: (-row["epoch"], row["run_directory"]))[
                0
            ], False
        if matches:
            raise RuntimeError(
                "A matching run exists, but its last.pt is missing. Refusing a silent "
                f"restart/best.pt fallback for seed={parameters['seed']}. "
                "Supply a resume checkpoint explicitly or restore last.pt."
            )
        return None, False

    def _candidates(self, parents: list[Row], space: Row, controls: bool) -> list[Row]:
        if not space or any(
            not isinstance(values, (list, tuple)) or not values for values in space.values()
        ):
            raise ValueError("Every search parameter needs a nonempty candidate list.")
        unique = {}
        variations = [
            dict(zip(space, values, strict=False)) for values in itertools.product(*space.values())
        ]
        for parent in parents:
            defaults: list[Row] = [{}] if controls else []
            for variation in defaults + variations:
                candidate = self.backend.merge(parent, variation)
                self.backend.validate(candidate, self.s.context)
                unique[canonical(candidate)] = candidate
        return list(unique.values())

    def _operation(
        self,
        phase: str,
        label: str,
        candidates: list[Row],
        target: int,
        policy: Row | None,
        pending_policy: str,
        *,
        dry_run: bool,
        require_existing: bool = False,
    ) -> Row:
        operation_id = identity(
            {
                "phase": phase,
                "label": label,
                "candidates": candidates,
                "target": target,
                "policy": policy,
                "pending_policy": pending_policy,
            }
        )
        directory = self.root / "operations" / f"{label}_e{target}_{operation_id[:8]}"
        plan_path = directory / "plan.json"
        if plan_path.exists():
            plan = read_json(plan_path)
        else:
            retained, skipped = [], []
            started_families = set()
            if phase == "seeds" and pending_policy == "finish_started":
                secondary = set(self.s.seeds) - {self.s.base_parameters["seed"]}
                started_families = {
                    canonical(family(row["parameters"]))
                    for row in self.records
                    if row["parameters"]["seed"] in secondary
                }
            for parameters in candidates:
                matches = self._matching(parameters)
                started = bool(matches)
                if phase == "seeds" and pending_policy == "finish_started":
                    started = started and canonical(family(parameters)) in started_families
                if pending_policy == "finish_started" and not started:
                    skipped.append(parameters)
                else:
                    retained.append(parameters)
            jobs = []
            for parameters in retained:
                source, reused = self._source(parameters, target, policy)
                task_id = identity({"operation": operation_id, "parameters": parameters})
                jobs.append(
                    {
                        "id": task_id,
                        "parameters": parameters,
                        "target_epoch": target,
                        "policy": policy,
                        "context": self.s.context,
                        "phase": phase,
                        "directory": str(self.root / "runs" / task_id),
                        "checkpoint_interval": self.s.checkpoint_interval,
                        "source": source,
                        "reused": reused,
                    }
                )
            plan = {
                "phase": phase,
                "label": label,
                "target_epoch": target,
                "policy": policy,
                "pending_policy": pending_policy,
                "planned_candidates": len(candidates),
                "jobs": jobs,
                "skipped_unstarted": skipped,
                "selection_scope": "all_candidates" if not skipped else "started_subset_only",
                "test_used": False,
            }
            if require_existing and any(not job["reused"] for job in jobs):
                raise RuntimeError(
                    f"Earlier HPO block '{label}' is incomplete. Run it explicitly first."
                )
            write_json(plan_path, plan)
        if require_existing:
            # A prior dry-run plan may still contain jobs marked non-reused.
            ready = all(self._source(job["parameters"], target, policy)[1] for job in plan["jobs"])
            if not ready:
                raise RuntimeError(
                    f"Earlier HPO block '{label}' is incomplete. Run it explicitly first."
                )
            for job in plan["jobs"]:
                job["source"], job["reused"] = self._source(job["parameters"], target, policy)
        self.announce(
            f"{label}: target={target}, retained={len(plan['jobs'])}, "
            f"skipped={len(plan['skipped_unstarted'])}; plan={plan_path}"
        )
        if dry_run and not require_existing:
            return {
                "complete": False,
                "dry_run": True,
                "plan_path": str(plan_path),
                "retained_trials": len(plan["jobs"]),
                "skipped_trials": len(plan["skipped_unstarted"]),
                "jobs": plan["jobs"],
                "selection_scope": plan["selection_scope"],
            }
        report = self._execute(plan, directory)
        report["plan_path"] = str(plan_path)
        return report

    def _execute(self, plan: Row, directory: Path) -> Row:
        report_path = directory / "results.json"
        previous = read_json(report_path) if report_path.exists() else {}
        results = {row["job_id"]: row for row in previous.get("results", [])}
        for row in results.values():
            if not Path(row["best_checkpoint"]).is_file():
                raise FileNotFoundError(
                    f"Saved completed result lost its checkpoint: {row['best_checkpoint']}"
                )
        # An explicit rerun retries failed jobs (e.g. a repaired disk or dead worker).
        # Preserve their diagnostics, while keeping all completed siblings untouched.
        failure_history = previous.get("failure_history", []) + previous.get("failed_trials", [])
        failures: dict[str, Row] = {}
        pending = []
        for job in plan["jobs"]:
            if job["id"] in results or job["id"] in failures:
                continue
            if job["reused"]:
                results[job["id"]] = dict(job["source"], job_id=job["id"], reused=True)
            else:
                pending.append(
                    {"job": job, "attempts": [], "blocked": set(), "ready": 0.0, "need": 0.0}
                )
        paused = False
        active: dict[Any, tuple[str, Row]] = {}
        executors: dict[str, ProcessPoolExecutor] = {}
        stop = self.root / "PAUSE_AFTER_EPOCH"
        drain = self.root / "STOP_AFTER_ACTIVE_TRIALS"

        def save() -> Row:
            report = {
                "complete": not pending and not active and not failures and not paused,
                "paused": paused,
                "phase": plan["phase"],
                "label": plan["label"],
                "target_epoch": plan["target_epoch"],
                "policy": plan["policy"],
                "selection_scope": plan["selection_scope"],
                "skipped_unstarted": plan["skipped_unstarted"],
                "results": list(results.values()),
                "failed_trials": list(failures.values()),
                "failure_history": failure_history,
                "unfinished_job_ids": [s["job"]["id"] for s in pending]
                + [s["job"]["id"] for _, s in active.values()],
                "test_used": False,
                "report_path": str(report_path),
            }
            write_json(report_path, report)
            return report

        save()
        resource = self.backend.resources(self.s.devices) if pending else {}
        largest = max((r.get("total_mib", 0) for r in resource.values()), default=0)
        limit = (
            min(self.s.max_trial_memory_mib, largest) if largest else self.s.max_trial_memory_mib
        )
        context = mp.get_context("spawn")
        last_wait = 0.0
        try:
            while pending or active:
                try:
                    if stop.exists() or drain.exists():
                        paused = True
                    if not paused:
                        idle = set(self.s.devices) - {device for device, _ in active.values()}
                        resource = self.backend.resources(sorted(idle)) if idle else {}
                        ordered = sorted(
                            idle,
                            key=lambda d: resource[d].get("free_mib", float("inf")),
                            reverse=True,
                        )
                        for device in ordered:
                            available = resource[device].get("free_mib", float("inf"))
                            total = resource[device].get("total_mib", float("inf"))
                            chosen = next(
                                (
                                    i
                                    for i, s in enumerate(pending)
                                    if s["ready"] <= time.monotonic()
                                    and device not in s["blocked"]
                                    and available
                                    >= min(s["need"], max(0, total - self.s.memory_margin_mib))
                                ),
                                None,
                            )
                            if chosen is None:
                                continue
                            state = pending.pop(chosen)
                            if device not in executors:
                                executors[device] = ProcessPoolExecutor(
                                    max_workers=1, mp_context=context
                                )
                            state["attempts"].append(
                                {
                                    "device": device,
                                    "free_mib": available if math.isfinite(available) else None,
                                }
                            )
                            job = state["job"]
                            self.announce(
                                f"[RUN] {job['id']} seed={job['parameters']['seed']} "
                                f"-> {device}, target={job['target_epoch']}"
                            )
                            future = executors[device].submit(
                                _worker, self.s.backend, job, device, str(stop)
                            )
                            active[future] = device, state
                    if not active:
                        if paused:
                            break
                        if pending and time.monotonic() - last_wait >= self.s.retry_seconds:
                            self.announce(
                                "[WAIT] Waiting for an allowed device with enough free memory."
                            )
                            last_wait = time.monotonic()
                        time.sleep(min(self.s.retry_seconds, 1.0))
                        continue
                    done, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
                    for future in done:
                        device, state = active.pop(future)
                        job = state["job"]
                        try:
                            result = future.result()
                        except Exception as error:
                            result = {"status": "failed", "error": f"Worker failed: {error}"}
                        result.update(job_id=job["id"], attempts=state["attempts"])
                        status = result.get("status")
                        if status == "completed":
                            results[job["id"]] = result
                        elif status == "paused":
                            paused = True
                            pending.append(state)
                        elif status == "oom":
                            # Termination releases the failed worker's CUDA context.
                            executors.pop(device).shutdown(wait=True)
                            required = result.get("required_mib")
                            exhausted = (
                                self.s.max_oom_attempts is not None
                                and len(state["attempts"]) >= self.s.max_oom_attempts
                            )
                            if required is not None and required > limit:
                                result["status"] = "failed_cuda_oom_capacity"
                                result["limit_mib"] = limit
                                failures[job["id"]] = result
                            elif exhausted:
                                result["status"] = "failed_cuda_oom_retries"
                                failures[job["id"]] = result
                            else:
                                state["blocked"].add(device)
                                state["need"] = (required or 0) + self.s.memory_margin_mib
                                if state["blocked"] == set(self.s.devices):
                                    state["blocked"].clear()
                                    state["ready"] = time.monotonic() + self.s.retry_seconds
                                pending.append(state)
                                self.announce(
                                    f"[OOM -> RETRY] {job['id']}; keep last.pt, try another device."
                                )
                        else:
                            failures[job["id"]] = result
                        save()
                except KeyboardInterrupt:
                    stop.touch()
                    paused = True
                    self.announce(
                        "[PAUSE] No new trials. Waiting for active trials "
                        "to save the current epoch."
                    )
            report = save()
        finally:
            for executor in executors.values():
                executor.shutdown(wait=True, cancel_futures=True)
        if failures:
            error_class = (
                ParameterSearchMemoryError
                if any(
                    row.get("status", "").startswith("failed_cuda_oom") for row in failures.values()
                )
                else PipelineRunError
            )
            raise error_class(
                f"{len(failures)} trial(s) failed. Successful results saved: {report_path}",
                str(report_path),
                list(failures.values()),
            )
        return report

    def _select(self, report: Row, label: str) -> Row:
        if not report.get("complete"):
            return report
        rows = report["results"]
        if not rows:
            raise RuntimeError(f"No completed candidates in {label}; inspect the saved plan.")
        maxima = {metric: max(row["metrics"][metric] for row in rows) for metric in self.s.metrics}
        survivors = [
            row
            for row in rows
            if all(
                maxima[m] - row["metrics"][m] <= self.s.gaps.get(m, 0) + 1e-12
                for m in self.s.metrics
            )
        ]
        selected = {**report, "survivors": survivors, "selection_split": "val"}
        path = self.root / f"{label}_selection.json"
        selected["selection_report"] = str(path)
        write_json(path, selected)
        if not survivors:
            raise RuntimeError(f"No candidate meets all gaps: {path}")
        return selected

    def _hpo(
        self, stop_at: str | None, pending_policy: str, dry_run: bool, execute: bool, target: int
    ) -> Row:
        parents = [self.s.base_parameters]
        result = {}
        names = [name for name, _ in self.s.stages]
        if stop_at is not None and stop_at not in names:
            raise ValueError(f"Unknown hpo_stage={stop_at!r}; expected {names}")
        for i, (name, space) in enumerate(self.s.stages):
            candidates = self._candidates(parents, space, controls=i > 0)
            current = execute and (stop_at is None or name == stop_at)
            result = self._operation(
                "hpo",
                name,
                candidates,
                target,
                self.s.policies["hpo"],
                pending_policy if current else "all",
                dry_run=dry_run and current,
                require_existing=not current,
            )
            result = self._select(result, name)
            if not result.get("complete"):
                return result
            parents = [row["parameters"] for row in result["survivors"]]
            if name == stop_at:
                break
        return result

    def run(
        self,
        stage: str,
        *,
        hpo_stage: str | None = None,
        pending_policy: str = "all",
        dry_run: bool = False,
        additional_epochs: int = 0,
        extension_id: str | None = None,
        final_parameters: Row | None = None,
    ) -> Row:
        self.ensure_protocol()
        target = self.s.stage_epochs[stage]
        if additional_epochs:
            if (
                not extension_id
                or Path(extension_id).name != extension_id
                or extension_id in (".", "..")
            ):
                raise ValueError(
                    "additional_epochs requires a simple extension_id (idempotency key)."
                )
            path = self.root / "extensions" / f"{stage}_{extension_id}.json"
            contract = {
                "stage": stage,
                "hpo_stage": hpo_stage,
                "base_target": target,
                "additional_epochs": additional_epochs,
                "target_epoch": target + additional_epochs,
            }
            if path.exists() and read_json(path) != contract:
                raise ValueError(
                    "This extension_id already has a different target; use a new extension_id."
                )
            write_json(path, contract)
            target += additional_epochs
        budget_path = self.root / "active_phase_targets.json"
        active_targets = read_json(budget_path) if budget_path.exists() else {}
        if target < active_targets.get(stage, target):
            raise ValueError(
                f"{stage} already targets {active_targets[stage]} epochs; requested {target}. "
                "Keep the extension arguments or update stage_epochs; "
                "targets never shrink silently."
            )
        for previous_phase in ("hpo", "seeds", "final"):
            if previous_phase == stage:
                break
            if active_targets.get(previous_phase, 0) > self.s.stage_epochs[previous_phase]:
                raise ValueError(
                    f"Update stage_epochs[{previous_phase!r}] to its extended target "
                    f"{active_targets[previous_phase]} before using its selection."
                )
        active_targets[stage] = target
        write_json(budget_path, active_targets)
        self.refresh()
        for control in ("PAUSE_AFTER_EPOCH", "STOP_AFTER_ACTIVE_TRIALS"):
            (self.root / control).unlink(missing_ok=True)
        if stage == "hpo":
            return self._hpo(hpo_stage, pending_policy, dry_run, True, target)
        if stage == "seeds":
            hpo = self._hpo(None, "all", False, False, self.s.stage_epochs["hpo"])
            seeds = self.s.seeds if self.s.additional_seeds else [self.s.base_parameters["seed"]]
            parents = [row["parameters"] for row in hpo["survivors"]]
            candidates = self._candidates(parents, {"seed": seeds}, controls=False)
            report = self._operation(
                stage,
                stage,
                candidates,
                target,
                self.s.policies[stage],
                pending_policy,
                dry_run=dry_run,
            )
            if not report.get("complete"):
                return report
            ranking, incomplete = summarize_families(report["results"], seeds, self.s.metrics)
            selected = ranking[0] if ranking else None
            report.update(
                ranking=ranking, incomplete_families=incomplete, seeds=seeds, selected=selected
            )
            report[f"selected_by_mean_validation_{self.s.metrics[0]}"] = selected
            path = self.root / "seed_validation_summary.json"
            report["selection_report"] = str(path)
            write_json(path, report)
            if not ranking:
                raise RuntimeError(
                    "No configuration has the complete required seed set. Partial families "
                    "are saved, but cannot win. Use pending_policy='all' "
                    "or choose final_parameters explicitly."
                )
            return report

        # Freeze a SINGLE configuration before final training; never select by test.
        choice_path = self.root / "final_choice.json"
        if choice_path.exists():
            choice = read_json(choice_path)
            parameters = choice["parameters"]
            if (
                final_parameters is not None
                and family(self.backend.normalize(final_parameters)) != parameters
            ):
                raise ValueError(
                    "Final configuration already frozen. Use a new search_name to change it."
                )
        else:
            if final_parameters is not None:
                parameters = family(self.backend.normalize(final_parameters))
                source = "explicit_parameters_before_test"
            elif self.s.additional_seeds:
                path = self.root / "seed_validation_summary.json"
                if not path.exists():
                    raise RuntimeError(
                        "Run stage='seeds' first, or provide explicit final_parameters."
                    )
                report = read_json(path)
                if (
                    not report.get("complete")
                    or report.get("seeds") != self.s.seeds
                    or report.get("target_epoch", 0) < self.s.stage_epochs["seeds"]
                ):
                    raise RuntimeError("Final selection requires a completed three-seed report.")
                selected = report.get("selected") or report.get("selected_by_mean_validation_f1")
                if selected is None:
                    raise RuntimeError("Seed stage has no complete family to select.")
                parameters = selected["parameters"]
                source = str(path)
            else:
                path = self.root / "seed_validation_summary.json"
                exploratory = read_json(path) if path.exists() else {}
                selected = exploratory.get("selected")
                if (
                    exploratory.get("complete")
                    and selected is not None
                    and exploratory.get("seeds") == [self.s.base_parameters["seed"]]
                ):
                    parameters = selected["parameters"]
                    source = "single_seed_exploratory_selection: " + str(path)
                else:
                    hpo = self._hpo(None, "all", False, False, self.s.stage_epochs["hpo"])
                    best = max(
                        hpo["results"],
                        key=lambda row: tuple(row["metrics"][m] for m in self.s.metrics),
                    )
                    parameters = family(best["parameters"])
                    source = "single_hpo_seed_exploratory_selection"
            choice = {"parameters": parameters, "selection_source": source, "test_used": False}
            write_json(choice_path, choice)
        candidates = [self.backend.merge(parameters, {"seed": seed}) for seed in self.s.seeds]
        for candidate in candidates:
            self.backend.validate(candidate, self.s.context)
        # Never let a queue-pruning option remove a mandatory final seed.
        report = self._operation(
            "final", "final", candidates, target, self.s.policies["final"], "all", dry_run=dry_run
        )
        if not report.get("complete"):
            return report
        ranking, incomplete = summarize_families(report["results"], self.s.seeds, self.s.metrics)
        if incomplete or len(ranking) != 1:
            raise RuntimeError("Final training did not produce the full seed set.")
        if any(row["epoch"] != target for row in report["results"]):
            raise RuntimeError("Final training must reach the same absolute target on every seed.")
        report.update(
            seeds=self.s.seeds,
            ranking=ranking,
            selected=ranking[0],
            best_checkpoints={
                str(row["parameters"]["seed"]): row["best_checkpoint"] for row in report["results"]
            },
            choice=choice,
        )
        report[f"selected_by_mean_validation_{self.s.metrics[0]}"] = ranking[0]
        write_json(self.root / "final_training_summary.json", report)
        return report
