"""HPO v3 code-compatibility (schema 4) and GPU-hour ledger tests with a fake backend and clock.

The real controller (hpo_core) runs unchanged; only the backend, the process pool and the
ledger clock are replaced, so no GPU, dataset or training is involved.
"""

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from satchange import automated_hpo, hpo_accounting, hpo_core
from satchange.hpo_accounting import GpuLedger

EPOCH_SECONDS = 900.0  # One fake epoch = 0.25 GPU-hours on one device.
CLOCK = [1_000_000.0]
FILES = {}
LEGACY = ["legacy-signature-v1"]
REAL_EPOCH_SLEEP = [0.0]
OOM_ONCE = set()


class FakeBackend:
    """Minimal Backend protocol implementation that writes tiny JSON 'checkpoints'."""

    def normalize(self, parameters):
        return dict(parameters)

    def merge(self, base, variation):
        return {**base, **variation}

    def validate(self, parameters, context):
        if "seed" not in parameters:
            raise ValueError("seed required")

    def code_signature(self):
        return LEGACY[0]

    def code_identity(self, context):
        return {
            "scheme": "satchange-code-v2",
            "files": dict(FILES),
            "categories": {
                "training.py": "training",
                "hpo_core.py": "control",
                "hpo_accounting.py": "monitoring",
            },
        }

    def resources(self, devices):
        return {d: {"free_mib": 1000.0, "total_mib": 1000.0, "name": "FakeGPU"} for d in devices}

    @staticmethod
    def _row(state_path, state):
        return {
            "parameters": state["parameters"],
            "epoch": state["epoch"],
            "policy": None,
            "stop_reason": "max_epochs" if state["complete"] else None,
            "status": "completed" if state["complete"] else "partial",
            "budget_epochs": state["target"],
            "run_directory": str(state_path.parent.resolve()),
            "run_name": state_path.parent.name,
            "last_checkpoint": str(state_path.resolve()),
            "best_checkpoint": str(state_path.resolve()),
            "best_epoch": state["epoch"],
            "metrics": {"f1": 0.5 + 0.01 * state["parameters"].get("lr", 0), "iou": 0.4},
        }

    def discover(self, roots, checkpoints, context):
        rows = []
        for root in roots:
            for path in sorted(Path(root).glob("runs/*/fake/*/state.json")):
                rows.append(self._row(path, json.loads(path.read_text(encoding="utf-8"))))
        return rows

    def train(self, job, device, stop_file):
        parameters = job["parameters"]
        if parameters.get("fail"):
            return {"status": "failed", "error": "synthetic failure"}
        if job["id"] in OOM_ONCE:
            OOM_ONCE.discard(job["id"])
            return {"status": "oom", "required_mib": 10}
        run_dir = Path(job["directory"]) / "fake" / f"seed_{parameters['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "state.json"
        source = job.get("source")
        if path.exists():
            epoch = json.loads(path.read_text(encoding="utf-8"))["epoch"]
        elif source:
            epoch = json.loads(Path(source["last_checkpoint"]).read_text(encoding="utf-8"))["epoch"]
        else:
            epoch = 0
        status = "completed"
        while epoch < job["target_epoch"]:
            time.sleep(REAL_EPOCH_SLEEP[0])
            CLOCK[0] += EPOCH_SECONDS
            epoch += 1
            if Path(stop_file).exists() and epoch < job["target_epoch"]:
                status = "paused"
                break
        state = {"parameters": parameters, "epoch": epoch, "target": job["target_epoch"],
                 "complete": status == "completed"}
        path.write_text(json.dumps(state), encoding="utf-8")
        return {**self._row(path, state), "status": status, "trial": job["id"]}


REFERENCE = f"{__name__}:FakeBackend"


class SyncExecutor:
    def __init__(self, **kwargs):
        assert kwargs["max_workers"] == 1

    def submit(self, operation, *args):
        future = Future()
        future.set_result(operation(*args))
        return future

    def shutdown(self, **kwargs):
        pass


class ThreadExecutor(ThreadPoolExecutor):
    def __init__(self, **kwargs):
        super().__init__(max_workers=kwargs["max_workers"])


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    CLOCK[0] = 1_000_000.0
    FILES.clear()
    FILES.update({"training.py": "t1", "hpo_core.py": "c1", "hpo_accounting.py": "m1"})
    LEGACY[0] = "legacy-signature-v1"
    REAL_EPOCH_SLEEP[0] = 0.0
    OOM_ONCE.clear()
    monkeypatch.setattr(hpo_accounting, "now", lambda: CLOCK[0])
    monkeypatch.setattr(hpo_core, "ProcessPoolExecutor", SyncExecutor)


def search(tmp_path, name="s", *, lrs=(1, 2), epochs=2, devices=1, **extra):
    return automated_hpo.run_automated_parameter_search(
        stages=[("lr", {"lr": list(lrs)})],
        base_parameters={"seed": 42, "lr": 0},
        model_name="fake",
        dataset_root=tmp_path / "data",
        thresholds=[0.5],
        output_root=tmp_path / "searches",
        search_name=name,
        stage="hpo",
        stage_epochs=(epochs, epochs, epochs),
        gpu_ids=list(range(devices)),
        verbose=False,
        backend=REFERENCE,
        **extra,
    )


def ledger_of(tmp_path, name="s"):
    return GpuLedger(tmp_path / "searches" / name)


# Ledger unit behaviour -------------------------------------------------------------------------


def test_ledger_counts_devices_once_and_survives_restart(tmp_path, monkeypatch):
    ledger = GpuLedger(tmp_path, heartbeat_seconds=60)
    one = ledger.start(devices=["cuda:0"], job_id="a")
    two = ledger.start(devices=["cuda:1", "cuda:2"], job_id="b")
    CLOCK[0] += 3600
    ledger.end(one, "completed", epochs_trained=4)
    ledger.end(one, "completed")  # Duplicate end is ignored.
    ledger.heartbeat([two])
    CLOCK[0] += 7200  # Controller dies; `two` keeps no end event.
    with (tmp_path / hpo_accounting.LEDGER_NAME).open("a", encoding="utf-8") as handle:
        handle.write('{"event": "end", "allocation_id": "torn')  # Torn final write.
    restarted = GpuLedger(tmp_path, heartbeat_seconds=60)
    assert restarted.invalid_lines == 1
    assert restarted.close_orphans() == [two]
    summary = restarted.summary()
    # one: 1 h x 1 device; two: charged to its last heartbeat, 1 h x 2 devices.
    assert summary["spent_gpu_hours"] == pytest.approx(3.0)
    assert summary["uncertain_seconds"] == 60
    assert restarted.seconds_per_epoch() == pytest.approx(900.0)
    assert GpuLedger(tmp_path).close_orphans() == []  # Replay never double counts.


# Controller accounting --------------------------------------------------------------------------


def test_four_devices_unallocated_device_and_per_trial_hours(tmp_path):
    report = search(tmp_path, lrs=(1, 2, 3, 4), devices=5)
    assert report["complete"]
    accounting = report["gpu_accounting"]
    rows = report["results"]
    # Allocations last from dispatch to OBSERVED completion (the synchronous fake executor
    # observes all four together); totals equal the sum of the per-trial allocations.
    assert all(row["allocated_gpu_hours"] >= 0.5 - 1e-9 for row in rows)
    assert accounting["operation_allocated_gpu_hours"] == pytest.approx(
        sum(row["allocated_gpu_hours"] for row in rows)
    )
    assert accounting["spent_gpu_hours"] == accounting["operation_allocated_gpu_hours"]
    assert {row["code_identity_id"] for row in rows} == {report["code_identity_id"]}
    events = [json.loads(line) for line in
              (tmp_path / "searches/s" / hpo_accounting.LEDGER_NAME).read_text().splitlines()]
    starts = [e for e in events if e["event"] == "start"]
    assert len(starts) == 4 and all(e["gpu_names"] == ["FakeGPU"] for e in starts)
    assert len({e["devices"][0] for e in starts}) == 4  # The fifth device is never charged.
    again = search(tmp_path, lrs=(1, 2, 3, 4), devices=5)  # Completed cache: no new allocation.
    assert again["gpu_accounting"]["allocations"] == 4


def test_budget_stops_dispatch_resume_keeps_spending_and_raise_is_recorded(tmp_path):
    report = search(tmp_path, lrs=(1, 2, 3), gpu_hour_budget=1.0)
    # Trial 1 (0.5 h, cost unknown beforehand), trial 2 fits exactly, trial 3 does not.
    assert not report["complete"] and report["paused"]
    assert report["gpu_accounting"]["budget_state"] == "exhausted"
    assert report["gpu_accounting"]["spent_gpu_hours"] == pytest.approx(1.0)
    assert report["gpu_accounting"]["overshoot_gpu_hours"] == 0
    assert len(report["results"]) == 2
    again = search(tmp_path, lrs=(1, 2, 3))  # None keeps the recorded budget.
    assert not again["complete"] and again["gpu_accounting"]["allocations"] == 2
    raised = search(tmp_path, lrs=(1, 2, 3), gpu_hour_budget=1.5, budget_reason="pilot extension")
    assert raised["complete"] and raised["gpu_accounting"]["spent_gpu_hours"] == pytest.approx(1.5)
    ledger = ledger_of(tmp_path)
    assert [b["gpu_hours"] for b in ledger.budgets] == [1.0, 1.5]
    assert ledger.budgets[-1]["reason"] == "pilot extension"


def test_mid_trial_exhaustion_pauses_at_epoch_boundary_and_records_overshoot(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(hpo_core, "ProcessPoolExecutor", ThreadExecutor)
    monkeypatch.setattr(hpo_core.signal, "signal", lambda *args: None)  # Worker runs in a thread.
    REAL_EPOCH_SLEEP[0] = 0.3
    report = search(tmp_path, lrs=(1,), epochs=8, gpu_hour_budget=0.6)
    assert report["paused"] and not report["complete"]
    assert report["gpu_accounting"]["budget_state"] == "exhausted_mid_run"
    assert (tmp_path / "searches/s/PAUSE_AFTER_EPOCH").exists()
    ledger = ledger_of(tmp_path)
    ends = [a["end"] for a in ledger.allocations.values()]
    assert [e["status"] for e in ends] == ["paused"] and ends[0]["epochs_trained"] < 8
    assert ledger.spent() > 0.6 and ledger.overshoots
    assert ledger.summary()["overshoot_gpu_hours"] == pytest.approx(ledger.spent() - 0.6)


def test_failed_and_oom_attempts_are_charged(tmp_path):
    plan = search(tmp_path, lrs=(1,), dry_run=True)
    OOM_ONCE.add(plan["jobs"][0]["id"])
    report = search(tmp_path, lrs=(1,), devices=2)
    assert report["complete"]
    row = report["results"][0]
    assert len(row["allocation_ids"]) == 2  # OOM attempt + successful retry.
    statuses = sorted(a["end"]["status"] for a in ledger_of(tmp_path).allocations.values())
    assert statuses == ["completed", "oom"]
    with pytest.raises(hpo_core.PipelineRunError):
        automated_hpo.run_automated_parameter_search(
            stages=[("lr", {"fail": [True]})],
            base_parameters={"seed": 42, "lr": 0},
            model_name="fake", dataset_root=tmp_path / "data", thresholds=[0.5],
            output_root=tmp_path / "searches", search_name="failing", stage="hpo",
            stage_epochs=(2, 2, 2), gpu_ids=[0], verbose=False, backend=REFERENCE,
        )
    failed = ledger_of(tmp_path, "failing")
    assert [a["end"]["status"] for a in failed.allocations.values()] == ["failed"]


def test_controller_crash_orphan_is_closed_once(tmp_path):
    root = tmp_path / "searches/s"
    search(tmp_path, lrs=(1,))
    ledger = GpuLedger(root)
    orphan = ledger.start(devices=["cuda:0"], job_id="ghost")
    CLOCK[0] += 600
    ledger.heartbeat([orphan])
    CLOCK[0] += 10_000  # Crash: no end event.
    report = search(tmp_path, lrs=(1,))
    assert report["gpu_accounting"]["uncertain_seconds"] == 60
    closed = GpuLedger(root).allocations[orphan]["end"]
    assert closed["status"] == "interrupted" and closed["seconds"] == pytest.approx(600)


def test_budget_parameter_validation(tmp_path):
    for value in (0, -1, float("inf"), True):
        with pytest.raises(ValueError, match="gpu_hour_budget"):
            search(tmp_path, gpu_hour_budget=value)
    with pytest.raises(ValueError, match="unverified_reuse"):
        search(tmp_path, unverified_reuse="sometimes")


# Code compatibility ------------------------------------------------------------------------------


def test_blocking_change_refuses_monitoring_change_is_recorded_ack_requires_evidence(tmp_path):
    root = tmp_path / "searches/s"
    search(tmp_path, lrs=(1,))
    history = json.loads((root / hpo_core.CODE_HISTORY).read_text())
    assert [h["kind"] for h in history] == ["initial"]
    FILES["hpo_accounting.py"] = "m2"
    search(tmp_path, lrs=(1,))  # Monitoring-only change: allowed and recorded.
    history = json.loads((root / hpo_core.CODE_HISTORY).read_text())
    assert history[-1]["kind"] == "monitoring_change"
    assert history[-1]["changed_files"] == ["hpo_accounting.py"]
    FILES["training.py"] = "t2"
    protocol = (root / "pipeline_protocol.json").read_bytes()
    with pytest.raises(hpo_core.CodeCompatibilityError, match="training.py"):
        search(tmp_path, lrs=(1,))
    for evidence in ([], "one string", [" "]):
        with pytest.raises(ValueError, match="evidence"):
            automated_hpo.acknowledge_code_change(root, reason="fix", evidence=evidence)
    with pytest.raises(ValueError, match="reason"):
        automated_hpo.acknowledge_code_change(root, reason=" ", evidence=["test"])
    entry = automated_hpo.acknowledge_code_change(
        root, reason="logging-only refactor", evidence=["tests/test_x.py::test_same_trajectory"]
    )
    assert entry["kind"] == "acknowledged_exception" and entry["blocking_files"] == ["training.py"]
    assert search(tmp_path, lrs=(1,))["complete"]
    assert (root / "pipeline_protocol.json").read_bytes() == protocol  # Never rewritten.
    again = automated_hpo.acknowledge_code_change(root, reason="noop", evidence=["none"])
    assert again["status"] == "already_compatible"


def test_definition_change_still_refuses(tmp_path):
    search(tmp_path, lrs=(1,))
    with pytest.raises(ValueError, match="definition changed"):
        search(tmp_path, lrs=(1, 2))


def test_schema3_protocol_needs_acknowledgement_and_stays_unmodified(tmp_path):
    root = tmp_path / "searches/s"
    search(tmp_path, lrs=(1,))
    path = root / "pipeline_protocol.json"
    legacy = json.loads(path.read_text())
    legacy.pop("code_identity")
    legacy.update(schema_version=3, backend_code_signature="signature-of-old-code")
    path.write_text(json.dumps(legacy), encoding="utf-8")
    (root / hpo_core.CODE_HISTORY).unlink()
    snapshot = path.read_bytes()
    with pytest.raises(hpo_core.CodeCompatibilityError, match="unknown"):
        search(tmp_path, lrs=(1,))
    LEGACY[0] = "signature-of-old-code"  # Same legacy signature = evidence of identical code.
    assert search(tmp_path, lrs=(1,))["complete"]
    LEGACY[0] = "legacy-signature-v2"
    automated_hpo.acknowledge_code_change(root, reason="migration", evidence=["review note"])
    history = json.loads((root / hpo_core.CODE_HISTORY).read_text())
    assert [h["kind"] for h in history] == ["origin", "acknowledged_exception"]
    assert history[0]["identity"]["signature"] == "signature-of-old-code"
    assert search(tmp_path, lrs=(1,))["complete"]
    assert path.read_bytes() == snapshot


def test_reuse_from_unverified_code_is_exploratory_only(tmp_path):
    old = tmp_path / "searches/old"
    search(tmp_path, "old", lrs=(1,), epochs=1)
    search(tmp_path, "old_full", lrs=(1,), epochs=2)
    FILES["training.py"] = "t2"
    # Completed result from different code: refused by default, labelled when exploratory.
    with pytest.raises(hpo_core.CodeCompatibilityError, match="not verified"):
        search(tmp_path, "new", lrs=(1,), epochs=2, reuse_from=tmp_path / "searches/old_full")
    report = search(tmp_path, "new2", lrs=(1,), epochs=2,
                    reuse_from=tmp_path / "searches/old_full", unverified_reuse="exploratory")
    assert report["complete"] and report["gpu_accounting"]["allocations"] == 0
    assert all(row["exploratory_only"] for row in report["results"])
    assert report["exploratory_only_job_ids"]
    # A PARTIAL run (epoch 1 < target 2) from unverified code is never continued.
    with pytest.raises(hpo_core.CodeCompatibilityError, match="Continuing partial runs"):
        search(tmp_path, "new3", lrs=(1,), epochs=2, reuse_from=old, unverified_reuse="exploratory")
    # After an evidence-backed exception in the source directory it is verified again.
    automated_hpo.acknowledge_code_change(old, reason="reviewed", evidence=["diff review"])
    resumed = search(tmp_path, "new4", lrs=(1,), epochs=2, reuse_from=old)
    assert resumed["complete"] and not resumed["exploratory_only_job_ids"]


def test_concurrent_controller_is_rejected(tmp_path):
    root = tmp_path / "searches/s"
    search(tmp_path, lrs=(1,))
    hold = threading.Event()
    release = threading.Event()

    def holder():
        with hpo_core.experiment_lock(root):
            hold.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    hold.wait(5)
    try:
        with pytest.raises(RuntimeError, match="Another HPO v3 controller"):
            automated_hpo.acknowledge_code_change(root, reason="x", evidence=["y"])
    finally:
        release.set()
        thread.join()
