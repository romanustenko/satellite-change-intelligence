"""Real HPO v3 controller/backend/engine on synthetic LEVIR-style CPU data (binary task).

Only the process pool and the GPU boundary are mocked; training, checkpoints, resume and the
final benchmark run for real on tiny images.
"""

import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from satchange import automated_hpo as hpo
from satchange import optimization
from satchange.training import load_training_checkpoint


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cpu_queue(monkeypatch):
    """Run the real v3 controller/backend/engine; replace only process and GPU boundaries."""
    from satchange import hpo_backend, hpo_core

    calls = []
    monkeypatch.setattr(hpo_backend, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        hpo_backend.SatelliteBackend,
        "resources",
        lambda self, devices: {
            device: {"free_mib": 16000, "total_mib": 16000, "name": "mock"} for device in devices
        },
    )

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 1
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def submit(self, operation, *args):
            calls.append({"job": args[1], "device": args[2]})
            future = Future()
            try:
                future.set_result(operation(*args))
            except Exception as error:
                future.set_exception(error)
            return future

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(hpo_core, "ProcessPoolExecutor", Executor)
    return calls


def make_dataset(root: Path, scenes: int = 2) -> Path:
    for split in ("train", "val", "test"):
        for index in range(scenes):
            for folder in ("A", "B", "label"):
                directory = root / split / folder
                directory.mkdir(parents=True, exist_ok=True)
                pixels = np.zeros((32, 32, 3), np.uint8)
                pixels[8 + index : 16 + index, 8:16] = 255 if folder != "A" else 20
                Image.fromarray(pixels).save(directory / f"scene_{index}.png")
    return root


def options(tmp_path, name="binary", epochs=1, **extra):
    base = dict(
        model_parameters={"base_channels": 4},
        seed=42,
        image_size=32,
        batch_size=1,
        num_workers=0,
        learning_rate=0.001,
        weight_decay=0.0001,
        scheduler="none",
        positive_weight=None,
        bce_weight=1.0,
        dice_weight=1.0,
        precision="fp32",
    )
    args = dict(
        stages=[("baseline", {"learning_rate": [0.001]})],
        base_parameters=base,
        model_name="fc_siam_diff",
        dataset_root=make_dataset(tmp_path / "levir"),
        thresholds=[0.4, 0.5],
        stage_epochs=[epochs, epochs, epochs],
        additional_seeds=False,
        output_root=tmp_path / "searches",
        search_name=name,
        gpu_ids=[0, 1, 2, 3],
        verbose=False,
        checkpoint_interval=1,
    )
    args.update(extra)
    return args


def search(**args):
    stage = args.pop("stage", None)
    if stage is None:
        hpo.run_automated_parameter_search(**args, stage="hpo")
        stage = "seeds"
    return hpo.run_automated_parameter_search(**args, stage=stage)


def best_checkpoint(summary):
    seed = summary["selected"]["seeds"][0]
    row = next(r for r in summary["selected"]["seed_results"] if r["parameters"]["seed"] == seed)
    return Path(row["best_checkpoint"])


def test_binary_controller_worker_schema_and_metric_regression(tmp_path, cpu_queue):
    from satchange.data import create_dataset_dataloader
    from satchange.losses import BCEDiceLoss
    from satchange.run import reconstruct_checkpoint
    from satchange.training import evaluate_thresholds

    args = options(tmp_path, gpu_ids=[0])
    args["stages"] = [("baseline", {"learning_rate": [0.001]}), ("next", {"weight_decay": [1e-3]})]
    search(**args, stage="baseline")
    with pytest.raises(RuntimeError, match="incomplete"):
        search(**args, stage="seeds")
    search(**args, stage="next")
    summary = search(**args, stage="seeds")
    family = summary["selected_by_mean_validation_f1"]
    row = family["seed_results"][0]
    assert "best_f1" in row and "best_iou" in row and "selection_score" not in row
    model, state = reconstruct_checkpoint(row["best_checkpoint"], torch.device("cpu"))
    assert "task" not in state and state["thresholds"] == [0.4, 0.5]
    _, loader = create_dataset_dataloader(args["dataset_root"], "val", 32, 1, 0)
    metrics = evaluate_thresholds(model, loader, BCEDiceLoss(), torch.device("cpu"), [0.4, 0.5])
    best = max(metrics["threshold_results"], key=lambda r: r["f1"])
    assert row["best_f1"] == best["f1"] and row["best_iou"] == best["iou"]


def test_queue_dispatch_over_four_devices_and_completed_cache(tmp_path, cpu_queue):
    args = options(tmp_path, f1_gap=1.0, iou_gap=1.0)
    args["stages"] = [("baseline", {"learning_rate": [0.001, 0.002, 0.003, 0.004, 0.005]})]
    summary = search(**args)
    assert len(cpu_queue) == 5
    assert {call["device"] for call in cpu_queue} == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}
    assert len(summary["ranking"]) == 5
    search(**args)
    assert len(cpu_queue) == 5  # Completed results never re-enter a worker.


def test_exact_epoch_resume_from_explicit_checkpoint(tmp_path, cpu_queue):
    full = search(**options(tmp_path, "full", epochs=2))
    one = search(**options(tmp_path, "one", epochs=1))
    initial = best_checkpoint(one).parent / "last.pt"
    resumed = search(**options(tmp_path, "continued", epochs=2), resume_from=initial)
    state = load_training_checkpoint(best_checkpoint(resumed).parent / "last.pt")
    reference = load_training_checkpoint(best_checkpoint(full).parent / "last.pt")
    assert state["epoch"] == 2
    for key, value in reference["model_state_dict"].items():
        torch.testing.assert_close(value, state["model_state_dict"][key], rtol=0, atol=0)
    for left, right in zip(reference["history"], state["history"], strict=True):
        for key in ("train_loss", "validation_loss", "validation_f1"):
            assert left[key] == right[key]


def test_failed_trial_resumes_own_last_and_preserves_best(tmp_path, cpu_queue, monkeypatch):
    from satchange import training

    args = options(tmp_path, "interrupt", epochs=3)
    original = training._save_checkpoint
    interrupted = False

    def stop_after_epoch(path=None, state=None, **kwargs):
        nonlocal interrupted
        path = path or kwargs["checkpoint_path"]
        state = state or kwargs["checkpoint_state"]
        original(path, state)
        if Path(path).name == "last.pt" and state["epoch"] == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("synthetic interruption after saved epoch")

    monkeypatch.setattr(training, "_save_checkpoint", stop_after_epoch)
    with pytest.raises(RuntimeError, match=r"trial\(s\) failed"):
        search(**args)
    monkeypatch.setattr(training, "_save_checkpoint", original)
    result = search(**args)
    row = json.loads(Path(result["selected"]["seed_results"][0]["summary_path"]).read_text())
    assert row["resumed_from_epoch"] == 1 and row["new_epochs_completed"] == 2
    assert Path(row["resume_source"]).name == "last.pt"
    assert best_checkpoint(result).is_file()


def test_early_stop_scheduler_and_checkpoint_parameter_validation(tmp_path, cpu_queue):
    from satchange.training import EarlyStopping

    args = options(tmp_path, epochs=4)
    args["base_parameters"]["scheduler"] = "reduce_on_plateau"
    summary = search(**args, patience=1, min_delta=1.0, seed_patience=1)
    row = summary["selected"]["seed_results"][0]
    assert row["stop_reason"] == "early_stopping" and row["epoch"] == 2
    checkpoint = Path(row["best_checkpoint"]).parent / "last.pt"
    state = load_training_checkpoint(checkpoint)
    assert state["scheduler_state_dict"] is not None
    assert state["early_stopping"]["policy"]["monitor"] == "validation_f1"
    check = dict(
        model_name=args["model_name"], dataset_root=args["dataset_root"],
        thresholds=args["thresholds"], budget=4,
        early_stopping=EarlyStopping(1, 1, 1.0, monitor="validation_f1"),
    )
    parameters = json.loads(json.dumps(args["base_parameters"]))
    parameters["model_parameters"]["base_channels"] = 8
    with pytest.raises(ValueError, match="parameters differ"):
        optimization.read_hpo_checkpoint(checkpoint, **check, parameters=parameters)
    parameters = json.loads(json.dumps(args["base_parameters"]))
    parameters.update(gpu_id=3, code_revision="another compatible host")
    optimization.read_hpo_checkpoint(checkpoint, **check, parameters=parameters)


def test_three_phase_promotion_final_benchmark_and_validation_gate(
    tmp_path, cpu_queue, monkeypatch
):
    """Real CPU epochs verify 1 -> 2 -> 3 targets, the final export and val-before-test."""
    from satchange import final_benchmark
    from satchange.final_benchmark import run_final_benchmark

    args = options(tmp_path, stage_epochs=[1, 2, 3], additional_seeds=True, patience=1)
    hpo.run_automated_parameter_search(**args, stage="hpo")
    seeds = hpo.run_automated_parameter_search(**args, stage="seeds")
    assert seeds["selected"]["seeds"] == [42, 123, 777]
    plan = hpo.run_automated_parameter_search(**args, stage="final", dry_run=True)
    assert len(plan["jobs"]) == 3
    assert all(job["source"]["epoch"] == 2 and job["target_epoch"] == 3 for job in plan["jobs"])
    final = hpo.run_automated_parameter_search(**args, stage="final")
    assert final["complete"] and all(row["epoch"] == 3 for row in final["results"])
    assert all(row["policy"] is None for row in final["results"])
    assert len(cpu_queue) == 7  # 1 HPO + 3 seeds + 3 final; no fourth seed.
    for row in final["results"]:
        last = load_training_checkpoint(row["last_checkpoint"])
        assert [r["epoch"] for r in last["history"]] == [1, 2, 3]
        assert row["new_epochs_completed"] == 1
        # Phase totals no longer double count copied epochs.
        assert row["epoch_compute_gpu_hours_this_attempt"] == 0  # CPU run: zero GPU-hours.
    accounting = final["gpu_accounting"]
    assert accounting["allocations"] == 7 and accounting["operation_allocated_gpu_hours"] >= 0
    hpo.run_automated_parameter_search(**args, stage="final")
    assert len(cpu_queue) == 7

    calls = []
    evaluate = final_benchmark._evaluate

    def observed(spec, loader, split, device, precision):
        calls.append(split)
        return evaluate(spec, loader, split, device, precision)

    monkeypatch.setattr(final_benchmark, "_evaluate", observed)
    benchmark = dict(
        variant="light_test",
        checkpoints=final["best_checkpoints"],
        project_root=tmp_path,
        dataset_root=args["dataset_root"],
        image_size=32,
        training_results=final,
        deployment_seed=42,
        device="cpu",
        warmup=0,
        repetitions=2,
    )
    report = run_final_benchmark(**benchmark, output_dir=tmp_path / "final_bundle")
    assert calls == ["validation"] * 3 + ["test"] * 3
    assert report["metadata"]["deployment_seed"] == 42
    out = Path(report["output_dir"])
    for path in ("metadata.json", "benchmark.json", "training_results.json",
                 "validation_results.json", "final_test_results.json", "deployment_weights.pt"):
        assert (out / path).is_file()
    state = torch.load(out / "deployment_weights.pt", map_location="cpu", weights_only=True)
    assert "optimizer_state_dict" not in state

    calls.clear()

    def changed_validation(spec, loader, split, device, precision):
        row = observed(spec, loader, split, device, precision)
        if split == "validation":
            row["validation_f1"] += 0.2
        return row

    monkeypatch.setattr(final_benchmark, "_evaluate", changed_validation)
    with pytest.raises(ValueError, match="No test evaluation"):
        run_final_benchmark(**benchmark, output_dir=tmp_path / "rejected_bundle")
    assert calls == ["validation"] and not (tmp_path / "rejected_bundle").exists()


def test_legacy_source_fingerprint_still_covers_package_files(monkeypatch):
    from satchange.hpo_backend import SatelliteBackend

    original = Path.read_bytes
    baseline = SatelliteBackend().code_signature()

    def changed(path):
        content = original(path)
        return content + b"\n# synthetic change" if path.name == "roi.py" else content

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert SatelliteBackend().code_signature() != baseline


def test_code_identity_is_model_specific_and_line_ending_independent(tmp_path, monkeypatch):
    from satchange.hpo_backend import SatelliteBackend

    backend = SatelliteBackend()
    light = backend.code_identity({"model_name": "fc_siam_diff"})
    medium = backend.code_identity({"model_name": "bit"})
    assert "models/fc_siam_diff.py" in light["files"] and "models/bit.py" not in light["files"]
    assert "models/bit.py" in medium["files"]
    assert not any(name.startswith("semantic/") for name in light["files"])
    assert light["categories"]["hpo_core.py"] == "control"
    assert light["categories"]["hpo_accounting.py"] == "monitoring"
    original = Path.read_bytes

    def flipped(path):
        content = original(path)
        normalized = content.replace(b"\r\n", b"\n")
        return normalized if b"\r\n" in content else normalized.replace(b"\n", b"\r\n")

    monkeypatch.setattr(Path, "read_bytes", flipped)
    assert backend.code_identity({"model_name": "fc_siam_diff"}) == light


def test_controller_lock_is_exclusive_and_released(tmp_path):
    from satchange.hpo_core import experiment_lock

    with experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another HPO"):
            with experiment_lock(tmp_path):
                pass
    with experiment_lock(tmp_path):
        pass
