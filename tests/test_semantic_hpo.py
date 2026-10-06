"""Actual controller/worker/engine tests on synthetic CPU data; GPU queue boundary is mocked."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_semantic import config, make_manifest, scene_tensors

from satchange import automated_hpo as hpo
from satchange import optimization
from satchange.execution import resolve_device
from satchange.models.factory import MODEL_REGISTRY, SEMANTIC_TASK
from satchange.semantic.data import data_protocol, prepare_dataset, read_manifest
from satchange.semantic.inference import (
    detector_probability_map,
    evaluate_classifier,
    load_classifier,
    predict_checkpoint,
)
from satchange.semantic.second import (
    SECOND_CLASSES,
    SECOND_COLORS,
    bootstrap_second,
    decode_label,
    prepare_second,
)
from satchange.semantic.training import hpo_parameters
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
    from concurrent.futures import Future

    from satchange import hpo_backend, hpo_core

    calls = []
    monkeypatch.setattr(hpo_backend, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        hpo_backend.SatelliteBackend,
        "resources",
        lambda self, devices: {
            device: {"free_mib": 16000, "total_mib": 16000} for device in devices
        },
    )

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 1
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def submit(self, operation, *args):
            calls.append({"job": args[1], "device": args[2], "gpu_ids": [0, 1, 2, 3]})
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


def search(**args):
    """Exercise the explicit v3 phases; attach only convenience fields used by these tests."""
    stage = args.pop("stage", None)
    if stage is None:
        hpo.run_automated_parameter_search(**args, stage="hpo")
        stage = "seeds"
    result = hpo.run_automated_parameter_search(**args, stage=stage)
    if "ranking" in result:
        for item in result["ranking"]:
            seed = args["base_parameters"]["seed"]
            item["deployment_seed"] = seed
            item["deployment_best_checkpoint"] = next(
                row["best_checkpoint"]
                for row in item["seed_results"]
                if row["parameters"]["seed"] == seed
            )
        if args.get("task") == SEMANTIC_TASK:
            result["selected_by_mean_selection_score"] = result["selected"]
    return result


def options(tmp_path, name="search", epochs=2):
    manifest, _ = make_manifest(tmp_path)
    prepared = tmp_path / "prepared/prepared.json"
    if not prepared.exists():
        prepare_dataset(config(), manifest, prepared.parent)
    return dict(
        stages=[("baseline", {"learning_rate": [0.001]})],
        base_parameters=hpo_parameters(config(), prepared),
        task=SEMANTIC_TASK,
        model_name="tiny_siamese_change_classifier",
        dataset_root=prepared,
        stage_epochs=[epochs, epochs, epochs],
        additional_seeds=False,
        output_root=tmp_path / "searches",
        search_name=name,
        gpu_ids=[0, 1, 2, 3],
        verbose=False,
        checkpoint_interval=1,
    )


def test_actual_hpo_inference_cache_and_exact_epoch_resume(tmp_path, cpu_queue):
    args = options(tmp_path)
    first = search(**args)
    chosen = first["selected_by_mean_selection_score"]
    assert chosen["n_seeds"] == 1 and chosen["std_selection_score"] is None
    assert "mean_iou" not in chosen and "best_iou" not in chosen["seed_results"][0]
    path = Path(chosen["deployment_best_checkpoint"])
    model, best = load_classifier(path)
    history = best["history"]
    score = max(row["selection_score"] for row in history)
    assert best["epoch"] == next(r["epoch"] for r in history if r["selection_score"] == score)
    selected = chosen["seed_results"][0]
    assert selected["selected_metrics"]["joint_accuracy"] == best["best_metrics"]["joint_accuracy"]
    assert best["task"] == SEMANTIC_TASK and not best.get("thresholds")
    assert "tiny_siamese_change_classifier" not in MODEL_REGISTRY
    assert evaluate_classifier(path, args["dataset_root"], "val")["selection_score"] == score
    before, after, probabilities = scene_tensors(17, 23)
    assert predict_checkpoint(path, before, after, probabilities)["predictions"] == []
    probabilities[0, 2:5, 3:6] = 0.7
    result = predict_checkpoint(
        path,
        before,
        after,
        probabilities,
        inference_mask_source="predicted",
        detector={"identity": "synthetic frozen detector"},
    )
    assert result["training_mask_source"] == "ground_truth"
    assert result["inference_mask_source"] == "predicted" and len(result["predictions"]) == 1
    assert result["inference_precision"] == "fp32"
    with pytest.raises(ValueError, match="bf16"):
        predict_checkpoint(
            path,
            before,
            after,
            probabilities,
            precision="bf16",
            inference_mask_source="predicted",
            detector={"identity": "test"},
        )
    count = len(cpu_queue)
    search(**args)
    assert len(cpu_queue) == count  # Completed cache does not re-enter a worker.

    one = search(**options(tmp_path, "one", 1))
    initial = Path(one["ranking"][0]["deployment_best_checkpoint"]).parent / "last.pt"
    resumed = search(**options(tmp_path, "continued", 2), resume_from=initial)
    state = load_training_checkpoint(
        Path(resumed["ranking"][0]["deployment_best_checkpoint"]).parent / "last.pt"
    )
    full = load_training_checkpoint(path.parent / "last.pt")
    assert state["epoch"] == 2 and state["best_metrics"]["epoch"] == full["best_metrics"]["epoch"]
    for key, value in full["model_state_dict"].items():
        torch.testing.assert_close(value, state["model_state_dict"][key], rtol=0, atol=0)
    for left, right in zip(full["history"], state["history"], strict=True):
        for key in (
            "train_loss",
            "validation_loss",
            "selection_score",
            "before_confusion",
            "after_confusion",
        ):
            assert left[key] == right[key]
    assert cpu_queue[-1]["gpu_ids"] == [0, 1, 2, 3]


def test_failed_trial_resumes_own_last_and_preserves_best(tmp_path, cpu_queue, monkeypatch):
    args = options(tmp_path, "interrupt", 3)
    save = optimization._save_checkpoint
    from satchange import training

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
    row = json.loads(Path(result["ranking"][0]["seed_results"][0]["summary_path"]).read_text())
    assert row["resumed_from_epoch"] == 1 and row["new_epochs_completed"] == 2
    assert Path(row["resume_source"]).name == "last.pt"
    assert Path(result["ranking"][0]["deployment_best_checkpoint"]).is_file()
    assert save is optimization._save_checkpoint


def test_semantic_parameters_stages_seed_families_and_protocol_guard(tmp_path, cpu_queue):
    args = options(tmp_path, epochs=1)
    args["stages"] = [
        ("baseline", {"learning_rate": [0.001]}),
        ("dropout", {"model__dropout": [0.0, 0.2]}),
    ]
    search(**args, stage="baseline")
    with pytest.raises(RuntimeError, match="incomplete"):
        search(**args, stage="seeds")
    search(**args, stage="dropout")
    summary = search(**args, stage="seeds")
    for row in summary["ranking"]:
        params = row["parameters"]
        assert (
            "semantic_config" in params and "prepared_identity" in params and "seed" not in params
        )
    from satchange.hpo_backend import SatelliteBackend
    from satchange.hpo_core import identity as trial_identity

    identity = SatelliteBackend().normalize(args["base_parameters"])
    altered = deepcopy(identity)
    altered["model_parameters"]["dropout"] = 0.4
    assert trial_identity(identity) != trial_identity(altered)
    altered = deepcopy(identity)
    altered["semantic_config"]["roi"]["context_pixels"] = 2
    assert trial_identity(identity) != trial_identity(altered)
    protocol = args["output_root"] / "search" / "pipeline_protocol.json"
    previous = protocol.read_bytes()
    broken = json.loads(previous)
    broken["source_sha256"] = "unverified old Heavy source"
    protocol.write_text(json.dumps(broken), encoding="utf-8")
    snapshot = protocol.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        search(**args)
    assert protocol.read_bytes() == snapshot


def test_rgb_pair_leak_reversal_copy_and_blank_masks(tmp_path):
    manifest, scenes = make_manifest(tmp_path)
    scenes[-1]["before"], scenes[-1]["after"] = scenes[0]["after"], scenes[0]["before"]
    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
    with pytest.raises(ValueError, match="RGB pair leakage"):
        read_manifest(manifest, data_protocol(config()))
    for key in ("before", "after"):
        source = tmp_path / scenes[-1][key]
        copy = tmp_path / f"copied_{key}.png"
        copy.write_bytes(source.read_bytes())
        scenes[-1][key] = copy.name
    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
    with pytest.raises(ValueError, match="RGB pair leakage"):
        read_manifest(manifest, data_protocol(config()))


def second_fixture(root, count=5):
    for i in range(count):
        for folder in ("im1", "im2", "label1", "label2"):
            target = root / "train" / folder
            target.mkdir(parents=True, exist_ok=True)
            if folder.startswith("im"):
                array = np.full((512, 512, 3), 20 + i + (50 if folder == "im2" else 0), np.uint8)
            else:
                array = np.full((512, 512, 3), 255, np.uint8)
                array[2:5, 3:7] = (0, 255, 0)  # Same-class tree change remains changed.
            Image.fromarray(array).save(target / f"{i:03d}.png")


def second_config():
    cfg = config()
    cfg["classes"] = list(SECOND_CLASSES)
    cfg["class_ids"] = {i + 1: name for i, name in enumerate(SECOND_CLASSES)}
    cfg["ignore_ids"] = [0]
    return cfg


def test_second_exact_palette_ids_support_stable_cache(tmp_path):
    colors = np.asarray(list(SECOND_COLORS), dtype=np.uint8).reshape(1, 7, 3)
    path = tmp_path / "palette.png"
    Image.fromarray(colors).save(path)
    np.testing.assert_array_equal(
        decode_label(path), np.asarray(list(SECOND_COLORS.values())).reshape(1, 7)
    )
    Image.fromarray(colors).convert("P", palette=Image.Palette.ADAPTIVE).save(path)
    np.testing.assert_array_equal(
        decode_label(path), np.asarray(list(SECOND_COLORS.values())).reshape(1, 7)
    )
    Image.fromarray(np.array([[1, 2]], dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="explicit"):
        decode_label(path)
    np.testing.assert_array_equal(decode_label(path, {1: "water", 2: "tree"}), [[4, 2]])
    Image.fromarray(np.zeros((1, 1, 3), np.uint8)).save(path)
    with pytest.raises(ValueError, match="Unknown SECOND colors"):
        decode_label(path)
    root = tmp_path / "SECOND"
    second_fixture(root)
    assert bootstrap_second(root) == root
    with pytest.raises(ValueError, match="local_holdout"):
        prepare_second(root, tmp_path / "cache", second_config())
    prepared = prepare_second(root, tmp_path / "cache", second_config(), local_holdout=True)
    doc = json.loads(prepared.read_text())
    assert doc["second"]["local_splits"] == ["val", "test"]
    assert not doc["second"]["geographic_independence_verified"]
    assert all(r["before_label"] == r["after_label"] == 1 for r in doc["records"])
    assert prepare_second(root, tmp_path / "cache", second_config(), local_holdout=True) == prepared
    other = second_config()
    other["roi"]["context_pixels"] = 2
    assert prepare_second(root, tmp_path / "cache", other, local_holdout=True) != prepared
    label = root / "train/label2/000.png"
    Image.fromarray(np.full((512, 512, 3), 255, np.uint8)).save(label)
    with pytest.raises(ValueError, match="Inconsistent"):
        prepare_second(root, tmp_path / "cache", second_config(), local_holdout=True)


def test_device_indices_and_detector_explicit_output_geometry(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_device("cuda") == torch.device("cuda:2")
    assert resolve_device("cuda:3") == torch.device("cuda:3")
    with pytest.raises(ValueError, match="not visible"):
        resolve_device("cuda:4")

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, before, after):
            return torch.zeros_like(before[:, :1]) + self.parameter

    detector = Detector()
    before, after, _ = scene_tensors(17, 23)
    output = detector_probability_map(
        detector, before, after, input_size=(8, 8), output_kind="logits"
    )
    assert output.shape == (1, 17, 23)
    torch.testing.assert_close(output, torch.full_like(output, 0.5))
    assert not detector.training and detector.parameter.grad is None
    output = detector_probability_map(
        detector, before, after, input_size=(8, 8), output_kind="probabilities"
    )
    assert (output == 0).all()  # No double sigmoid or heuristic.
    with pytest.raises(ValueError, match="Unsupported"):
        detector_probability_map(
            detector, before, after, input_size=(8, 8), output_kind="logits", geometry="tiles"
        )


def test_three_seeds_family_and_predicted_evaluation(tmp_path, cpu_queue):
    args = options(tmp_path, epochs=1)
    summary = search(**dict(args, additional_seeds=True))
    family = summary["ranking"][0]
    assert family["n_seeds"] == 3 and family["std_selection_score"] is not None
    assert [r["parameters"]["seed"] for r in family["seed_results"]] == [42, 123, 777]
    checkpoint = Path(family["deployment_best_checkpoint"])
    assert family["deployment_seed"] == 42
    cfg = config()
    cfg["mask_source"] = "predicted"
    cfg["detector"] = {"identity": "synthetic saved probabilities", "training_groups": []}
    scenes = json.loads((tmp_path / "manifest.json").read_text())["scenes"]
    for scene in scenes:
        path = tmp_path / scene["change"]
        np.save(path.with_name(path.stem + "_predicted.npy"), np.load(path) * 0.7)
        scene["change"] = path.stem + "_predicted.npy"
    alternate = tmp_path / "predicted_manifest.json"
    alternate.write_text(json.dumps({"scenes": scenes}))
    prepare_dataset(cfg, alternate, tmp_path / "predicted")
    prepared = tmp_path / "predicted/prepared.json"
    with pytest.raises(ValueError, match="explicit evaluation_config"):
        evaluate_classifier(checkpoint, prepared, "val")
    metrics = evaluate_classifier(
        checkpoint, prepared, "val", evaluation_config=cfg, precision="fp32"
    )
    assert metrics["domain_shift"] and metrics["evaluation_mask_source"] == "predicted"
    assert metrics["training_data_identity"] != metrics["evaluation_data_identity"]
    assert metrics["evaluation_precision"] == "fp32" and metrics["precision_override"] == "fp32"


def test_semantic_scheduler_early_stop_and_resume_validation(tmp_path, cpu_queue):
    from satchange.training import EarlyStopping

    args = options(tmp_path, epochs=4)
    args["base_parameters"]["scheduler"] = "reduce_on_plateau"
    summary = search(**args, patience=1, min_delta=1.0, seed_patience=1)
    selected = summary["ranking"][0]
    row = selected["seed_results"][0]
    assert row["stop_reason"] == "early_stopping" and row["epoch"] == 2
    checkpoint = Path(selected["deployment_best_checkpoint"]).parent / "last.pt"
    state = load_training_checkpoint(checkpoint)
    assert state["scheduler_state_dict"] is not None
    assert state["early_stopping"]["policy"]["monitor"] == "selection_score"
    parameters = deepcopy(args["base_parameters"])
    parameters["model_parameters"]["dropout"] = 0.9
    with pytest.raises(ValueError, match="parameters differ"):
        optimization.read_hpo_checkpoint(
            checkpoint,
            model_name=args["model_name"],
            dataset_root=args["dataset_root"],
            thresholds=[],
            budget=4,
            early_stopping=EarlyStopping(1, 1, 1.0, monitor="selection_score"),
            parameters=parameters,
        )
    parameters = deepcopy(args["base_parameters"])
    parameters.update(gpu_id=3, code_revision="another compatible host")
    optimization.read_hpo_checkpoint(
        checkpoint,
        model_name=args["model_name"],
        dataset_root=args["dataset_root"],
        thresholds=[],
        budget=4,
        early_stopping=EarlyStopping(1, 1, 1.0, monitor="selection_score"),
        parameters=parameters,
    )


def test_binary_controller_worker_schema_and_metric_regression(tmp_path, cpu_queue):
    from satchange.data import create_dataset_dataloader
    from satchange.losses import BCEDiceLoss
    from satchange.run import reconstruct_checkpoint
    from satchange.training import evaluate_thresholds

    for split in ("train", "val"):
        for folder in ("A", "B", "label"):
            directory = tmp_path / "binary" / split / folder
            directory.mkdir(parents=True)
            pixels = np.zeros((32, 32, 3), np.uint8)
            pixels[8:16, 8:16] = 255 if folder != "A" else 20
            Image.fromarray(pixels).save(directory / "scene.png")
    base = dict(
        model_parameters={},
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
        stages=[("baseline", {"learning_rate": [0.001]}), ("next", {"weight_decay": [0.0001]})],
        base_parameters=base,
        model_name="fc_siam_diff",
        dataset_root=tmp_path / "binary",
        thresholds=[0.4, 0.5],
        stage_epochs=[1, 1, 1],
        additional_seeds=False,
        output_root=tmp_path / "searches",
        search_name="binary",
        gpu_ids=[0],
    )
    search(**args, stage="baseline")
    search(**args, stage="next")
    summary = search(**args, stage="seeds")
    family = summary["selected_by_mean_validation_f1"]
    row = family["seed_results"][0]
    assert "best_f1" in row and "best_iou" in row and "selection_score" not in row
    model, state = reconstruct_checkpoint(family["deployment_best_checkpoint"], torch.device("cpu"))
    assert "task" not in state and state["thresholds"] == [0.4, 0.5]
    _, loader = create_dataset_dataloader(tmp_path / "binary", "val", 32, 1, 0)
    metrics = evaluate_thresholds(model, loader, BCEDiceLoss(), torch.device("cpu"), [0.4, 0.5])
    best = max(metrics["threshold_results"], key=lambda r: r["f1"])
    assert row["best_f1"] == best["f1"] and row["best_iou"] == best["iou"]


def test_second_archive_safe_and_official_download_contract(tmp_path, monkeypatch):
    import zipfile

    import gdown

    source = tmp_path / "source"
    second_fixture(source, count=1)
    archive = tmp_path / "SECOND.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.rglob("*.png"):
            bundle.write(path, str(path.relative_to(source)))
    assert bootstrap_second(tmp_path / "local", archive=archive).is_dir()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output"]).write_bytes(archive.read_bytes())
        return kwargs["output"]

    monkeypatch.setattr(gdown, "download", download)
    installed = bootstrap_second(tmp_path / "downloaded", allow_download=True)
    assert calls[0]["id"] == "1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u"
    assert "fuzzy" not in calls[0]
    assert json.loads((installed / "source.json").read_text())["checksum_is_local"]
    malicious = tmp_path / "bad.zip"
    with zipfile.ZipFile(malicious, "w") as bundle:
        bundle.writestr("../escaped.txt", "synthetic")
    with pytest.raises(ValueError, match="Unsafe"):
        bootstrap_second(tmp_path / "unsafe", archive=malicious)
    assert not (tmp_path / "escaped.txt").exists()


def test_actual_v3_queue_dispatch_with_mock_resources(tmp_path, cpu_queue):
    args = options(tmp_path, epochs=1)
    args["stages"] = [("baseline", {"learning_rate": [0.001, 0.002, 0.003, 0.004, 0.005]})]
    args["selection_gap"] = 1.0
    summary = search(**args)
    assert len(cpu_queue) == 5
    assert {call["device"] for call in cpu_queue} == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}
    assert len(summary["ranking"]) == 5


def test_v3_source_fingerprint_includes_semantic_and_controller(monkeypatch):
    from satchange.hpo_backend import SatelliteBackend

    original = Path.read_bytes
    baseline = SatelliteBackend().code_signature()

    def changed(path):
        content = original(path)
        return content + b"\n# synthetic change" if path.name == "roi.py" else content

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert SatelliteBackend().code_signature() != baseline


def test_inference_only_best_export_remains_loadable(tmp_path, cpu_queue):
    from satchange.training import restore_best_checkpoint

    args = options(tmp_path)
    summary = search(**args)
    directory = Path(summary["ranking"][0]["deployment_best_checkpoint"]).parent
    state = load_training_checkpoint(directory / "last.pt")
    first = load_training_checkpoint(
        directory / "checkpoints/tiny_siamese_change_classifier_epoch_0001.pt"
    )
    state["best_metrics"] = first["best_metrics"]
    state["best_model_state_dict"] = first["model_state_dict"]
    export = tmp_path / "isolated/best.pt"
    restore_best_checkpoint(state, tmp_path / "isolated/last.pt", export)
    model, loaded = load_classifier(export)
    assert loaded["checkpoint_kind"] == "selection_weights" and loaded["task"] == SEMANTIC_TASK
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, first["model_state_dict"][key])
    with pytest.raises(ValueError, match="complete training checkpoint"):
        load_training_checkpoint(export)


def test_v3_three_phase_promotion_final_benchmark_and_validation_gate(
    tmp_path, cpu_queue, monkeypatch
):
    """Real small CPU epochs verify 1 -> 2 -> 3 totals, final export, and val-before-test."""
    from satchange.final_benchmark import run_final_benchmark
    from satchange.semantic import benchmark as semantic_benchmark

    args = options(tmp_path, epochs=1)
    args.update(stage_epochs=[1, 2, 3], additional_seeds=True, patience=1)
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
        assert last["early_stopping"]["policy"] is None
    saved_calls = len(cpu_queue)
    hpo.run_automated_parameter_search(**args, stage="final")
    assert len(cpu_queue) == saved_calls

    calls = []
    evaluate = semantic_benchmark.evaluate_classifier

    def observed(*positional, **kwargs):
        calls.append(positional[2])
        return evaluate(*positional, **kwargs)

    monkeypatch.setattr(semantic_benchmark, "evaluate_classifier", observed)
    options_benchmark = dict(
        task=SEMANTIC_TASK,
        variant="tiny_test",
        checkpoints=final["best_checkpoints"],
        project_root=tmp_path,
        dataset_root=args["dataset_root"],
        training_results=final,
        deployment_seed=42,
        device="cpu",
        warmup=0,
        repetitions=2,
    )
    report = run_final_benchmark(**options_benchmark, output_dir=tmp_path / "final_bundle")
    assert calls == ["val"] * 3 + ["test"] * 3
    assert report["metadata"]["deployment_seed"] == 42
    assert report["final_test_results"]["test_selection_score_std"] is not None
    assert "test_iou_mean" not in report["final_test_results"]
    assert report["benchmark"]["protocol"]["scope"].startswith("classifier only")
    out = Path(report["output_dir"])
    for path in (
        "metadata.json",
        "benchmark.json",
        "training_results.json",
        "validation_results.json",
        "final_test_results.json",
        "deployment_weights.pt",
    ):
        assert (out / path).is_file()
    state = torch.load(out / "deployment_weights.pt", map_location="cpu", weights_only=True)
    assert "optimizer_state_dict" not in state

    calls.clear()

    def changed_validation(*positional, **kwargs):
        result = observed(*positional, **kwargs)
        result["selection_score"] += 0.2
        return result

    monkeypatch.setattr(semantic_benchmark, "evaluate_classifier", changed_validation)
    with pytest.raises(ValueError, match="No test evaluation"):
        run_final_benchmark(**options_benchmark, output_dir=tmp_path / "rejected_bundle")
    assert calls == ["val"] and not (tmp_path / "rejected_bundle").exists()


def test_controller_lock_is_exclusive_and_released(tmp_path):
    from satchange.hpo_core import experiment_lock

    with experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another HPO"):
            with experiment_lock(tmp_path):
                pass
    with experiment_lock(tmp_path):
        pass
