"""Per-scene system evaluation, strata/seed bootstrap and freeze verification (CPU, synthetic)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from satchange.models import FCSiamDiff
from satchange.semantic.analysis import (
    assign_strata,
    latency_ratio_by_stratum,
    metrics_from_histogram,
    non_inferior,
    paired_bootstrap_seeds,
    quantile_edges,
    spearman,
)
from satchange.semantic.freeze import (
    build_freeze_manifest,
    claim_test_read,
    verify_freeze_manifest,
)
from satchange.semantic.system_eval import (
    evaluate_system_a,
    evaluate_system_b,
    load_semantic_checkpoint,
)
from satchange.semantic.what_stage import evaluate_what_pipeline
from tests.test_what_stage import _FakeBase

CPU = torch.device("cpu")


def _model() -> FCSiamDiff:
    torch.manual_seed(0)
    return FCSiamDiff(in_channels=3, out_channels=14, base_channels=8).eval()


def test_system_b_oracle_matches_the_pooled_pipeline_and_regions_partition_pixels() -> None:
    model, data = _model(), _FakeBase()
    result = evaluate_system_b(model, data, CPU, None)  # type: ignore[arg-type]
    pooled = metrics_from_histogram(result["histograms"].sum(0))
    expected = evaluate_what_pipeline(model, data, CPU)  # type: ignore[arg-type]
    for key in ("sek", "miou", "fscd", "oa"):
        assert pooled[key] == pytest.approx(expected[key])
    assert np.array_equal(
        result["histograms"], result["histograms_inside_roi"] + result["histograms_outside_roi"]
    )
    assert result["histograms"].shape == (2, 7, 7) and result["density"].shape == (2,)
    assert result["coverage"][0]["missed_change_pixels"] == 0  # oracle ROIs cover all change


def test_system_b_with_empty_detector_predicts_no_change_everywhere() -> None:
    data = _FakeBase()
    result = evaluate_system_b(
        _model(), data, CPU, lambda sample: torch.zeros_like(sample["mask"])  # type: ignore[arg-type]
    )
    assert result["histograms"][:, 1:, :].sum() == 0  # no predicted class other than 0
    assert result["coverage"][0]["missed_fraction"] == 1.0


def test_system_a_histograms_cover_every_pixel_of_both_dates() -> None:
    result = evaluate_system_a(_model(), _FakeBase(), CPU)  # type: ignore[arg-type]
    assert (result["histograms"].sum((1, 2)) == 2 * 256 * 256).all()


def test_load_semantic_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "best.pt"
    torch.save({"model_name": "fc_siam_diff", "model_state_dict": model.state_dict(),
                "config": {"model_parameters": {"in_channels": 3, "out_channels": 14,
                                                "base_channels": 8}}}, path)
    loaded = load_semantic_checkpoint(path, CPU)
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def _hists(accuracy: float, seeds: int = 2, scenes: int = 20) -> np.ndarray:
    out = []
    for s in range(seeds):
        generator = torch.Generator().manual_seed(s)
        scene_hists = []
        for _ in range(scenes):
            label = torch.randint(0, 7, (24, 24), generator=generator)
            noise = torch.randint(0, 7, (24, 24), generator=generator)
            keep = torch.rand(24, 24, generator=generator) < accuracy
            prediction = torch.where(keep, label, noise)
            hist = np.zeros((7, 7), dtype=np.int64)
            np.add.at(hist, (prediction.numpy().ravel(), label.numpy().ravel()), 1)
            scene_hists.append(2 * hist)
        out.append(np.stack(scene_hists))
    return np.stack(out)


def test_seed_bootstrap_is_paired_seeded_and_supports_strata_and_margin() -> None:
    better, worse = _hists(0.95), _hists(0.6)
    first = paired_bootstrap_seeds(better, worse, resamples=100, seed=1)
    assert first["ci_low"] > 0 and first == paired_bootstrap_seeds(better, worse, resamples=100,
                                                                   seed=1)
    assert non_inferior(first, 0.01) and not non_inferior(
        paired_bootstrap_seeds(worse, better, resamples=100), 0.01
    )
    stratum = paired_bootstrap_seeds(better, worse, resamples=50, subset=np.arange(5))
    assert stratum["scenes"] == 5
    with pytest.raises(ValueError):
        paired_bootstrap_seeds(better, worse[:1])
    with pytest.raises(ValueError):
        non_inferior(first, 0.0)


def test_strata_edges_assignment_and_latency_trend() -> None:
    density = np.linspace(0.01, 0.9, 90)
    edges = quantile_edges(density, 3)
    strata = assign_strata(density, edges)
    assert sorted(set(strata)) == [0, 1, 2] and np.bincount(strata).tolist() == [30, 30, 30]
    with pytest.raises(ValueError):
        quantile_edges(np.ones(10), 3)
    latency_a = np.full(90, 100.0)
    latency_b = 60.0 + 200.0 * density  # B becomes slower than A as density grows
    report = latency_ratio_by_stratum(latency_b, latency_a, density, edges, resamples=100)
    ratios = [row["ratio"] for row in report["strata"]]  # type: ignore[index]
    assert ratios[0] < 1 < ratios[2] and report["trend"]["spearman"] > 0.99  # type: ignore[index]
    assert spearman(np.arange(5), np.arange(5)[::-1]) == pytest.approx(-1.0)


def _repository(root: Path) -> None:
    for command in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(root), *command], check=True)
    (root / "src" / "satchange").mkdir(parents=True)
    (root / "src" / "satchange" / "a.py").write_text("x = 1\n")
    (root / "protocol.md").write_text("v1")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)


def test_verify_detects_changed_checkpoint_and_code_and_test_claim_is_single_use(
    tmp_path: Path,
) -> None:
    _repository(tmp_path)
    checkpoint, evidence = tmp_path / "w.pt", tmp_path / "e.json"
    checkpoint.write_bytes(b"x" * 2048)
    evidence.write_text("{}")
    manifest_path = tmp_path / "freeze.json"
    build_freeze_manifest(
        {"A": {"checkpoints": [checkpoint], "config": {}, "selection_evidence": evidence}},
        root=tmp_path, protocol=tmp_path / "protocol.md", output=manifest_path,
    )
    assert verify_freeze_manifest(manifest_path, root=tmp_path)["test_split_read"] is False
    marker = claim_test_read(manifest_path, system_set="A")
    assert json.loads(marker.read_text())["system_set"] == "A"
    with pytest.raises(FileExistsError):
        claim_test_read(manifest_path, system_set="A")

    checkpoint.write_bytes(b"y" * 2048)
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        verify_freeze_manifest(manifest_path, root=tmp_path)
    checkpoint.write_bytes(b"x" * 2048)
    (tmp_path / "src" / "satchange" / "a.py").write_text("x = 2\n")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qam", "code change"], check=True)
    with pytest.raises(RuntimeError, match="frozen code changed"):
        verify_freeze_manifest(manifest_path, root=tmp_path)


def test_error_decomposition_splits_misses_by_region() -> None:
    from satchange.semantic.analysis import error_decomposition

    inside, outside = np.zeros((7, 7), dtype=np.int64), np.zeros((7, 7), dtype=np.int64)
    inside[2, 2], inside[3, 2], inside[1, 0] = 8, 2, 5  # 8 correct, 2 wrong class, 5 false change
    outside[0, 1], outside[0, 0] = 6, 100  # 6 changed pixels predicted as no change
    report = error_decomposition(inside, outside)
    assert report["changed_pixels"] == 16 and report["missed_inside_roi"] == 2
    assert report["missed_outside_roi"] == 6 and report["false_change_inside_roi"] == 5
    assert report["outside_share_of_missed"] == pytest.approx(0.75)


def test_analysis_script_end_to_end_on_synthetic_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "analyze_r2_frozen", Path(__file__).parents[1] / "src/scripts/analyze_r2_frozen.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenes = 30
    good, bad = _hists(0.95, scenes=scenes), _hists(0.6, scenes=scenes)
    zeros = np.zeros_like(good)
    np.savez_compressed(
        tmp_path / "scene_results.npz", seeds=np.array([1, 2]), density=np.linspace(0, 1, scenes),
        sample_ids=np.array([f"s{i}" for i in range(scenes)]), hist_a=bad, hist_b=good,
        hist_oracle=good, hist_b_inside=good, hist_b_outside=zeros,
    )
    (tmp_path / "run.json").write_text(json.dumps({"split": "val"}))
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"systems": {"B": {"config": {"analysis": {
        "delta": 0.01, "density_edges": [0.33, 0.66], "resamples": 20}}}}}))
    monkeypatch.setattr(sys, "argv", ["x", "--results", str(tmp_path), "--manifest", str(manifest)])
    module.main()
    report = json.loads((tmp_path / "analysis.json").read_text())
    assert report["comparisons"]["all"]["B_minus_A_sek"]["non_inferior"] is True
    assert [report["comparisons"][f"stratum_{k}"]["scenes"] for k in range(3)] == [10, 10, 10]
    with pytest.raises(FileExistsError):
        module.main()
