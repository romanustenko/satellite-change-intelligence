"""Freeze manifest and ROI policy selection (CPU, synthetic)."""

import json
import subprocess
from pathlib import Path

import pytest
import torch

from satchange.semantic.freeze import build_freeze_manifest, sha256_file
from satchange.semantic.roi_policy import coverage_diagnostics, select_roi_policy


def _repository(tmp_path: Path) -> Path:
    for command in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(tmp_path), *command], check=True)
    (tmp_path / "protocol.md").write_text("v1")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "x"], check=True)
    return tmp_path


def test_manifest_hashes_files_refuses_overwrite_and_dirty_tree(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    checkpoint = tmp_path / "w.pt"
    checkpoint.write_bytes(b"x" * 2048)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    systems = {"A": {"checkpoints": [checkpoint], "config": {"lr": 1e-3},
                     "selection_evidence": evidence}}
    output = tmp_path / "freeze.json"
    manifest = build_freeze_manifest(
        systems, root=root, protocol=root / "protocol.md", output=output
    )
    saved = json.loads(output.read_text())
    assert saved["systems"]["A"]["checkpoints"][0]["sha256"] == sha256_file(checkpoint)
    assert manifest["test_split_read"] is False and not saved["git_dirty"]
    with pytest.raises(FileExistsError):
        build_freeze_manifest(systems, root=root, protocol=root / "protocol.md", output=output)
    (root / "protocol.md").write_text("changed")
    with pytest.raises(RuntimeError):
        build_freeze_manifest(
            systems, root=root, protocol=root / "protocol.md", output=tmp_path / "other.json"
        )


def test_policy_selection_prefers_best_and_earlier_on_ties() -> None:
    def evaluate(settings):  # type: ignore[no-untyped-def]
        return {"sek": 0.5 if settings.threshold >= 0.5 else 0.2}

    result = select_roi_policy(
        evaluate, thresholds=[0.3, 0.5, 0.7], context_pixels=[16], min_component_areas=[1]
    )
    assert result["best"]["settings"]["threshold"] == 0.5 and len(result["rows"]) == 3


def test_coverage_diagnostics_counts_misses_and_false_crops() -> None:
    probabilities = torch.zeros(1, 256, 256)
    probabilities[:, 20:40, 200:220] = 1  # detected, but no change here: a false crop
    probabilities[:, 200:210, 200:210] = 1  # detected change
    change = torch.zeros(1, 256, 256)
    change[:, 200:210, 200:210] = 1
    change[:, 100:110, 100:110] = 1  # missed by the detector
    result = coverage_diagnostics(probabilities, change)
    assert result["crops"] == 2 and result["false_crops"] == 1
    assert result["missed_fraction"] == pytest.approx(0.5)
