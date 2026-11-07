"""SECOND binary change detection via a frozen split manifest, and LEVIR -> SECOND fine-tuning."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_hpo_v3 import cpu_queue, search, threads  # noqa: F401  (shared fixtures)

from satchange.data import create_dataset_dataloader
from satchange.data.second_binary import (
    SECOND_MANIFEST_NAME,
    SECONDBinaryDataset,
    verify_second_sources,
)
from satchange.hpo_core import PipelineRunError
from satchange.semantic.second import prepare_second_splits

WHITE, TREE = (255, 255, 255), (0, 255, 0)


def write_scene(root: Path, split: str, name: str, index: int) -> None:
    for folder in ("im1", "im2", "label1", "label2"):
        (root / split / folder).mkdir(parents=True, exist_ok=True)
    before = np.full((32, 32, 3), 20 + index, np.uint8)
    after = before.copy()
    label = np.full((32, 32, 3), WHITE, np.uint8)
    after[4 + index : 12 + index, 6:14] = 200
    label[4 + index : 12 + index, 6:14] = TREE
    for folder, array in (("im1", before), ("im2", after), ("label1", label), ("label2", label)):
        Image.fromarray(array).save(root / split / folder / f"{name}.png")


def make_second(root: Path, train: int = 4, test: int = 2) -> Path:
    for index in range(train):
        write_scene(root, "train", f"t{index:03d}", index)
    for index in range(test):
        write_scene(root, "test", f"s{index:03d}", index)
    prepare_second_splits(root, root / SECOND_MANIFEST_NAME, split_seed=7, val_fraction=0.5)
    return root


def test_manifest_root_dispatches_to_second_with_levir_contract(tmp_path):
    root = make_second(tmp_path / "SECOND")
    dataset, loader = create_dataset_dataloader(root, "train", None, 2, 0)
    assert isinstance(dataset, SECONDBinaryDataset) and len(dataset) == 2
    sample = dataset[0]
    assert sample["before"].shape == (3, 32, 32) and sample["before"].dtype == torch.float32
    assert 0 <= float(sample["before"].min()) and float(sample["before"].max()) <= 1
    assert sample["mask"].shape == (1, 32, 32) and set(sample["mask"].unique().tolist()) == {0, 1}
    assert int(sample["mask"].sum()) == 64  # The 8x8 tree change; white is non-change.
    assert sample["sample_id"].startswith("train/")
    resized, _ = create_dataset_dataloader(root, "val", 16, 1, 0)
    assert resized[0]["mask"].shape == (1, 16, 16)
    assert next(iter(loader))["before"].shape == (2, 3, 32, 32)
    test_split, _ = create_dataset_dataloader(root, "test", None, 1, 0)
    assert [row for row in test_split.sample_ids] == ["test/s000", "test/s001"]
    assert verify_second_sources(root) == 6 * 4
    # final_benchmark reads native shapes through the LEVIRCDDataset path maps.
    for files in (test_split.before_files, test_split.after_files, test_split.label_files):
        assert sorted(files) == test_split.sample_ids and all(p.is_file() for p in files.values())


def test_label_disagreement_missing_files_and_changed_sources_fail(tmp_path):
    root = make_second(tmp_path / "SECOND")
    dataset = SECONDBinaryDataset(root, "train", None)
    sample_id = dataset.sample_ids[0]
    label2 = dataset.files[sample_id]["after_semantic"]
    Image.fromarray(np.full((32, 32, 3), WHITE, np.uint8)).save(label2)
    with pytest.raises(ValueError, match="disagrees"):
        dataset[0]
    with pytest.raises(ValueError, match="source changed"):
        verify_second_sources(root)
    label2.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        SECONDBinaryDataset(root, "train", None)


def binary_options(tmp_path, root, name, **extra):
    base = dict(
        model_parameters={"base_channels": 4}, seed=42, image_size=None, batch_size=1,
        num_workers=0, learning_rate=0.001, weight_decay=0.0001, scheduler="none",
        positive_weight=None, bce_weight=1.0, dice_weight=1.0, precision="fp32",
    )
    base.update(extra)
    return dict(
        stages=[("baseline", {"learning_rate": [0.001]})], base_parameters=base,
        model_name="fc_siam_diff", dataset_root=root, thresholds=[0.5], stage_epochs=[1, 1, 1],
        additional_seeds=False, output_root=tmp_path / "searches", search_name=name,
        gpu_ids=[0], verbose=False, checkpoint_interval=1,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_scratch_and_initialized_runs_differ_only_by_initialization(tmp_path, cpu_queue):  # noqa: F811
    root = make_second(tmp_path / "SECOND")
    scratch = search(**binary_options(tmp_path, root, "scratch"))
    source = Path(scratch["selected"]["seed_results"][0]["best_checkpoint"])
    spec = {"path": str(source), "sha256": digest(source), "exclude": []}
    tuned = search(**binary_options(tmp_path, root, "tuned", initial_weights=spec))
    row = tuned["selected"]["seed_results"][0]
    assert row["parameters"]["initial_weights"] == spec
    assert "initial_weights" not in scratch["selected"]["seed_results"][0]["parameters"]
    summary = Path(row["summary_path"]).read_text(encoding="utf-8")
    assert '"loaded_keys": 46' in summary and '"excluded_keys": []' in summary
    head = dict(spec, exclude=["output_layer."])
    report = search(**binary_options(tmp_path, root, "head", initial_weights=head))
    text = Path(report["selected"]["seed_results"][0]["summary_path"]).read_text(encoding="utf-8")
    assert '"loaded_keys": 44' in text and "output_layer.weight" in text
    with pytest.raises(PipelineRunError) as failed:
        search(**binary_options(tmp_path, root, "bad", initial_weights=dict(spec, sha256="0" * 64)))
    assert "SHA-256 mismatch" in failed.value.failed_trials[0]["error"]
    wider = binary_options(tmp_path, root, "wider", initial_weights=spec)
    wider["base_parameters"]["model_parameters"] = {"base_channels": 8}
    with pytest.raises(PipelineRunError) as failed:
        search(**wider)
    assert "shape mismatch" in failed.value.failed_trials[0]["error"]
    with pytest.raises(ValueError, match="exactly path, sha256 and exclude"):
        search(**binary_options(tmp_path, root, "malformed", initial_weights={"path": "x"}))
