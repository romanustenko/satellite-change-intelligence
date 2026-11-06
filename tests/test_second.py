"""SECOND split manifest on a tiny synthetic layout; no ROI preparation, no real data."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from satchange.semantic import second
from satchange.semantic.data import file_digest
from satchange.semantic.second import (
    LOCAL_HOLDOUT_FRACTION,
    SECOND_COLORS,
    bootstrap_second,
    decode_label,
    discover_second,
    prepare_second_splits,
    read_second_split_manifest,
)

# Splits that prepare_second(local_holdout=True) produced with the legacy inline algorithm for
# stems t000..t019 (seed 2099, 15 % per missing split); recorded before the logic was shared.
LEGACY_2099 = {
    "val": ["train/t001", "train/t005", "train/t007"],
    "test": ["train/t002", "train/t009", "train/t010"],
}


def stems(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index:03d}" for index in range(count)]


def write_leaf(root: Path, split: str, names: list[str], *, labels: bool = True) -> None:
    """Write tiny PNG files; split manifests never inspect pixels, only names and bytes."""
    folders = ("im1", "im2", "label1", "label2") if labels else ("im1", "im2")
    for offset, folder in enumerate(folders):
        target = root / split / folder
        target.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(names):
            value = (3 * index + 50 * offset + len(split)) % 256
            Image.fromarray(np.full((4, 4, 3), value, np.uint8)).save(target / f"{name}.png")


def make_second(
    root: Path, *, train: int = 12, val: int = 0, test: int = 0, unlabeled_test: int = 0
) -> Path:
    write_leaf(root, "train", stems("t", train))
    if val:
        write_leaf(root, "val", stems("v", val))
    if test:
        write_leaf(root, "test", stems("s", test))
    if unlabeled_test:
        write_leaf(root, "test", stems("u", unlabeled_test), labels=False)
    return root


def test_official_splits_are_preserved_and_hashed(tmp_path):
    root = make_second(tmp_path / "SECOND", train=6, val=3, test=2)
    destination = tmp_path / "out" / "splits.json"
    manifest = prepare_second_splits(root, destination, split_seed=2099)
    assert manifest["splits"] == {
        "train": [f"train/{name}" for name in stems("t", 6)],
        "val": [f"val/{name}" for name in stems("v", 3)],
        "test": [f"test/{name}" for name in stems("s", 2)],
    }
    assert manifest["counts"] == manifest["official_counts"] == {"train": 6, "val": 3, "test": 2}
    assert manifest["policy"] == {
        "val": "official",
        "test": "official",
        "val_fraction": None,
        "test_fraction": None,
    }
    assert manifest["split_seed"] == 2099
    assert manifest["schema_version"] == 1
    assert manifest["geographic_independence_verified"] is False
    assert manifest["group_basis"] == "filename only"
    assert all(s["split"] == s["origin_split"] for s in manifest["samples"].values())
    sample = manifest["samples"]["val/v001"]
    assert sample["group_id"] == "v001"
    assert sample["files"]["after_semantic"]["path"] == "val/label2/v001.png"
    assert sample["files"]["after_semantic"]["sha256"] == file_digest(root / "val/label2/v001.png")
    assert read_second_split_manifest(destination) == manifest
    # Only the manifest is produced: no ROI crops, caches or converted maps.
    assert [path.name for path in destination.parent.iterdir()] == ["splits.json"]


def test_missing_val_needs_explicit_holdout_and_test_policy(tmp_path):
    root = make_second(tmp_path / "SECOND", train=12)
    with pytest.raises(ValueError, match="val_fraction"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=1)
    with pytest.raises(ValueError, match="test_fraction.*allow_missing_test"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=1, val_fraction=0.25)
    assert not (tmp_path / "a.json").exists()
    manifest = prepare_second_splits(
        root, tmp_path / "b.json", split_seed=1, val_fraction=0.25, allow_missing_test=True
    )
    assert manifest["counts"] == {"train": 9, "val": 3, "test": 0}
    assert manifest["policy"]["val"] == "local_holdout"
    assert manifest["policy"]["test"] == "unavailable"
    assert manifest["policy"]["val_fraction"] == 0.25
    assert manifest["splits"]["test"] == []
    assert manifest["official_counts"] == {"train": 12, "val": 0, "test": 0}
    assert {manifest["samples"][name]["origin_split"] for name in manifest["splits"]["val"]} == {
        "train"
    }


def test_missing_test_fails_by_default_and_local_test_is_explicit(tmp_path):
    root = make_second(tmp_path / "SECOND", train=12, val=3)
    with pytest.raises(ValueError, match="test_fraction"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=1)
    manifest = prepare_second_splits(root, tmp_path / "b.json", split_seed=1, test_fraction=0.25)
    assert manifest["counts"] == {"train": 9, "val": 3, "test": 3}
    assert manifest["policy"]["val"] == "official"
    assert manifest["policy"]["test"] == "local_holdout"
    # Official val is never touched by the local test carve-out.
    assert manifest["splits"]["val"] == [f"val/{name}" for name in stems("v", 3)]


def test_unlabeled_test_leaf_is_recorded_not_invented(tmp_path):
    root = make_second(tmp_path / "SECOND", train=12, val=3, unlabeled_test=4)
    assert discover_second(root)["test"] == []
    with pytest.raises(ValueError, match="test_fraction"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=1)
    manifest = prepare_second_splits(
        root, tmp_path / "b.json", split_seed=1, allow_missing_test=True
    )
    assert manifest["splits"]["test"] == []
    assert manifest["official_leaves"] == ["test", "train", "val"]
    assert manifest["official_counts"]["test"] == 0
    local = prepare_second_splits(root, tmp_path / "c.json", split_seed=1, test_fraction=0.25)
    # The unlabeled images are never used as a labeled test split.
    assert all(name.startswith("train/") for name in local["splits"]["test"])
    assert not any("/u0" in name for name in local["samples"])


def test_official_splits_are_never_overridden(tmp_path):
    root = make_second(tmp_path / "SECOND", train=12, val=3, test=3)
    with pytest.raises(ValueError, match="Official labeled val exists"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=1, val_fraction=0.1)
    with pytest.raises(ValueError, match="Official labeled test exists"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=1, test_fraction=0.1)
    assert not (tmp_path / "a.json").exists()


def test_local_holdout_is_group_disjoint(tmp_path):
    root = make_second(tmp_path / "SECOND", train=20)
    groups = {f"train/{name}": f"scene{index // 4}" for index, name in enumerate(stems("t", 20))}
    manifest = prepare_second_splits(
        root,
        tmp_path / "groups.json",
        split_seed=5,
        val_fraction=0.2,
        test_fraction=0.2,
        source_groups=groups,
    )
    assert manifest["group_basis"] == "source metadata"
    assert manifest["counts"] == {"train": 12, "val": 4, "test": 4}
    assert manifest["group_counts"] == {"train": 3, "val": 1, "test": 1}
    owners: dict[str, set[str]] = {}
    for sample_id, sample in manifest["samples"].items():
        assert sample["group_id"] == groups[sample_id]
        owners.setdefault(sample["group_id"], set()).add(sample["split"])
    assert all(len(splits) == 1 for splits in owners.values())
    with pytest.raises(ValueError, match="Missing source group"):
        prepare_second_splits(
            root,
            tmp_path / "missing.json",
            split_seed=5,
            val_fraction=0.2,
            allow_missing_test=True,
            source_groups={"train/t000": "scene0"},
        )


def test_supplied_groups_must_not_span_official_splits(tmp_path):
    root = make_second(tmp_path / "SECOND", train=6, val=2, test=2)
    groups = {
        scene["sample_id"]: scene["sample_id"]
        for rows in discover_second(root).values()
        for scene in rows
    }
    groups["val/v000"] = groups["train/t000"]  # same scene in official train and val
    with pytest.raises(ValueError, match="spans splits"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=1, source_groups=groups)
    assert not (tmp_path / "a.json").exists()


def test_filename_groups_report_shared_stems_without_failing(tmp_path):
    root = tmp_path / "SECOND"
    write_leaf(root, "train", ["000", "001", "002"])
    write_leaf(root, "val", ["000", "009"])
    write_leaf(root, "test", ["010"])
    manifest = prepare_second_splits(root, tmp_path / "a.json", split_seed=1)
    assert manifest["diagnostics"]["stems_shared_across_official_splits"] == ["000"]
    assert set(manifest["samples"]) >= {"train/000", "val/000"}


def test_same_seed_is_identical_and_matches_legacy_holdout(tmp_path):
    root = make_second(tmp_path / "SECOND", train=20)
    args = {"val_fraction": LOCAL_HOLDOUT_FRACTION, "test_fraction": LOCAL_HOLDOUT_FRACTION}
    first = prepare_second_splits(root, tmp_path / "one" / "m.json", split_seed=2099, **args)
    second_run = prepare_second_splits(root, tmp_path / "two" / "m.json", split_seed=2099, **args)
    assert first == second_run
    assert (tmp_path / "one" / "m.json").read_bytes() == (tmp_path / "two" / "m.json").read_bytes()
    assert first["splits"]["val"] == LEGACY_2099["val"]
    assert first["splits"]["test"] == LEGACY_2099["test"]
    other = prepare_second_splits(root, tmp_path / "three.json", split_seed=7, **args)
    assert other["splits"]["val"] != first["splits"]["val"]
    assert other["manifest_sha256"] != first["manifest_sha256"]
    assert other["counts"] == first["counts"]


def test_shared_holdout_helper_keeps_prepare_second_semantics(tmp_path):
    """prepare_second calls the helper with exactly these arguments."""
    root = make_second(tmp_path / "SECOND", train=20)
    scenes = [scene for rows in discover_second(root).values() for scene in rows]
    second._assign_local_holdout(
        scenes, dict.fromkeys(["val", "test"], LOCAL_HOLDOUT_FRACTION), 2099
    )
    for split in ("val", "test"):
        assert [s["sample_id"] for s in scenes if s["split"] == split] == LEGACY_2099[split]
    assert sum(s["split"] == "train" for s in scenes) == 14


def test_manifest_is_frozen_atomic_and_tamper_evident(tmp_path, monkeypatch):
    root = make_second(tmp_path / "SECOND", train=12)
    destination = tmp_path / "out" / "splits.json"
    args = {"val_fraction": 0.25, "allow_missing_test": True}

    def broken_replace(*_args, **_kwargs):
        raise OSError("simulated crash before commit")

    with monkeypatch.context() as patch:
        patch.setattr(second.os, "replace", broken_replace)
        with pytest.raises(OSError, match="simulated"):
            prepare_second_splits(root, destination, split_seed=3, **args)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []  # no partial or temporary file

    manifest = prepare_second_splits(root, destination, split_seed=3, **args)
    snapshot = destination.read_bytes()
    assert prepare_second_splits(root, destination, split_seed=3, **args) == manifest
    assert destination.read_bytes() == snapshot
    with pytest.raises(FileExistsError, match="frozen"):
        prepare_second_splits(root, destination, split_seed=4, **args)
    assert destination.read_bytes() == snapshot

    edited = json.loads(snapshot)
    edited["splits"]["val"].append("train/t000")
    destination.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_second_split_manifest(destination)
    with pytest.raises(ValueError, match="digest mismatch"):
        prepare_second_splits(root, destination, split_seed=3, **args)
    destination.write_text(json.dumps({"kind": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Not a supported"):
        read_second_split_manifest(destination)


def test_source_changes_alter_the_manifest_digest(tmp_path):
    root = make_second(tmp_path / "SECOND", train=12)
    args = {"split_seed": 3, "val_fraction": 0.25, "allow_missing_test": True}
    first = prepare_second_splits(root, tmp_path / "a.json", **args)
    Image.fromarray(np.full((4, 4, 3), 99, np.uint8)).save(root / "train/label1/t004.png")
    changed = prepare_second_splits(root, tmp_path / "b.json", **args)
    assert changed["splits"] == first["splits"]
    assert changed["manifest_sha256"] != first["manifest_sha256"]
    assert (
        changed["samples"]["train/t004"]["files"]["before_semantic"]["sha256"]
        != first["samples"]["train/t004"]["files"]["before_semantic"]["sha256"]
    )


@pytest.mark.parametrize("seed", [-1, 1.5, True, "7", None])
def test_split_seed_must_be_a_fixed_nonnegative_integer(tmp_path, seed):
    root = make_second(tmp_path / "SECOND", train=12, val=3, test=3)
    with pytest.raises(ValueError, match="split_seed"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=seed)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5, True, "0.1"])
def test_fractions_must_be_strictly_between_zero_and_one(tmp_path, fraction):
    root = make_second(tmp_path / "SECOND", train=12)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        prepare_second_splits(
            root, tmp_path / "a.json", split_seed=1, val_fraction=fraction, allow_missing_test=True
        )


def test_policy_and_capacity_failures(tmp_path):
    root = make_second(tmp_path / "SECOND", train=4)
    out = tmp_path / "a.json"
    with pytest.raises(ValueError, match="leave training groups"):
        prepare_second_splits(root, out, split_seed=1, val_fraction=0.5, test_fraction=0.5)
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_second_splits(
            root, out, split_seed=1, val_fraction=0.2, test_fraction=0.2, allow_missing_test=True
        )
    with pytest.raises(ValueError, match="no training groups"):
        prepare_second_splits(root, out, split_seed=1, val_fraction=0.75, test_fraction=0.2)
    tiny = make_second(tmp_path / "TINY", train=2)
    with pytest.raises(ValueError, match="Too few source groups"):
        prepare_second_splits(tiny, out, split_seed=1, val_fraction=0.4, test_fraction=0.4)
    assert not out.exists()


def test_broken_layouts_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No labeled SECOND train"):
        prepare_second_splits(tmp_path / "empty", tmp_path / "a.json", split_seed=1)
    root = make_second(tmp_path / "SECOND", train=4, val=2, test=2)
    (root / "val/label2/v000.png").unlink()
    with pytest.raises(ValueError, match="Missing/extra"):
        prepare_second_splits(root, tmp_path / "a.json", split_seed=1)
    assert not (tmp_path / "a.json").exists()


# Moved from the removed TinySiamese test module (SECOND decoding/installation).


def archive_fixture(root, count=5):
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


def test_second_exact_palette_ids(tmp_path):
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


def test_second_archive_safe_and_official_download_contract(tmp_path, monkeypatch):
    import zipfile

    import gdown

    source = tmp_path / "source"
    archive_fixture(source, count=1)
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
