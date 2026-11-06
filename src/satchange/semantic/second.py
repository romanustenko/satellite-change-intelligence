"""Exact SECOND conversion and versioned preparation; never starts training on import."""

import json
import os
import random
import stat
import zipfile
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from satchange.semantic.data import file_digest, write_json

SECOND_SOURCE = "https://captain-whu.github.io/SCD/"
SECOND_DOWNLOAD = "https://drive.google.com/file/d/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u/view"
PALETTE_SOURCE = "https://raw.githubusercontent.com/yzygit1230/SCD-SAM/main/datasets/RS_ST.py"
SECOND_CLASSES = (
    "non-vegetated ground surface",
    "tree",
    "low vegetation",
    "water",
    "buildings",
    "playgrounds",
)
# Canonical local IDs, independent of a PNG's palette indices. White is non-change.
SECOND_COLORS = {
    (255, 255, 255): 0,
    (128, 128, 128): 1,
    (0, 255, 0): 2,
    (0, 128, 0): 3,
    (0, 0, 255): 4,
    (128, 0, 0): 5,
    (255, 0, 0): 6,
}
LOCAL_HOLDOUT_FRACTION = 0.15  # Share of official-train groups per locally created split.
SPLIT_MANIFEST_SCHEMA = 1
SCENE_FILES = ("before", "after", "before_semantic", "after_semantic")


def decode_label(path: str | Path, indexed_ids: dict[int, str] | None = None) -> np.ndarray:
    """Decode exact RGB/P colors or explicitly declared scalar IDs; unknown values fail."""
    with Image.open(path) as image:
        if image.mode in ("RGB", "P"):
            pixels = np.asarray(image.convert("RGB"))
            result = np.full(pixels.shape[:2], -1, dtype=np.int16)
            for color, identifier in SECOND_COLORS.items():
                result[(pixels == color).all(axis=2)] = identifier
            if (result < 0).any():
                unknown = np.unique(pixels[result < 0], axis=0)[:5].tolist()
                raise ValueError(f"Unknown SECOND colors {unknown}: {path}; no color guessing.")
            return result
        if image.mode not in ("L", "I", "I;16") or indexed_ids is None:
            raise ValueError(f"{path}: mode {image.mode} needs explicit indexed_id_names mapping.")
        pixels = np.asarray(image)
        result = np.full(pixels.shape, -1, dtype=np.int16)
        for raw, name in indexed_ids.items():
            if name != "non-change" and name not in SECOND_CLASSES:
                raise ValueError(f"Undeclared SECOND class {name!r}; scalar ID {raw}.")
            result[pixels == int(raw)] = (
                0 if name == "non-change" else SECOND_CLASSES.index(name) + 1
            )
        if (result < 0).any():
            raise ValueError(f"Unknown SECOND IDs {np.unique(pixels[result < 0])[:5]}: {path}")
        return result


def discover_second(root: str | Path) -> dict[str, list[dict[str, str]]]:
    """Inspect installed train/val/test leaves with im1/im2/label1/label2; match exact stems.

    Unlabeled test leaves are recorded as unavailable rather than invented annotations.
    Duplicate/extra/missing IDs and ambiguous duplicate split directories fail.
    """
    root = Path(root).resolve()
    found: dict[str, list[dict[str, str]]] = {}
    for im1 in sorted(root.rglob("im1")):
        if not im1.is_dir():
            continue
        leaf = im1.parent
        split = leaf.name.lower()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unrecognized SECOND split directory: {leaf}")
        if split in found:
            raise ValueError(f"Multiple SECOND {split} directories beneath {root}.")
        paths = {
            key: leaf / name
            for key, name in (
                ("before", "im1"),
                ("after", "im2"),
                ("before_semantic", "label1"),
                ("after_semantic", "label2"),
            )
        }
        if split == "test" and not any(
            paths[k].exists() for k in ("before_semantic", "after_semantic")
        ):
            found[split] = []
            continue
        indexes: dict[str, dict[str, str]] = {}
        for key, directory in paths.items():
            if not directory.is_dir():
                raise FileNotFoundError(f"SECOND missing {directory}.")
            index = {}
            for file in sorted(directory.iterdir()):
                if file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                    continue
                if file.stem in index:
                    raise ValueError(f"Duplicate SECOND identifier {file.stem}: {directory}")
                index[file.stem] = str(file.resolve())
            indexes[key] = index
        identifiers = set(indexes["before"])
        if not identifiers or any(set(index) != identifiers for index in indexes.values()):
            raise ValueError(f"Missing/extra or empty SECOND pairs/annotations in {leaf}.")
        found[split] = [
            {
                "sample_id": f"{split}/{identifier}",
                "group_id": identifier,
                "split": split,
                **{k: v[identifier] for k, v in indexes.items()},
            }
            for identifier in sorted(identifiers)
        ]
    if not found.get("train"):
        raise FileNotFoundError(f"No labeled SECOND train/im1,im2,label1,label2 beneath {root}.")
    return found


def bootstrap_second(
    root: str | Path,
    *,
    archive: str | Path | None = None,
    allow_download: bool = False,
) -> Path:
    """Reuse a valid layout or explicitly install a local/official ZIP with safe extraction.

    A private staging directory and .part download never count as completed. Local SHA256
    is provenance, not an author-published checksum. Authentication/quota errors propagate.
    """
    root = Path(root).resolve()
    try:
        discover_second(root)
        return root
    except FileNotFoundError:
        if root.exists() and any(root.iterdir()):
            raise ValueError(
                f"Incomplete SECOND directory: {root}; choose a new installation path."
            ) from None
    if archive is None and not allow_download:
        raise FileNotFoundError(
            f"SECOND absent at {root}; provide local_archive or enable download."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    if archive is None:
        import gdown

        partial = root.parent / f"SECOND-{uuid4().hex}.zip.part"
        try:
            downloaded = gdown.download(
                id="1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u", output=str(partial), quiet=False
            )
        except Exception as error:
            raise RuntimeError(
                f"Official SECOND download failed (access/quota/network): {error}; "
                "provide a local archive or installed folder."
            ) from error
        if downloaded is None or not zipfile.is_zipfile(partial):
            raise RuntimeError(
                f"SECOND download incomplete/not ZIP: {partial}; use a local archive."
            )
        archive = partial.with_suffix("")
        partial.rename(archive)
    archive = Path(archive).resolve()
    stage = root.parent / f".{root.name}.incomplete-{uuid4().hex}"
    stage.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = entry.filename.replace("\\", "/")
            target = (stage / name).resolve()
            if (
                not target.is_relative_to(stage)
                or ":" in name
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe SECOND archive member: {entry.filename}")
        bundle.extractall(stage)
    discover_second(stage)
    write_json(
        stage / "source.json",
        {
            "source": SECOND_SOURCE,
            "download": SECOND_DOWNLOAD,
            "archive": str(archive),
            "local_archive_sha256": file_digest(archive),
            "checksum_is_local": True,
            "complete": True,
        },
    )
    if root.exists():
        root.rmdir()  # Only an empty directory, never recursive removal.
    stage.rename(root)
    return root


def _check_split_seed(split_seed: int) -> None:
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a fixed nonnegative integer.")


def _apply_source_groups(
    scenes: list[dict[str, str]], source_groups: dict[str, str] | None
) -> None:
    """Replace filename groups by supplied source metadata (sample_id -> group); in place."""
    if source_groups is None:
        return
    for scene in scenes:
        if scene["sample_id"] not in source_groups:
            raise ValueError(f"Missing source group: {scene['sample_id']}")
        scene["group_id"] = source_groups[scene["sample_id"]]


def _assign_local_holdout(
    scenes: list[dict[str, str]], fractions: dict[str, float], split_seed: int
) -> None:
    """Move whole official-train groups into the missing splits; in place, seed-determined.

    fractions maps each missing split to its share of train groups. One seeded shuffle of the
    sorted group IDs is cut in mapping order; each split receives max(1, int(n * fraction)).
    """
    groups = sorted({scene["group_id"] for scene in scenes if scene["split"] == "train"})
    if len(groups) < len(fractions) + 1:
        raise ValueError("Too few source groups for disjoint local train/val/test.")
    random.Random(split_seed).shuffle(groups)
    assignments: dict[str, str] = {}
    offset = 0
    for split, fraction in fractions.items():
        count = max(1, int(len(groups) * fraction))
        assignments.update(dict.fromkeys(groups[offset : offset + count], split))
        offset += count
    if offset >= len(groups):
        raise ValueError("Local holdout would leave no training groups; lower the fractions.")
    for scene in scenes:
        if scene["split"] == "train":
            scene["split"] = assignments.get(scene["group_id"], "train")


def _manifest_digest(content: dict[str, Any]) -> str:
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    """Commit JSON atomically on the same filesystem; a failed write leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _holdout_fraction(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
        raise ValueError(f"{name} must be a fraction strictly between 0 and 1: {value!r}")
    return float(value)


def _relative_posix(path: str, root: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def prepare_second_splits(
    root: str | Path,
    destination: str | Path,
    *,
    split_seed: int,
    val_fraction: float | None = None,
    test_fraction: float | None = None,
    allow_missing_test: bool = False,
    source_groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write a persistent SECOND split manifest; no ROI crops, label conversion or model config.

    Official labeled splits are always preserved, and a missing one is an error unless the
    policy is explicit: val_fraction / test_fraction carve that share of official-train groups
    into the split (whole groups, one seeded shuffle, val before test: the prepare_second
    local_holdout rule), allow_missing_test records that no labeled test exists. A fraction for
    a split the release already provides is rejected. Groups are filenames unless source_groups
    (sample_id -> group) is supplied; filename groups claim no geographic independence, while
    supplied groups must not span splits. An existing manifest is never replaced: identical
    content is returned, different content fails. Diagnostics report, but do not repair, shared
    filename stems and byte-identical RGB pairs across splits. Returns the manifest.
    """
    _check_split_seed(split_seed)
    shares = {
        "val": _holdout_fraction("val_fraction", val_fraction),
        "test": _holdout_fraction("test_fraction", test_fraction),
    }
    if sum(share or 0.0 for share in shares.values()) >= 1:
        raise ValueError("val_fraction + test_fraction must leave training groups.")
    if allow_missing_test and shares["test"] is not None:
        raise ValueError("allow_missing_test and test_fraction are mutually exclusive.")
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    splits = discover_second(root)
    policy: dict[str, Any] = {"val_fraction": shares["val"], "test_fraction": shares["test"]}
    fractions: dict[str, float] = {}
    for split, share in shares.items():
        if splits.get(split):
            if share is not None:
                raise ValueError(
                    f"Official labeled {split} exists; {split}_fraction must stay None "
                    "(official splits are never overridden)."
                )
            policy[split] = "official"
        elif share is not None:
            fractions[split] = share
            policy[split] = "local_holdout"
        elif split == "test" and allow_missing_test:
            policy[split] = "unavailable"
        else:
            hint = " or allow_missing_test=True" if split == "test" else ""
            raise ValueError(
                f"SECOND labeled {split} unavailable; pass {split}_fraction explicitly{hint}."
            )
    scenes = [deepcopy(scene) for rows in splits.values() for scene in rows]
    _apply_source_groups(scenes, source_groups)
    origin = {scene["sample_id"]: scene["split"] for scene in scenes}
    if fractions:
        _assign_local_holdout(scenes, fractions, split_seed)
    if source_groups is not None:
        owner: dict[str, str] = {}
        for scene in scenes:
            if owner.setdefault(scene["group_id"], scene["split"]) != scene["split"]:
                raise ValueError(
                    f"Source group {scene['group_id']!r} spans splits {owner[scene['group_id']]!r}"
                    f" and {scene['split']!r}; official splits are not group-disjoint."
                )
    samples: dict[str, dict[str, Any]] = {
        scene["sample_id"]: {
            "split": scene["split"],
            "origin_split": origin[scene["sample_id"]],
            "group_id": scene["group_id"],
            "files": {
                key: {
                    "path": _relative_posix(scene[key], root),
                    "sha256": file_digest(Path(scene[key])),
                }
                for key in SCENE_FILES
            },
        }
        for scene in sorted(scenes, key=lambda item: item["sample_id"])
    }
    members = {
        split: [name for name, sample in samples.items() if sample["split"] == split]
        for split in ("train", "val", "test")
    }
    stem_origins: dict[str, set[str]] = {}
    for sample_id, split in origin.items():
        stem_origins.setdefault(sample_id.split("/", 1)[1], set()).add(split)
    # Byte-identical RGB pairs (either temporal order) under different names are invisible to
    # filename groups; reported, never silently accepted as independent.
    pair_members: dict[tuple[str, ...], list[str]] = {}
    for name, sample in samples.items():
        digests = sorted(sample["files"][key]["sha256"] for key in ("before", "after"))
        pair_members.setdefault(tuple(digests), []).append(name)
    content = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "kind": "second_split_manifest",
        "source": SECOND_SOURCE,
        "split_seed": split_seed,
        "policy": policy,
        "group_basis": "source metadata" if source_groups else "filename only",
        "geographic_independence_verified": False,
        "official_leaves": sorted(splits),
        "official_counts": {split: len(splits.get(split, [])) for split in members},
        "counts": {split: len(ids) for split, ids in members.items()},
        "group_counts": {
            split: len({samples[name]["group_id"] for name in ids})
            for split, ids in members.items()
        },
        "splits": members,
        "samples": samples,
        "diagnostics": {
            "stems_shared_across_official_splits": sorted(
                stem for stem, origins in stem_origins.items() if len(origins) > 1
            ),
            "identical_rgb_pairs_across_splits": sorted(
                sorted(names)
                for names in pair_members.values()
                if len({samples[name]["split"] for name in names}) > 1
            ),
        },
    }
    manifest = {**content, "manifest_sha256": _manifest_digest(content)}
    if destination.exists():
        existing = read_second_split_manifest(destination)
        if existing["manifest_sha256"] != manifest["manifest_sha256"]:
            raise FileExistsError(
                f"A different SECOND split manifest exists: {destination}; manifests are "
                "frozen, so preserve it and choose another destination."
            )
        return existing
    _write_json_atomic(destination, manifest)
    return manifest


def read_second_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load a split manifest and verify its schema and content digest."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != "second_split_manifest"
        or manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA
    ):
        raise ValueError(f"Not a supported SECOND split manifest: {path}")
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _manifest_digest(content):
        raise ValueError(f"SECOND split manifest digest mismatch (edited or corrupted): {path}")
    return manifest
