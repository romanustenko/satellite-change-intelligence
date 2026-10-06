"""Exact SECOND conversion and versioned preparation; never starts training on import."""

import json
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

from satchange.semantic.data import (
    SemanticROIDataset,
    data_protocol,
    file_digest,
    prepare_dataset,
    read_rgb,
    write_json,
)

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


def prepare_second(
    root: str | Path,
    cache: str | Path,
    config: dict[str, Any],
    *,
    split_seed: int = 2099,
    local_holdout: bool = False,
    indexed_ids: dict[int, str] | None = None,
    source_groups: dict[str, str] | None = None,
    sensor_metadata: dict[str, Any] | None = None,
) -> Path:
    """Convert native labels once and reuse only a verified complete content/protocol cache.

    Preserve provided labeled splits. Missing val/test needs explicit local_holdout; groups
    are source metadata when supplied, otherwise filenames, with no geographic-independence
    claim. Training seeds never alter splits. Native SECOND images must be 512x512.
    """
    protocol = data_protocol(config)
    if protocol["classes"] != list(SECOND_CLASSES) or protocol["mask_source"] != "ground_truth":
        raise ValueError(
            "SECOND oracle preparation requires the six documented classes and GT masks."
        )
    expected = {str(i + 1): i for i in range(6)}
    if protocol["class_ids"] != expected or protocol["ignore_ids"] != [0]:
        raise ValueError("SECOND conversion uses canonical IDs 1..6 with non-change ID 0 ignored.")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a fixed nonnegative integer.")
    splits = discover_second(root)
    scenes = [deepcopy(scene) for rows in splits.values() for scene in rows]
    for scene in scenes:
        if source_groups is not None:
            if scene["sample_id"] not in source_groups:
                raise ValueError(f"Missing source group: {scene['sample_id']}")
            scene["group_id"] = source_groups[scene["sample_id"]]
    missing = [split for split in ("val", "test") if not splits.get(split)]
    if missing:
        if not local_holdout:
            raise ValueError(
                f"SECOND labeled {missing} unavailable; explicitly enable local_holdout."
            )
        groups = sorted({scene["group_id"] for scene in scenes if scene["split"] == "train"})
        if len(groups) < len(missing) + 1:
            raise ValueError("Too few source groups for disjoint local train/val/test.")
        random.Random(split_seed).shuffle(groups)
        count = max(1, int(len(groups) * 0.15))
        assignments = {}
        for index, split in enumerate(missing):
            assignments.update(dict.fromkeys(groups[index * count : (index + 1) * count], split))
        for scene in scenes:
            if scene["split"] == "train":
                scene["split"] = assignments.get(scene["group_id"], "train")
    hashes = {
        scene[key]: file_digest(Path(scene[key]))
        for scene in scenes
        for key in ("before", "after", "before_semantic", "after_semantic")
    }
    identity = {
        "adapter_schema": 1,
        "protocol": protocol,
        "sources": hashes,
        "scenes": scenes,
        "split_seed": split_seed,
        "local_splits": missing,
        "indexed_ids": indexed_ids,
        "sensor_metadata": sensor_metadata,
        "palette_source": PALETTE_SOURCE,
    }
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache).resolve()
    destination = cache / key
    marker = destination / "complete.json"
    prepared = destination / "rois/prepared.json"
    if marker.is_file():
        completion = json.loads(marker.read_text(encoding="utf-8"))
        if completion.get("identity") != key or completion.get("prepared_sha256") != file_digest(
            prepared
        ):
            raise ValueError(f"SECOND cache completion identity corrupted: {destination}")
        SemanticROIDataset(prepared, "train", config).verify_sources()
        return prepared
    if destination.exists():
        raise ValueError(
            f"Incomplete SECOND cache: {destination}; preserve it and choose another cache."
        )
    cache.mkdir(parents=True, exist_ok=True)
    stage = cache / f".{key}.incomplete-{uuid4().hex}"
    (stage / "maps").mkdir(parents=True)
    converted = []
    for index, scene in enumerate(scenes):
        left, right = (
            decode_label(scene[k], indexed_ids) for k in ("before_semantic", "after_semantic")
        )
        if left.shape != (512, 512) or right.shape != left.shape:
            raise ValueError(f"SECOND requires native 512x512 labels: {scene['sample_id']}")
        if any(read_rgb(scene[k]).shape != (3, 512, 512) for k in ("before", "after")):
            raise ValueError(f"SECOND requires aligned native 512x512 RGB: {scene['sample_id']}")
        if not np.array_equal(left > 0, right > 0):
            raise ValueError(f"Inconsistent before/after change support: {scene['sample_id']}")
        row = dict(scene)
        for field, array in (
            ("before_semantic", left),
            ("after_semantic", right),
            ("change", (left > 0).astype(np.float32)),
        ):
            filename = f"maps/{index:06d}_{field}.npy"
            np.save(stage / filename, array, allow_pickle=False)
            # Final absolute paths survive the atomic staging rename.
            row[field] = str(destination / filename)
        converted.append(row)
    # Build against staging paths, then publish source paths to the final immutable directory.
    staging_scenes = [
        {
            **row,
            **{
                k: str(stage / Path(row[k]).relative_to(destination))
                for k in ("before_semantic", "after_semantic", "change")
            },
        }
        for row in converted
    ]
    write_json(stage / "manifest.json", {"scenes": staging_scenes})
    doc = prepare_dataset(config, stage / "manifest.json", stage / "rois")
    encoded = json.dumps(doc).replace(
        str(stage).replace("\\", "\\\\"), str(destination).replace("\\", "\\\\")
    )
    doc = json.loads(encoded)
    doc["second"] = {
        **identity,
        "source": SECOND_SOURCE,
        "checksum_is_local": True,
        "group_basis": "source metadata" if source_groups else "filename only",
        "geographic_independence_verified": False,
        "change_support": "both labels non-white, checked equal; same-class retained",
    }
    write_json(stage / "rois/prepared.json", doc)
    write_json(stage / "manifest.json", {"scenes": converted})
    write_json(
        stage / "complete.json",
        {
            "identity": key,
            "prepared_sha256": file_digest(stage / "rois/prepared.json"),
            "complete": True,
        },
    )
    stage.rename(destination)
    return prepared
