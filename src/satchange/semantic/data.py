"""Manifest-based semantic supervision; binary LEVIR masks cannot supply these labels."""

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from satchange.data.transforms import RandomFlipRotate
from satchange.models.tiny_siamese_change_classifier import validate_classes
from satchange.semantic.roi import ROISettings, extract_rois

NORMALIZATION = "RGB uint8 / 255; native resolution"


def write_json(path: Path, value: Any) -> None:
    """Save readable JSON with finite numeric values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_digest(path: Path) -> str:
    """Hash source content so a resume cannot silently use changed annotations/images."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: str | Path) -> torch.Tensor:
    """Load an actual RGB uint8 image as native [3,H,W] float32 in [0,1]."""
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB imagery, not {image.mode}: {path}")
        pixels = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255).contiguous()


def read_map(path: str | Path, *, semantic: bool = False) -> torch.Tensor:
    """Read a non-pickled .npy [H,W] map, preserving integer semantic IDs."""
    if Path(path).suffix.lower() != ".npy":
        raise ValueError(f"Maps must be .npy [H,W] arrays: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"Map must have shape [H,W]: {path}")
    if semantic and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Semantic maps require integer IDs: {path}")
    if not semantic and not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"Change maps require numeric probabilities: {path}")
    if not semantic and (
        np.iscomplexobj(array)
        or not np.isfinite(array).all()
        or (array < 0).any()
        or (array > 1).any()
    ):
        raise ValueError(f"Change probabilities must be finite and in [0,1]: {path}")
    if semantic and np.issubdtype(array.dtype, np.unsignedinteger):
        if array.size and array.max() > np.iinfo(np.int64).max:
            raise ValueError(f"Semantic IDs must fit signed int64: {path}")
    # int64 also allows explicit negative ignore IDs without binarizing labels.
    return torch.from_numpy(array.astype(np.int64 if semantic else np.float32, copy=True))


def data_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the single explicit taxonomy, ROI, annotation, and mask-source contract."""
    classes = validate_classes(config.get("classes", []))
    mapping = config.get("class_ids")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("class_ids must explicitly map raw integer IDs to class names.")
    ids: dict[str, int] = {}
    for raw, name in mapping.items():
        if str(int(raw)) != str(raw) or name not in classes:
            raise ValueError("class_ids keys must be integer IDs and values declared class names.")
        ids[str(raw)] = classes.index(name)
    if set(ids.values()) != set(range(len(classes))):
        raise ValueError("Every declared class needs an explicit raw-ID mapping.")
    ignore_ids = config.get("ignore_ids")
    if not isinstance(ignore_ids, list) or any(type(value) is not int for value in ignore_ids):
        raise ValueError("ignore_ids must explicitly list integer unlabeled/background IDs.")
    if set(map(str, ignore_ids)) & ids.keys():
        raise ValueError("An ID cannot be both a learned class and ignored.")
    source = config.get("mask_source")
    if source not in ("ground_truth", "predicted"):
        raise ValueError("mask_source must be 'ground_truth' or 'predicted'.")
    settings = ROISettings(**config.get("roi", {}))
    policy = {"min_transition_purity": 0.8, "min_valid_fraction": 0.8}
    supplied = config.get("label_policy", {})
    if set(supplied) - policy.keys():
        raise ValueError("Unknown label_policy key.")
    policy.update(supplied)
    if any(not 0 < value <= 1 for value in policy.values()):
        raise ValueError("Label purity and valid-fraction thresholds must be in (0,1].")
    detector = config.get("detector")
    if source == "predicted":
        if not isinstance(detector, dict) or not detector.get("identity"):
            raise ValueError(
                "Predicted masks need detector identity and training_groups provenance."
            )
        groups = detector.get("training_groups")
        if not isinstance(groups, list) or any(not isinstance(group, str) for group in groups):
            raise ValueError(
                "detector.training_groups must list the frozen detector's fitting groups."
            )
    elif detector is not None:
        raise ValueError("ground_truth experiments must set detector: null.")
    return {
        "classes": list(classes),
        "class_ids": ids,
        "ignore_ids": sorted(set(ignore_ids)),
        "roi": asdict(settings),
        "label_policy": policy,
        "mask_source": source,
        "detector": detector,
        "normalization": NORMALIZATION,
    }


def read_manifest(path: str | Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate scene splits, unique IDs, source files, and detector fitting-group provenance."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Semantic manifest missing: {path}. Supply real before/after semantic annotations; "
            "LEVIR-CD binary labels are insufficient. See docs/semantic_classifier.md."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    scenes = document.get("scenes") if isinstance(document, dict) else None
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Manifest must contain a nonempty 'scenes' list.")
    group_splits: dict[str, str] = {}
    ids: set[str] = set()
    result = []
    pair_splits: dict[tuple[str, ...], str] = {}
    content_splits: dict[tuple[str, ...], str] = {}
    rgb_hashes: dict[str, str] = {}
    fields = ("before", "after", "before_semantic", "after_semantic", "change")
    for scene in scenes:
        required = set(fields + ("sample_id", "group_id", "split"))
        if not isinstance(scene, dict) or required - scene.keys():
            raise ValueError("Each scene needs RGB, both semantic labels, change, split, and IDs.")
        sample_id, group, split = (scene[key] for key in ("sample_id", "group_id", "split"))
        if not all(isinstance(value, str) and value for value in (sample_id, group, split)):
            raise ValueError("sample_id, group_id, and split must be nonempty strings.")
        if sample_id in ids or split not in ("train", "val", "test"):
            raise ValueError("sample_id must be unique and split must be train, val, or test.")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Cross-split group leakage: {group}")
        if split != "train" and group in (protocol.get("detector") or {}).get(
            "training_groups", []
        ):
            raise ValueError(f"Detector fitting-group leakage into {split}: {group}")
        ids.add(sample_id)
        group_splits[group] = split
        resolved = dict(scene)
        for field in fields:
            if not isinstance(scene[field], str) or not scene[field]:
                raise ValueError(
                    f"{sample_id}: missing {field} path; no fabricated-label fallback."
                )
            source = (path.parent / scene[field]).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"{sample_id}: missing {field}: {source}")
            resolved[field] = str(source)
        pair = tuple(
            sorted((os.path.normcase(resolved["before"]), os.path.normcase(resolved["after"])))
        )
        for source in (resolved["before"], resolved["after"]):
            if source not in rgb_hashes:
                rgb_hashes[source] = file_digest(Path(source))
        content = tuple(sorted((rgb_hashes[resolved["before"]], rgb_hashes[resolved["after"]])))
        for identity, seen in ((pair, pair_splits), (content, content_splits)):
            if identity in seen and seen[identity] != split:
                raise ValueError(
                    f"Cross-split RGB pair leakage (including temporal reversal): {sample_id}"
                )
            seen[identity] = split
        resolved["rgb_hashes"] = {key: rgb_hashes[resolved[key]] for key in ("before", "after")}
        result.append(resolved)
    return result


def dominant_transition(
    before: torch.Tensor,
    after: torch.Tensor,
    support: torch.Tensor,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Select an observed joint label pair only within component support; report exclusions."""
    if before.shape != after.shape or support.shape != before.shape:
        raise ValueError("Semantic maps and component support must align as [H,W].")
    if before.ndim != 2 or support.dtype != torch.bool:
        raise ValueError("Component support must be boolean [H,W].")
    integer_types = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    if before.dtype not in integer_types or after.dtype not in integer_types:
        raise ValueError("Semantic maps require integer class IDs.")
    mapped = []
    for labels in (before, after):
        values = torch.full_like(labels, -1, dtype=torch.long)
        known = torch.zeros_like(labels, dtype=torch.bool)
        for raw, target in protocol["class_ids"].items():
            selected = labels == int(raw)
            values[selected] = target
            known |= selected
        for raw in protocol["ignore_ids"]:
            known |= labels == raw
        if not known.all():
            raise ValueError(
                "Semantic map contains undeclared class IDs; set class_ids/ignore_ids."
            )
        mapped.append(values)
    valid = support.bool() & (mapped[0] >= 0) & (mapped[1] >= 0)
    area, valid_count = int(support.sum()), int(valid.sum())
    fraction = valid_count / area if area else 0.0
    quality: dict[str, Any] = {"valid_pixels": valid_count, "valid_fraction": fraction}
    if not valid_count or fraction < protocol["label_policy"]["min_valid_fraction"]:
        return {**quality, "reason": "insufficient_annotation"}
    pairs = torch.stack((mapped[0][valid], mapped[1][valid]), dim=1)
    unique, counts = torch.unique(pairs, dim=0, return_counts=True)
    largest = int(counts.max())
    purity = largest / valid_count
    quality["transition_purity"] = purity
    if int((counts == largest).sum()) != 1:
        return {**quality, "reason": "tied_transition"}
    if purity < protocol["label_policy"]["min_transition_purity"]:
        return {**quality, "reason": "mixed_transition"}
    pair = unique[int(counts.argmax())].tolist()
    return {**quality, "reason": None, "before_label": pair[0], "after_label": pair[1]}


def prepare_dataset(
    config: dict[str, Any],
    manifest: str | Path,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Validate real sources and generate ROI records/masks, without duplicating RGB imagery.

    destination=None validates and reports without writing. An existing output is never replaced.
    Coverage is annotation-eligible ROI coverage, not detection matching or end-to-end accuracy.
    """
    protocol = data_protocol(config)
    scenes = read_manifest(manifest, protocol)
    settings = ROISettings(**protocol["roi"])
    output = Path(destination).resolve() if destination is not None else None
    if output is not None and output.exists():
        raise FileExistsError(f"Preparation output already exists: {output}")
    # A failed preparation never produces a completed prepared.json at the requested path.
    final_output = output
    if output is not None:
        from uuid import uuid4

        output = output.with_name(f".{output.name}.incomplete-{uuid4().hex}")
    records = []
    reports: dict[str, Any] = {
        split: {
            "proposed": 0,
            "retained": 0,
            "exclusions": Counter(),
            "before_distribution": Counter(),
            "after_distribution": Counter(),
            "transition_distribution": Counter(),
            "scenes_without_components": 0,
            "crop_sizes": [],
        }
        for split in ("train", "val", "test")
    }
    hashes = {}
    for scene in scenes:
        for key in ("before", "after", "before_semantic", "after_semantic", "change"):
            source = scene[key]
            if source not in hashes:
                hashes[source] = (
                    scene["rgb_hashes"][key]
                    if key in ("before", "after")
                    else file_digest(Path(source))
                )
        before, after = read_rgb(scene["before"]), read_rgb(scene["after"])
        probabilities = read_map(scene["change"]).unsqueeze(0)
        if protocol["mask_source"] == "ground_truth":
            if not ((probabilities == 0) | (probabilities == 1)).all():
                raise ValueError("ground_truth change maps must contain binary 0/1 support.")
        semantic_before = read_map(scene["before_semantic"], semantic=True)
        semantic_after = read_map(scene["after_semantic"], semantic=True)
        if semantic_before.shape != before.shape[-2:] or semantic_after.shape != before.shape[-2:]:
            raise ValueError(f"{scene['sample_id']}: native semantic/RGB spatial mismatch.")
        # Validate the whole taxonomy even when no component is proposed.
        dominant_transition(
            semantic_before,
            semantic_after,
            torch.zeros_like(semantic_before, dtype=torch.bool),
            protocol,
        )
        rois = extract_rois(before, after, probabilities, replace(settings, min_component_area=1))
        report = reports[scene["split"]]
        report["scenes_without_components"] += int(not rois)
        for roi in rois:
            report["proposed"] += 1
            x0, y0, x1, y1 = roi.crop_bbox
            quality = dominant_transition(
                semantic_before[y0:y1, x0:x1],
                semantic_after[y0:y1, x0:x1],
                roi.support[0],
                protocol,
            )
            reason = (
                "small_component" if roi.area < settings.min_component_area else quality["reason"]
            )
            if reason:
                report["exclusions"][reason] += 1
                records.append(
                    {
                        **scene,
                        "component_id": roi.component_id,
                        "crop_bbox": list(roi.crop_bbox),
                        "area": roi.area,
                        "component_bbox": list(roi.component_bbox),
                        "retained": False,
                        "minimum_extent_limited": roi.minimum_extent_limited,
                        "quality": {**quality, "reason": reason},
                    }
                )
                continue
            mask_name = f"masks/roi_{len(records):08d}.npz"
            if output is not None:
                (output / "masks").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output / mask_name,
                    probabilities=roi.probabilities.numpy(),
                    support=roi.support.numpy(),
                )
            record = {
                **scene,
                "component_id": roi.component_id,
                "crop_bbox": list(roi.crop_bbox),
                "component_bbox": list(roi.component_bbox),
                "area": roi.area,
                "mask_file": mask_name,
                "retained": True,
                "quality": quality,
                "minimum_extent_limited": roi.minimum_extent_limited,
                "before_label": quality["before_label"],
                "after_label": quality["after_label"],
            }
            records.append(record)
            report["retained"] += 1
            report["crop_sizes"].append([y1 - y0, x1 - x0])
            left, right = (
                protocol["classes"][record[key]] for key in ("before_label", "after_label")
            )
            report["before_distribution"][left] += 1
            report["after_distribution"][right] += 1
            report["transition_distribution"][f"{left} -> {right}"] += 1
    for report in reports.values():
        report["retained_fraction"] = (
            report["retained"] / report["proposed"] if report["proposed"] else 0.0
        )
    provenance = {
        "manifest": str(Path(manifest).resolve()),
        "scenes": scenes,
        "source_hashes": hashes,
    }
    document = {
        "schema_version": 1,
        "protocol": protocol,
        "provenance": provenance,
        "records": records,
        "coverage": reports,
        "coverage_definition": "annotation-eligible / proposed connected components",
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        document["mask_hashes"] = {
            record["mask_file"]: file_digest(output / record["mask_file"])
            for record in records
            if record["retained"]
        }
        write_json(output / "prepared.json", document)
        assert final_output is not None
        output.rename(final_output)
    return document


class SemanticROIDataset(Dataset):
    """Load variable-size real RGB crops and component weights from prepared records."""

    def __init__(
        self,
        prepared: str | Path,
        split: str,
        config: dict[str, Any],
        *,
        augment: bool = False,
    ) -> None:
        self.path = Path(prepared).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Prepared semantic data missing: {self.path}. Run semantic prepare with "
                "real before/after semantic annotations; no synthetic-label fallback is provided."
            )
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        if self.document.get("schema_version") != 1:
            raise ValueError("Unsupported prepared-data schema.")
        if self.document["protocol"] != data_protocol(config):
            raise ValueError(
                "Prepared data protocol differs from configuration; prepare a new dataset."
            )
        if split not in ("train", "val", "test") or (augment and split != "train"):
            raise ValueError("Use train/val/test and augmentation only on train.")
        self.records = [
            row for row in self.document["records"] if row["retained"] and row["split"] == split
        ]
        if not self.records:
            raise ValueError(f"No eligible semantic ROIs in {split}; supply annotated real data.")
        self.transform = RandomFlipRotate() if augment else None
        self.split = split

    def verify_sources(self) -> None:
        """Check input and generated-mask identities once before a run/evaluation."""
        for source, expected in self.document["provenance"]["source_hashes"].items():
            if file_digest(Path(source)) != expected:
                raise ValueError(f"Prepared source changed: {source}; regenerate prepared data.")
        for source, expected in self.document["mask_hashes"].items():
            if file_digest(self.path.parent / source) != expected:
                raise ValueError(f"Prepared ROI mask changed: {source}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        x0, y0, x1, y1 = row["crop_bbox"]
        with np.load(self.path.parent / row["mask_file"], allow_pickle=False) as archive:
            mask = torch.from_numpy((archive["probabilities"] * archive["support"]).copy()).float()
        sample = {
            "before": read_rgb(row["before"])[:, y0:y1, x0:x1].contiguous(),
            "after": read_rgb(row["after"])[:, y0:y1, x0:x1].contiguous(),
            "mask": mask.contiguous(),
            "sample_id": row["sample_id"],
            "before_label": row["before_label"],
            "after_label": row["after_label"],
        }
        return self.transform(sample) if self.transform is not None else sample


def collate_rois(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve different native crop sizes; never resize or pad logical batches."""
    return samples
