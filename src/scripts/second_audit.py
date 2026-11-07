"""Read-only SECOND audit (R2-P05-T01): label sanity, duplicates, leakage candidates, groups.

Usage: python src/scripts/second_audit.py data/SECOND docs/research2/second_audit.json

Writes a small JSON report. ``source_groups`` in the report (sample_id -> group) merges only
*within-train* near-duplicate / adjacent-edge candidates so a locally carved validation split
keeps them together; cross-split candidates are reported, never repaired (official splits stay).
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from satchange.semantic.second import SECOND_CLASSES, decode_label

SPLITS = ("train", "test")
THUMB_SIDE = 16
NEAR_DUPLICATE_THUMB_SIMILARITY = 0.95
NEAR_DUPLICATE_MAX_MAD = 5.0  # mean absolute pixel difference of full-resolution T1
EDGE_MAX_MAD = 3.0  # boundary strip mean abs difference (0-255 scale)
EDGE_MAX_RATIO = 0.5  # best edge match must be < half the second best (unambiguous)


def _features(job: tuple[str, str, str]) -> dict[str, Any]:
    root, split, sample = job
    base = Path(root) / split
    before = np.asarray(Image.open(base / "im1" / f"{sample}.png").convert("RGB"))
    after = np.asarray(Image.open(base / "im2" / f"{sample}.png").convert("RGB"))
    label1 = decode_label(base / "label1" / f"{sample}.png")
    label2 = decode_label(base / "label2" / f"{sample}.png")
    small = Image.fromarray(before).convert("L")
    small = small.resize((THUMB_SIDE, THUMB_SIDE), Image.Resampling.BILINEAR)
    thumb = np.asarray(small, dtype=np.float32).ravel()
    return {
        "split": split,
        "sample": sample,
        "support_equal": bool(np.array_equal(label1 > 0, label2 > 0)),
        "change_fraction": float((label1 > 0).mean()),
        "hist1": np.bincount(label1.ravel(), minlength=7),
        "hist2": np.bincount(label2.ravel(), minlength=7),
        "md5_before": hashlib.md5(before.tobytes()).hexdigest(),
        "md5_after": hashlib.md5(after.tobytes()).hexdigest(),
        "thumb": thumb,
        "edges": np.stack(
            [before[0].mean(1), before[-1].mean(1), before[:, 0].mean(1), before[:, -1].mean(1)]
        ).astype(np.float32),
    }


def _best_edge_matches(source: torch.Tensor, target: torch.Tensor) -> tuple[Any, Any, Any]:
    distance = torch.cdist(source, target, p=1) / source.shape[1]
    distance.fill_diagonal_(1e9)
    values, index = distance.topk(2, largest=False, dim=1)
    return values[:, 0].numpy(), values[:, 1].numpy(), index[:, 0].numpy()


def _share(histogram: np.ndarray) -> dict[str, float]:
    values = np.round(histogram / histogram.sum(), 5).tolist()
    return dict(zip(("no_change", *SECOND_CLASSES), values, strict=True))


def audit(root: Path, workers: int = 8) -> dict[str, Any]:
    jobs = [
        (str(root), split, path.stem)
        for split in SPLITS
        for path in sorted((root / split / "im1").iterdir())
    ]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_features, jobs, chunksize=32))
    names = [f"{row['split']}/{row['sample']}" for row in rows]
    split_of = [row["split"] for row in rows]

    summary: dict[str, Any] = {}
    for split in SPLITS:
        part = [row for row in rows if row["split"] == split]
        hist1 = sum(row["hist1"] for row in part)
        hist2 = sum(row["hist2"] for row in part)
        summary[split] = {
            "tiles": len(part),
            "change_support_mismatch": sum(not row["support_equal"] for row in part),
            "tiles_without_change": sum(row["change_fraction"] == 0 for row in part),
            "mean_change_fraction": round(float(np.mean([r["change_fraction"] for r in part])), 4),
            "label1_pixel_share": _share(hist1),
            "label2_pixel_share": _share(hist2),
            "tiles_with_class": {
                name: int(sum((r["hist1"][i + 1] > 0) or (r["hist2"][i + 1] > 0) for r in part))
                for i, name in enumerate(SECOND_CLASSES)
            },
        }

    exact: dict[str, list[list[str]]] = {}
    for key in ("md5_before", "md5_after"):
        groups: dict[str, list[str]] = {}
        for name, row in zip(names, rows, strict=True):
            groups.setdefault(row[key], []).append(name)
        exact[key] = sorted([sorted(v) for v in groups.values() if len(v) > 1])

    thumbs = np.stack([row["thumb"] for row in rows])
    thumbs = thumbs - thumbs.mean(1, keepdims=True)
    thumbs /= np.linalg.norm(thumbs, axis=1, keepdims=True) + 1e-6
    similarity = thumbs @ thumbs.T
    np.fill_diagonal(similarity, -1)

    def full_mad(a: int, b: int) -> float:
        def load(name: str) -> np.ndarray:
            split, sample = name.split("/")
            image = Image.open(root / split / "im1" / f"{sample}.png").convert("RGB")
            return np.asarray(image, dtype=np.float32)

        return float(np.abs(load(names[a]) - load(names[b])).mean())

    pairs: list[dict[str, Any]] = []
    for a, b in np.argwhere(np.triu(similarity > NEAR_DUPLICATE_THUMB_SIMILARITY)):
        mad = full_mad(int(a), int(b))
        if mad < NEAR_DUPLICATE_MAX_MAD:
            pairs.append(
                {"kind": "near_duplicate", "a": names[a], "b": names[b], "t1_mad": round(mad, 3)}
            )

    edges = torch.tensor(np.stack([row["edges"] for row in rows]))
    for kind, source, target in (
        ("adjacent_bottom_top", edges[:, 1], edges[:, 0]),
        ("adjacent_right_left", edges[:, 3], edges[:, 2]),
    ):
        best, second, index = _best_edge_matches(source, target)
        for a in np.where((best < EDGE_MAX_MAD) & (best < EDGE_MAX_RATIO * second))[0]:
            match = {"kind": kind, "a": names[a], "b": names[index[a]]}
            pairs.append({**match, "edge_mad": round(float(best[a]), 3)})

    parent = {name: name for name in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cross, within_train = [], []
    for pair in pairs:
        a_split, b_split = pair["a"].split("/")[0], pair["b"].split("/")[0]
        if a_split == b_split == "train":
            within_train.append(pair)
            parent[find(pair["a"])] = find(pair["b"])
        elif a_split != b_split:
            cross.append(pair)
    members: dict[str, list[str]] = {}
    for name in names:
        members.setdefault(find(name), []).append(name)
    return {
        "kind": "second_audit",
        "thresholds": {
            "near_duplicate_thumb_similarity": NEAR_DUPLICATE_THUMB_SIMILARITY,
            "near_duplicate_max_mad": NEAR_DUPLICATE_MAX_MAD,
            "edge_max_mad": EDGE_MAX_MAD,
            "edge_max_ratio": EDGE_MAX_RATIO,
        },
        "summary": summary,
        "exact_duplicate_groups": exact,
        "candidates_within_train": within_train,
        "candidates_across_splits": cross,
        "merged_groups": sorted(
            sorted(g) for g in members.values() if len(g) > 1
        ),
        "split_of_sample": dict(zip(names, split_of, strict=True)),
    }


def source_groups_from_report(report: dict[str, Any], sample_ids: list[str]) -> dict[str, str]:
    """Full sample_id -> group map for prepare_second_splits: filename stem unless merged."""
    groups = {name: name.split("/")[1] for name in sample_ids}
    for merged in report["merged_groups"]:
        label = "merged:" + "+".join(sorted(name.split("/")[1] for name in merged))
        groups.update(dict.fromkeys(merged, label))
    return groups


def main() -> None:
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    report = audit(root)
    report.pop("split_of_sample")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("exact_duplicate_groups", "candidates_within_train",
                                    "candidates_across_splits", "merged_groups")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
