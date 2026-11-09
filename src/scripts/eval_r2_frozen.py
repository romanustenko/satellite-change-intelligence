"""Frozen-system evaluation of Research 2 systems A and B on SECOND val or test (R2-P11-T06).

Everything that influences predictions comes from the freeze manifest (``semantic.freeze``): the
checkpoints (re-hashed), the ROI policy and the detector preprocessing. Nothing is selected here.
``--split val`` is a rehearsal of the identical code path on the validation split. ``--split test``
additionally requires a verified manifest (clean tree, code unchanged since the freeze) and
atomically claims the single test read (``<manifest>.test_read.json``, D-007); a second claim fails.

Manifest layout expected by this script (``systems`` -> name -> ``config``):

* ``A``: ``{"checkpoint_by_seed": {"42": ".../best.pt", ...}}``
* ``B``: ``{"detector_by_seed": {...}, "what_by_seed": {...}, "roi_settings": {...},
  "detector_input_size": 512, "detector_output_kind": "logits"}``

Outputs (new directory): ``scene_results.npz`` (per-seed per-scene histograms and densities) and
``run.json``. Statistics are computed afterwards by ``analyze_r2_frozen.py`` from those files only.

    python src/scripts/eval_r2_frozen.py --manifest <m.json> --split val --gpu 1 --out-dir <dir>
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.execution import resolve_device
from satchange.run import reconstruct_checkpoint
from satchange.semantic.freeze import claim_test_read, verify_freeze_manifest
from satchange.semantic.inference import detector_probability_map
from satchange.semantic.roi import ROISettings
from satchange.semantic.system_eval import (
    evaluate_system_a,
    evaluate_system_b,
    load_semantic_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "SECOND"


def _listed(manifest: dict[str, Any], system: str, mapping: dict[str, str]) -> dict[str, Path]:
    """Seed -> path, requiring every path to be a hashed checkpoint of that manifest system."""
    frozen = {Path(item["path"]).resolve() for item in manifest["systems"][system]["checkpoints"]}
    paths = {seed: Path(path) for seed, path in mapping.items()}
    for path in paths.values():
        if path.resolve() not in frozen:
            raise ValueError(f"{path} is not a frozen checkpoint of system {system}.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="debug (val only): first N scenes")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if args.split == "test" and args.limit:
        raise ValueError("--limit is a validation-only debugging option.")

    manifest = verify_freeze_manifest(args.manifest, root=ROOT)
    systems = manifest["systems"]
    config_a, config_b = systems["A"]["config"], systems["B"]["config"]
    a_paths = _listed(manifest, "A", config_a["checkpoint_by_seed"])
    detectors = _listed(manifest, "B", config_b["detector_by_seed"])
    whats = _listed(manifest, "B", config_b["what_by_seed"])
    seeds = sorted(a_paths, key=int)
    if sorted(detectors, key=int) != seeds or sorted(whats, key=int) != seeds:
        raise ValueError("A and B must be frozen for the same seeds.")
    settings = ROISettings(**config_b["roi_settings"])
    side = int(config_b["detector_input_size"])
    input_size = (side, side)
    if not seeds:
        raise ValueError("The manifest freezes no seeds.")

    if args.split == "test":
        if not (DATA / "test").is_dir():
            raise FileNotFoundError(f"{DATA / 'test'} is not extracted.")
        claim_test_read(args.manifest, system_set="A+B")  # before any test sample is read
    device = resolve_device(f"cuda:{args.gpu}")
    dataset, _ = create_dataset_dataloader(DATA, args.split, None, 1, 0, semantic_labels=True)
    started = time.perf_counter()
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("a", "b", "b_inside", "b_outside", "oracle")
    }
    coverage: dict[str, list[Any]] = {}
    sample_ids: list[str] = []
    density = np.zeros(0)
    for seed in seeds:
        print(f"seed {seed}: system A", flush=True)
        a_model = load_semantic_checkpoint(a_paths[seed], device)
        a = evaluate_system_a(a_model, dataset, device, limit=args.limit)
        del a_model
        detector, _ = reconstruct_checkpoint(detectors[seed], device)
        what = load_semantic_checkpoint(whats[seed], device)

        def source(sample: dict[str, Any], model: torch.nn.Module = detector) -> torch.Tensor:
            return detector_probability_map(
                model, sample["before"], sample["after"], input_size=input_size,
                output_kind=config_b["detector_output_kind"],
            )

        print(f"seed {seed}: system B (detector ROIs)", flush=True)
        b = evaluate_system_b(what, dataset, device, source, settings=settings, limit=args.limit)
        print(f"seed {seed}: system B (oracle ROIs)", flush=True)
        oracle = evaluate_system_b(what, dataset, device, None, settings=settings,
                                   limit=args.limit)
        del detector, what
        torch.cuda.empty_cache()
        if not (a["sample_ids"] == b["sample_ids"] == oracle["sample_ids"]):
            raise RuntimeError("Scene order differs between systems.")
        arrays["a"].append(a["histograms"])
        arrays["b"].append(b["histograms"])
        arrays["b_inside"].append(b["histograms_inside_roi"])
        arrays["b_outside"].append(b["histograms_outside_roi"])
        arrays["oracle"].append(oracle["histograms"])
        coverage[seed] = b["coverage"]
        sample_ids, density = a["sample_ids"], a["density"]

    args.out_dir.mkdir(parents=True)
    saved: dict[str, Any] = {f"hist_{key}": np.stack(value) for key, value in arrays.items()}
    np.savez_compressed(
        args.out_dir / "scene_results.npz", seeds=np.array([int(s) for s in seeds]),
        sample_ids=np.array(sample_ids), density=density, **saved,
    )
    (args.out_dir / "run.json").write_text(json.dumps({
        "split": args.split, "scenes": len(sample_ids), "seeds": [int(s) for s in seeds],
        "manifest": str(args.manifest), "manifest_git_revision": manifest["git_revision"],
        "roi_settings": config_b["roi_settings"], "coverage_per_scene_by_seed": coverage,
        "gpu": args.gpu, "wall_hours": (time.perf_counter() - started) / 3600,
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }, indent=2))
    print(f"saved {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
