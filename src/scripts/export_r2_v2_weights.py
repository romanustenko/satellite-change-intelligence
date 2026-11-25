"""Export the weights of every frozen Research 2 network as small, weights-only files.

Reads the freeze manifest (``artifacts/research2/v2/v9/freeze_manifest_v2.json``), checks the
SHA-256 of each frozen ``best.pt`` against it, and writes one file per network with exactly the
tensors the evaluation loads (SCD models: ``model_state_dict``, as in
``semantic.system_eval.load_semantic_checkpoint``; detectors: ``best_model_state_dict`` or
``model_state_dict``, as in ``eval_r2_v2_detector_screening._load_detector``), the model name and
constructor parameters. Optimizer state, RNG state and absolute paths are dropped. Every exported
tensor is compared bitwise with the source. The files load with ``torch.load(weights_only=True)``.

Usage: python src/scripts/export_r2_v2_weights.py --out <dir>
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts/research2/v2/v9/freeze_manifest_v2.json"
FORMAT = "satchange-weights-v1"
# run directory (relative to artifacts/research2/v2/runs) -> published name prefix and role
ROLES = {
    "final/siam_unet_resnet18": ("scd_siam_unet_resnet18", "semantic"),
    "final/scannet": ("scd_scannet", "semantic"),
    "final/mamba_scd": ("scd_mamba_scd", "semantic"),
    "screening/medium_full": ("detector_case2_bit_levircd_to_second", "binary"),
    "v9_medium_scratch/medium_full": ("detector_case3_bit_second_only", "binary"),
    "v9_light_resized/light_resized": ("detector_case4_fc_siam_diff_resized", "binary"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_checkpoints() -> dict[str, dict]:
    """Unique frozen checkpoints (repo-relative path -> manifest entry)."""
    manifest = json.loads(MANIFEST.read_text())
    unique: dict[str, dict] = {}
    for system in manifest["systems"].values():
        for entry in system["checkpoints"]:
            path = Path(entry["path"])
            relative = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
            unique[relative] = entry
    return unique


def export(relative: str, entry: dict, out: Path) -> dict:
    source = ROOT / relative
    if sha256(source) != entry["sha256"]:
        raise ValueError(f"{relative}: SHA-256 differs from the freeze manifest")
    run = source.parent
    group = str(run.parent.relative_to(ROOT / "artifacts/research2/v2/runs"))
    if group not in ROLES:
        raise ValueError(f"{relative}: unknown run group {group}")
    prefix, task = ROLES[group]
    state = torch.load(source, map_location="cpu", weights_only=False)  # own, frozen file
    if task == "semantic":
        weights = state["model_state_dict"]
        model_name = state["model_name"]
        parameters = state["config"]["model_parameters"]
    else:
        weights = state.get("best_model_state_dict") or state["model_state_dict"]
        config = json.loads((run / "run_record.json").read_text())["config"]
        model_name = config["init"]["model_name"]
        parameters = config["model_parameters"]
    seed = int(run.name.rsplit("_s", 1)[1])
    payload = {
        "format": FORMAT,
        "task": task,
        "model_name": model_name,
        "model_parameters": parameters,
        "state_dict": {key: tensor.clone() for key, tensor in weights.items()},
        "seed": seed,
        "epoch": int(state["epoch"]),
        "source_run": run.name,
        "source_sha256": entry["sha256"],
    }
    target = out / f"{prefix}_s{seed}.pt"
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    loaded = torch.load(target, map_location="cpu", weights_only=True)
    if loaded["state_dict"].keys() != weights.keys() or not all(
        torch.equal(loaded["state_dict"][key], weights[key]) for key in weights
    ):
        raise ValueError(f"{target.name}: exported tensors differ from the source")
    return {"file": target.name, "bytes": target.stat().st_size, "sha256": sha256(target),
            "task": task, "model_name": model_name, "seed": seed, "epoch": payload["epoch"],
            "source_run": run.name, "source_sha256": entry["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [export(relative, entry, args.out)
            for relative, entry in sorted(frozen_checkpoints().items())]
    (args.out / "weights_manifest.json").write_text(json.dumps(
        {"format": FORMAT, "files": sorted(rows, key=lambda row: row["file"])}, indent=1) + "\n")
    for row in rows:
        print(f"{row['file']:48s} {row['bytes'] / 1e6:7.1f} MB  {row['sha256'][:12]}", flush=True)


if __name__ == "__main__":
    main()
