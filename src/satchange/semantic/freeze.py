"""Freeze manifest for confirmatory Research 2 runs (D-010, R2-P11-T05).

Records, for each frozen system, checkpoint paths with SHA-256, the configuration, the code
revision and the protocol file hash. Refuses a dirty working tree and never overwrites an
existing manifest. Building the manifest never reads any dataset.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def build_freeze_manifest(
    systems: dict[str, dict[str, Any]], *, root: Path, protocol: Path, output: Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """``systems[name] = {"checkpoints": [paths], "config": {...}, "selection_evidence": path}``."""
    if output.exists():
        raise FileExistsError(f"Freeze manifest already exists: {output}")
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    if dirty and not allow_dirty:
        raise RuntimeError("Working tree has uncommitted changes; commit before freezing.")
    entries = {}
    for name, system in systems.items():
        checkpoints = []
        for item in system["checkpoints"]:
            path = Path(item)
            if not path.is_file() or path.stat().st_size < 1024:
                raise FileNotFoundError(f"Checkpoint missing or LFS pointer: {path}")
            checkpoints.append({"path": str(path), "sha256": sha256_file(path),
                                "bytes": path.stat().st_size})
        evidence = Path(system["selection_evidence"])
        if not evidence.is_file():
            raise FileNotFoundError(f"Selection evidence missing: {evidence}")
        entries[name] = {
            "checkpoints": checkpoints, "config": system["config"],
            "selection_evidence": {"path": str(evidence), "sha256": sha256_file(evidence)},
        }
    manifest = {
        "schema": 1, "created_utc": time.strftime("%FT%TZ", time.gmtime()),
        "git_revision": _git(root, "rev-parse", "HEAD"), "git_dirty": dirty,
        "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
        "systems": entries, "test_split_read": False,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return manifest


FROZEN_CODE = ("src/satchange", "src/scripts/eval_r2_frozen.py")


def verify_freeze_manifest(
    manifest_path: Path, *, root: Path, code_paths: tuple[str, ...] = FROZEN_CODE
) -> dict[str, Any]:
    """Re-hash everything a freeze manifest references and check the code is still the frozen one.

    Raises on any mismatch: a checkpoint, the selection evidence or the protocol file changed;
    the frozen revision is not an ancestor of HEAD; any of ``code_paths`` (the library and the
    evaluation script) differs from the frozen revision; uncommitted changes under ``src/``.
    """
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1:
        raise ValueError(f"Unsupported freeze manifest schema: {manifest.get('schema')}")
    problems = []
    for name, system in manifest["systems"].items():
        for item in system["checkpoints"]:
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                problems.append(f"{name}: checkpoint changed or missing: {path}")
        evidence = system["selection_evidence"]
        if sha256_file(Path(evidence["path"])) != evidence["sha256"]:
            problems.append(f"{name}: selection evidence changed: {evidence['path']}")
    protocol = manifest["protocol"]
    if sha256_file(Path(protocol["path"])) != protocol["sha256"]:
        problems.append(f"protocol changed: {protocol['path']}")
    revision = manifest["git_revision"]
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", revision, "HEAD"], check=False
    ).returncode
    if ancestor != 0:
        problems.append(f"frozen revision {revision} is not an ancestor of HEAD")
    elif changed := _git(root, "diff", "--name-only", revision, "HEAD", "--", *code_paths):
        problems.append(f"frozen code changed after the freeze: {changed.splitlines()}")
    if _git(root, "status", "--porcelain", "--untracked-files=no", "--", "src"):
        problems.append("uncommitted changes under src/")
    if problems:
        raise RuntimeError("Freeze verification failed:\n- " + "\n- ".join(problems))
    return manifest


def claim_test_read(manifest_path: Path, *, system_set: str) -> Path:
    """Atomically record that the test split is about to be read for this manifest (once only).

    The marker ``<manifest>.test_read.json`` is created with O_EXCL; a second claim raises
    ``FileExistsError``. A failed pass must be re-run as a new, documented experiment (D-007).
    """
    marker = manifest_path.with_suffix(manifest_path.suffix + ".test_read.json")
    descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps({
            "system_set": system_set, "claimed_utc": time.strftime("%FT%TZ", time.gmtime()),
            "manifest_sha256": sha256_file(manifest_path),
        }, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    return marker
