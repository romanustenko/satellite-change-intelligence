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
