"""Install the three HPO modules and minimally patch the project's trainer.

Run only AFTER stopping all old HPO workers. No data/artifacts are changed.
Every existing source is backed up before replacement. Unknown trainer layouts
are rejected BEFORE any source is changed; the function is not guessed.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ATOMIC_SAVER = '''def _save_checkpoint(checkpoint_path: Path, checkpoint_state: dict[str, Any]) -> None:
    """Commit one checkpoint atomically; an interrupted write keeps the old file."""
    import os
    from uuid import uuid4

    if rank() != 0:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(checkpoint_state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)
'''


def patch_training(source: str) -> str:
    if "HPO_TRAINER_PATCH_VERSION = 1" in source:
        ast.parse(source)
        return source
    tree = ast.parse(source)
    nodes = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    if not {"_save_checkpoint", "fit_binary_change_model", "EarlyStopping"}.issubset(nodes):
        raise RuntimeError("Unsupported trainer: expected project fit, EarlyStopping, and checkpoint saver.")
    fit = nodes["fit_binary_change_model"]
    text = "\n".join(source.splitlines()[fit.lineno - 1:fit.end_lineno])
    marker = ") -> list[dict[str, Any]]:"
    if marker not in text:
        raise RuntimeError("Unsupported trainer signature layout. Installation stopped without changing files.")
    signature_end = text.index(marker)
    signature = text[:signature_end]
    if "reset_early_stopping" in signature or "stop_file" in signature:
        raise RuntimeError("Trainer already has different pause/resume hooks; review before installing.")
    text = text[:signature_end] + (
        "    reset_early_stopping: bool = False,\n"
        "    stop_file: str | Path | None = None,\n"
    ) + text[signature_end:]
    old = '        early_stopping.load_state_dict(checkpoint.get("early_stopping"), saved_epoch)'
    if text.count(old) != 1:
        raise RuntimeError("Unknown early-stopping resume code. Installation stopped without changing files.")
    text = text.replace(old, '''        if reset_early_stopping:
            # Explicit phase promotion: preserve weights/optimizer/scheduler/RNG/best history.
            early_stopping.last_epoch = saved_epoch
        else:
            early_stopping.load_state_dict(checkpoint.get("early_stopping"), saved_epoch)
        del checkpoint  # Do not retain a second copy of checkpoint tensors during training.''')
    anchor = '        on_rank_zero(save_epoch)'
    if text.count(anchor) != 1:
        raise RuntimeError("Unknown epoch-checkpoint layout. Installation stopped without changing files.")
    text = text.replace(anchor, anchor + '''
        pause_requested = any(gather_objects(
            stop_file is not None and Path(stop_file).exists()
        ))
        if pause_requested:
            if rank() == 0:
                print(f"{log_prefix}Paused after saving epoch {epoch}.", flush=True)
            break''')
    # Replace functions from the end to keep original line offsets valid.
    lines = source.splitlines(keepends=True)
    replacements = [(fit.lineno - 1, fit.end_lineno, text + "\n"),
                    (nodes["_save_checkpoint"].lineno - 1, nodes["_save_checkpoint"].end_lineno, ATOMIC_SAVER)]
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines) + "\n\nHPO_TRAINER_PATCH_VERSION = 1\n"
    ast.parse(result)
    return result


def install(project_root: Path, package_root: Path, *, dry_run: bool = False) -> Path | None:
    target = project_root / "src/satchange"
    training_path = target / "training.py"
    original = training_path.read_text(encoding="utf-8-sig")
    patched = patch_training(original)
    sources = {name: (package_root / "src/satchange" / name).read_text(encoding="utf-8")
               for name in ("automated_hpo.py", "hpo_core.py", "hpo_backend.py")}
    for name, content in sources.items():
        ast.parse(content, filename=name)
    sources["training.py"] = patched
    if dry_run:
        print("Trainer layout accepted. No files changed.")
        print("Would install:", ", ".join(sources))
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    backup = project_root / "source_backups" / f"hpo_v3_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    existing = {}
    try:
        for name in sources:
            path = target / name
            existing[name] = path.exists()
            if path.exists():
                shutil.copy2(path, backup / name)
        for name, content in sources.items():
            path = target / name
            temporary = path.with_suffix(".py.installing")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
    except BaseException:
        for name, existed in existing.items():
            if existed and (backup / name).exists():
                shutil.copy2(backup / name, target / name)
            elif not existed:
                (target / name).unlink(missing_ok=True)
        raise
    print("Installed. Restart the notebook kernel. Backup:", backup)
    return backup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()

    install(
        project_root,
        project_root,
        dry_run=args.dry_run,
    )