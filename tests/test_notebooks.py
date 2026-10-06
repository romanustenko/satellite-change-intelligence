"""Validate notebook Python and execute the Heavy safe path in a fresh project kernel."""

import sys
from pathlib import Path

import pytest


def test_notebook_syntax_and_heavy_safe_fresh_kernel():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    jupyter_client = pytest.importorskip("jupyter_client")
    transformer = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager()
    root = Path(__file__).resolve().parents[1]
    for path in (root / "notebooks").glob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(transformer.transform_cell(cell.source), f"{path}:{index}", "exec")
    legacy = root / "notebooks/03_heavy.ipynb"
    if not legacy.is_file():
        pytest.skip("Legacy 03_heavy notebook is absent; all notebook syntax was validated.")
    notebook = nbformat.read(legacy, as_version=4)
    manager = jupyter_client.KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    try:
        nbclient.NotebookClient(
            notebook,
            km=manager,
            timeout=120,
            resources={"metadata": {"path": str(root / "notebooks")}},
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    outputs = " ".join(str(cell.get("outputs", [])) for cell in notebook.cells)
    assert "45772961" in outputs and "finite:" in outputs


def test_tiny_second_safe_fresh_kernel():
    """Execute default guards and CPU smoke; never downloads or runs real HPO/test."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    jupyter_client = pytest.importorskip("jupyter_client")
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/tiny_siamese_second_staged_resume.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert all(not cell.get("outputs") for cell in notebook.cells)
    manager = jupyter_client.KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    try:
        nbclient.NotebookClient(
            notebook,
            km=manager,
            timeout=60,
            resources={"metadata": {"path": str(root / "notebooks")}},
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    outputs = " ".join(str(cell.get("outputs", [])) for cell in notebook.cells)
    assert "Finite synthetic CE" in outputs and "Final test disabled" in outputs
    assert "Preparation disabled" in outputs and "01_batch_lr disabled" in outputs
    # Executed outputs remain in memory; the delivered notebook stays clean.
    assert all(not cell.get("outputs") for cell in nbformat.read(path, as_version=4).cells)
