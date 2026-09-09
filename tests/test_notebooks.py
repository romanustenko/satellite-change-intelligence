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
    notebook = nbformat.read(root / "notebooks/03_heavy.ipynb", as_version=4)
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
