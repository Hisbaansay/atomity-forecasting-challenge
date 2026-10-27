"""Execute the review notebook from a clean kernel and retain its outputs."""
from pathlib import Path
import os
import subprocess
import sys

import nbformat
from materialize_notebook_plots import materialize


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/workload_analysis.ipynb"
    materialize(path)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    # Jupyter dispatches subcommands through PATH, which may otherwise select a
    # system Conda installation even when this script uses the project Python.
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    environment["JUPYTER_PATH"] = str(Path(sys.prefix) / "share/jupyter") + os.pathsep + environment.get("JUPYTER_PATH", "")
    subprocess.run([
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace", str(path), "--ExecutePreprocessor.timeout=600",
        "--ExecutePreprocessor.kernel_name=python3",
    ], cwd=root, env=environment, check=True)
    notebook = nbformat.read(path, as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    print(f"Executed {len(cells)} code cells without errors: {path}")


if __name__ == "__main__":
    main()
