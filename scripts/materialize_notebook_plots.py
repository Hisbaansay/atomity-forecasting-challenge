"""Place the canonical chart implementations in genuine executable notebook cells.

The cells contain ordinary Python definitions and draw their own figures. They
are synchronized from plotting.py, avoiding two manually maintained versions.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import nbformat

from atomity_forecasting.plotting import PLOTTERS


def plot_cell(name: str):
    function = PLOTTERS[name]
    source = inspect.getsource(function).rstrip()
    source += (
        f"\n\nfig = {function.__name__}(*computed['plot_inputs']['{name}'])\n"
        f"save_figure(fig, FIGURES / '{name}.png')\n"
        f"computed['figures']['{name}'] = fig\n"
        "FigureCanvasAgg(fig)\ndisplay(fig)"
    )
    cell = nbformat.v4.new_code_cell(source)
    cell.id = "plot-" + name.replace("_", "-")
    cell.metadata["plot_name"] = name
    return cell


def materialize(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    cells = []
    for cell in notebook.cells:
        if cell.metadata.get("plot_name"):
            cells.append(plot_cell(cell.metadata["plot_name"]))
            continue
        if cell.cell_type != "code":
            cells.append(cell)
            continue
        source = cell.source
        if cell.id == "setup" and "from matplotlib.figure import Figure" not in source:
            source += "\nimport matplotlib.pyplot as plt\nfrom matplotlib.figure import Figure\nfrom typing import Any\nfrom atomity_forecasting.models import TARGETS\nfrom atomity_forecasting.plotting import save_figure"
        if "computed = run_analysis(" in source:
            source = source.replace("str(REPORTS)])", "str(REPORTS)], render_figures=False)")
            source = source.replace("set(computed['figures'])", "set(computed['plot_inputs'])")
        tree = ast.parse(source)
        statements = []
        pending = []

        def flush():
            if pending:
                new = nbformat.v4.new_code_cell("\n".join(pending))
                new.id = cell.id if not statements else cell.id + "-continued"
                statements.append(new)
                pending.clear()

        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == "show_plot":
                continue
            if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "show_plot"):
                flush()
                statements.append(plot_cell(ast.literal_eval(statement.value.args[0])))
            elif (isinstance(statement, ast.For) and isinstance(statement.iter, (ast.List, ast.Tuple))
                  and len(statement.body) == 1 and isinstance(statement.body[0], ast.Expr)
                  and isinstance(statement.body[0].value, ast.Call)
                  and isinstance(statement.body[0].value.func, ast.Name)
                  and statement.body[0].value.func.id == "show_plot"):
                flush()
                statements.extend(plot_cell(name) for name in ast.literal_eval(statement.iter))
            else:
                pending.append(ast.get_source_segment(source, statement))
        flush()
        cells.extend(statements)
    notebook.cells = cells
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Synchronized {sum(bool(cell.metadata.get('plot_name')) for cell in cells)} executable plotting cells")


if __name__ == "__main__":
    materialize(Path(__file__).resolve().parents[1] / "notebooks/workload_analysis.ipynb")
