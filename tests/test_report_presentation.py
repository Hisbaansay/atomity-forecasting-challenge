import json
from pathlib import Path

import pytest

from atomity_forecasting.typography import times_new_roman_files

ROOT = Path(__file__).resolve().parents[1]


def test_complete_times_new_roman_override(tmp_path, monkeypatch):
    names = ("Times New Roman.ttf", "Times New Roman Bold.ttf",
             "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf")
    for name in names:
        (tmp_path / name).touch()
    monkeypatch.setenv("ATOMITY_FONT_DIR", str(tmp_path))
    fonts = times_new_roman_files()
    assert set(fonts) == {"regular", "bold", "italic", "bold_italic"}
    assert {path.name for path in fonts.values()} == set(names)


def test_incomplete_explicit_font_override_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOMITY_FONT_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="all four"):
        times_new_roman_files()


def test_pdf_uses_times_family_and_justified_paragraph_spacing():
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import ContentStream

    reader = pypdf.PdfReader(ROOT / "reports/final_report.pdf")
    fonts = [font.get_object() for font in reader.pages[0]["/Resources"]["/Font"].values()]
    assert any("Times" in str(font.get("/BaseFont")) for font in fonts)
    for font in fonts:
        if "TimesNewRoman" in str(font.get("/BaseFont")):
            assert "/FontFile2" in font["/FontDescriptor"].get_object()
    operations = ContentStream(reader.pages[0].get_contents(), reader).operations
    assert any(operator == b"Tw" and float(operands[0]) > .1 for operands, operator in operations)


def test_notebook_shows_plotting_code_and_no_saved_image_inputs():
    import ast
    import inspect
    from atomity_forecasting.plotting import PLOTTERS

    notebook = json.loads((ROOT / "notebooks/workload_analysis.ipynb").read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "Image(filename" not in code
    assert "display(fig)" in code
    assert "assert set(computed['plot_inputs']) == set(PLOTTERS)" in code
    assert "plt.subplots(" in code
    assert "pd.read_csv(" not in code and ".read_text(" not in code
    assert "computed['tables']" in code
    assert "'color': '#17212b !important'" in code and "'background-color': '#ffffff !important'" in code
    assert "display(HTML(" not in code
    assert "on_figure=" not in code
    assert "display(pd.DataFrame" not in code
    assert "Code(inspect.getsource" not in code
    assert "show_table(saved)" in code
    cells = [cell for cell in notebook["cells"] if cell["metadata"].get("plot_name")]
    assert len(cells) == len(PLOTTERS) == 12
    for cell in cells:
        source = "".join(cell["source"])
        function = PLOTTERS[cell["metadata"]["plot_name"]]
        definition = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
        assert ast.dump(definition) == ast.dump(ast.parse(inspect.getsource(function)).body[0])
        assert f"fig = {function.__name__}(*computed['plot_inputs']" in source
