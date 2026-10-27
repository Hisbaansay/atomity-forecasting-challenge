"""Optional review deliverables remain testable without notebook/PDF extras."""
import json
import hashlib
import re
import runpy
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_schema_and_protocol():
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(ROOT / "notebooks/workload_analysis.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 22
    assert "forecast_targets" in code[1]
    assert "train_size=110, horizon=24" in code[1]
    assert "del targets, cpu, metadata" in code[1]
    assert "run_analysis" in code[2]
    assert "holdout_scored" in code[2]
    assert "isolation['passed']" in code[-2]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not any(output.output_type == "error" for output in cell.outputs)


def test_pdf_has_six_main_report_pages_when_available():
    pypdf = pytest.importorskip("pypdf")
    path = ROOT / "reports/final_report.pdf"
    if not path.exists():
        pytest.skip("Render report to check pagination")
    reader = pypdf.PdfReader(path)
    assert "Executive Summary" in reader.pages[0].extract_text()
    assert "Limitations And Future Work" in reader.pages[5].extract_text()
    assert "Appendix" in reader.pages[6].extract_text()
    assert len(reader.pages) >= 7


def test_notebook_file_has_no_stored_holdout_actual_table():
    notebook = json.loads((ROOT / "notebooks/workload_analysis.ipynb").read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "display(targets)" not in sources
    assert "display(cpu)" not in sources
    assert "saved = predictions.copy()" in sources
    assert "show_table(saved)" in sources  # Forecasts, not observed holdout labels.


def test_submission_archive_has_required_files_and_no_local_artifacts(tmp_path):
    build_archive = runpy.run_path(str(ROOT / "scripts/package_submission.py"))["build_archive"]
    path = build_archive(ROOT, tmp_path / "submission.zip")
    with ZipFile(path) as archive:
        names = [Path(name).relative_to("atomity-forecasting-challenge") for name in archive.namelist()]
        assert archive.testzip() is None
    assert {Path(name) for name in (
        "README.md", "predictions.csv", "pyproject.toml", "notebooks/workload_analysis.ipynb",
        "reports/final_report.md", "reports/final_report.pdf",
        "reports/results/model_comparison.csv", "reports/figures/backtest_cpu.png",
        "src/atomity_forecasting/plotting.py",
        "reports/results/validation_sensitivity.csv",
    )}.issubset(names)
    assert not any(path.suffix in {".gz", ".pyc", ".zip"} for path in names)
    assert not any(part in {".venv", ".git", ".official", "__pycache__", ".pytest_cache"}
                   for path in names for part in path.parts)
    assert not any(path.parts[0] in {"docs", ".vscode"} for path in names)
    assert Path("reports/revision_record.md") not in names
    assert Path("reports/results/revision_integrity.json") not in names
    assert Path("scripts/verify_revision.py") not in names


@pytest.mark.parametrize("missing", ["reports/final_report.pdf", "notebooks/workload_analysis.ipynb",
                                     "reports/results/capacity_schedule.csv"])
def test_packaging_fails_if_required_generated_deliverable_is_missing(tmp_path, monkeypatch, missing):
    build_archive = runpy.run_path(str(ROOT / "scripts/package_submission.py"))["build_archive"]
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: path != ROOT / missing and original(path))
    output = tmp_path / "incomplete.zip"
    with pytest.raises(FileNotFoundError, match="Required submission file missing"):
        build_archive(ROOT, output)
    assert not output.exists()


def test_report_layout_and_notebook_paths():
    assert not list((ROOT / "reports").glob("*.png"))
    assert not list((ROOT / "reports").glob("*.csv"))
    assert not list((ROOT / "reports").glob("*.json"))
    assert len(list((ROOT / "reports/figures").glob("*.png"))) == 12
    notebook = json.loads((ROOT / "notebooks/workload_analysis.ipynb").read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RESULTS = REPORTS / 'results'" in sources
    assert "FIGURES = REPORTS / 'figures'" in sources


def test_review_markdown_local_links_resolve():
    documents = [ROOT / "README.md", *(ROOT / "reports").glob("*.md"), *(ROOT / "docs").glob("*.md")]
    notebook_path = ROOT / "notebooks/workload_analysis.ipynb"
    notebook = json.loads(notebook_path.read_text())
    sources = [(path, path.read_text()) for path in documents]
    sources.extend((notebook_path, "".join(cell["source"]))
                   for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    for source, text in sources:
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if "://" not in link and not link.startswith("#"):
                assert (source.parent / link.split("#")[0]).exists(), (source, link)


def test_pdf_has_three_informative_main_figures():
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(ROOT / "reports/final_report.pdf")
    assert [i + 1 for i, page in enumerate(reader.pages[:6]) if list(page.images)] == [2, 4, 5]


def test_frozen_prediction_hash_is_preserved():
    assert hashlib.sha256((ROOT / "predictions.csv").read_bytes()).hexdigest() == (
        "e224a7d8e711b4d7d47da25f0bcb1a52e24ff8eb04f76b39b321d0ce04c9da5f"
    )


def test_generated_diagnostics_match_training_only_calculations():
    import pandas as pd
    from atomity_forecasting.evaluation import validation_sensitivity

    results = ROOT / "reports/results"
    train = pd.read_csv(results / "targets_train.csv")
    actual = pd.read_csv(results / "validation_sensitivity.csv")
    expected = validation_sensitivity(train)
    # CSV encodes undefined improvement as NaN; the in-memory API uses None.
    expected["relative_improvement"] = expected.relative_improvement.astype(float)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                  check_exact=False, rtol=1e-12)
    comparison = pd.read_csv(results / "model_comparison.csv")
    cpu = comparison[comparison.target == "aggregate_cpu_utilization"]
    for row in cpu.itertuples():
        assert row.training_mean == pytest.approx(train.aggregate_cpu_utilization.mean())
        assert row.normalized_MAE_pct == pytest.approx(100 * row.MAE / row.training_mean)


def test_notebook_cli_prefers_invoking_environment(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(ROOT / "notebooks/workload_analysis.ipynb", as_version=4)
    for index, cell in enumerate(cell for cell in notebook.cells if cell.cell_type == "code"):
        cell.execution_count = index + 1
        cell.outputs = []
    # The mocked CLI does not execute cells; emulate its successful return rather
    # than relying on whichever outputs happen to be saved in a user's notebook.
    monkeypatch.setattr(nbformat, "read", lambda *args, **kwargs: notebook)
    # Synchronization writes generated notebook cells; this test only exercises
    # the CLI environment setup, not the separate synchronization step.
    import materialize_notebook_plots
    monkeypatch.setattr(materialize_notebook_plots, "materialize", lambda path: None)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    runpy.run_path(str(ROOT / "scripts/execute_notebook.py"))["main"]()
    command, options = calls[0]
    assert command[:4] == [sys.executable, "-m", "jupyter", "nbconvert"]
    assert "--execute" in command and "--inplace" in command
    assert options["env"]["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert options["env"]["JUPYTER_PATH"].split(os.pathsep)[0] == str(Path(sys.prefix) / "share/jupyter")


def test_generated_headroom_and_cross_section_artifacts():
    import numpy as np
    import pandas as pd

    headroom = pd.read_csv(ROOT / "reports/results/capacity_schedule.csv")
    assert len(headroom) == 24
    assert headroom.minutes_ahead.tolist() == list(range(5, 121, 5))
    for percentile in ("p90", "p95"):
        values = headroom[f"buffer_{percentile}_pp"].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all()
        assert (np.diff(values) >= 0).all()
    assert headroom.buffer_p95_pp.ge(headroom.buffer_p90_pp).all()
    section = pd.read_csv(ROOT / "reports/results/cross_sectional_quantiles.csv")
    quantiles = section[[f"p{q}_avg_cpu" for q in (50, 75, 90, 95, 99)] + ["max_avg_cpu"]].to_numpy()
    assert len(section) == 110
    assert (np.diff(quantiles, axis=1) >= 0).all()
    assert section.reporting_vm_count.eq(10000).all()
    assert section.workload_active_vm_count.le(section.reporting_vm_count).all()
