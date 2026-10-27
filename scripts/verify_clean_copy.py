"""Reproduce the submission in an isolated source copy and fresh environment.

Requires network/package-cache access for pip. Temporary raw-data copies and
environment are removed afterwards; results are retained as verification JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from package_submission import build_archive


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = {"isolated_source_copy": True, "submission_archive_extracted": True,
                "fresh_environment": True,
                "data_copied_from_verified_release": True, "steps": []}
    with tempfile.TemporaryDirectory(prefix="atomity-clean-review-") as temporary:
        archive = build_archive(root, Path(temporary) / "submission.zip")
        with ZipFile(archive) as package:
            package.extractall(temporary)
        project = Path(temporary) / "atomity-forecasting-challenge"
        # A reviewer downloads these separately; use the locally verified release.
        for name in ("vm_cpu_readings.csv.gz", "vm_metadata.csv.gz"):
            shutil.copy2(root / "challenge_data" / name, project / "challenge_data" / name)
        environment = Path(temporary) / "venv"

        def run(label: str, command: list[str]) -> str:
            print(f"Running {label}...", flush=True)
            result = subprocess.run(command, cwd=project, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")
            evidence["steps"].append({"stage": label, "passed": True})
            return result.stdout

        run("create environment", [sys.executable, "-m", "venv", str(environment)])
        python = str(environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        run("install", [python, "-m", "pip", "install", "-c", "constraints.txt", "-e", ".[dev,notebook,report]"])
        run("predict", [python, "-m", "atomity_forecasting.predict", "--data", "challenge_data", "--output", "predictions.csv"])
        run("analysis", [python, "-m", "atomity_forecasting.analysis", "--data", "challenge_data", "--output", "reports"])
        evidence["pdf_result"] = run("render PDF", [python, "scripts/render_report.py"]).strip()
        evidence["test_result"] = run("tests", [python, "-m", "pytest", "-q"]).strip()
        evidence["notebook_result"] = run("execute notebook", [python, "scripts/execute_notebook.py"]).strip().split(":")[0]
        expected = hashlib.sha256((root / "predictions.csv").read_bytes()).hexdigest()
        actual = hashlib.sha256((project / "predictions.csv").read_bytes()).hexdigest()
        if expected != actual:
            raise AssertionError("Clean-copy predictions differ from workspace predictions")
        evidence.update(predictions_byte_identical=True, prediction_sha256=actual)
    (root / "reports/results/clean_copy_verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
