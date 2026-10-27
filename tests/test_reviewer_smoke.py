"""Exercise the public prediction command without the official release files."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def write_smoke_data(path: Path, vm_count: int, timestamps: int) -> pd.DataFrame:
    frame = pd.DataFrame([
        {"timestamp": 123000 + step * 300, "vm_id": f"smoke-{vm}",
         "min_cpu": 0, "avg_cpu": 10 + step / 10 + vm, "max_cpu": 100}
        for step in range(timestamps) for vm in range(vm_count)
    ])
    frame.to_csv(path / "vm_cpu_readings.csv.gz", index=False)
    pd.DataFrame({"vm_id": [f"smoke-{vm}" for vm in range(vm_count)]}).to_csv(
        path / "vm_metadata.csv.gz", index=False)
    return frame


def update_manifest(path: Path) -> None:
    names = ("vm_cpu_readings.csv.gz", "vm_metadata.csv.gz")
    (path / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}\n" for name in names
    ))


def run_prediction(path: Path, *options: str) -> pd.DataFrame:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([
        sys.executable, "-m", "atomity_forecasting.predict", "--data", str(path),
        "--output", str(path / "predictions.csv"), *options,
    ], cwd=path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    prediction = pd.read_csv(path / "predictions.csv")
    assert prediction.columns.tolist() == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert np.isfinite(prediction.to_numpy()).all()
    assert not prediction.timestamp.duplicated().any()
    assert prediction.active_vm_count.ge(0).all()
    assert prediction.aggregate_cpu_utilization.between(0, 100).all()
    return prediction


@pytest.mark.parametrize("vm_count", [1, 3, 5])
def test_default_command_accepts_tiny_vm_population_and_ignores_holdout_labels(tmp_path, vm_count):
    frame = write_smoke_data(tmp_path, vm_count, 134)
    update_manifest(tmp_path)
    prediction = run_prediction(tmp_path)
    assert len(prediction) == 24
    np.testing.assert_array_equal(prediction.timestamp, 123000 + np.arange(110, 134) * 300)
    assert prediction.active_vm_count.eq(vm_count).all()
    assert not prediction.aggregate_cpu_utilization.eq(9.164556174985147).any()
    original = (tmp_path / "predictions.csv").read_bytes()
    original_selection = json.loads((tmp_path / "predictions.selection.json").read_text())
    frame.loc[frame.timestamp >= 123000 + 110 * 300, "avg_cpu"] = 99.0
    frame.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    update_manifest(tmp_path)
    run_prediction(tmp_path)
    assert (tmp_path / "predictions.csv").read_bytes() == original
    assert json.loads((tmp_path / "predictions.selection.json").read_text()) == original_selection


def test_short_smoke_series_runs_with_explicit_split_and_horizon(tmp_path):
    write_smoke_data(tmp_path, 3, 16)
    # An independent synthetic input need not ship the public release manifest.
    prediction = run_prediction(tmp_path, "--train-size", "12", "--horizon", "4")
    assert len(prediction) == 4
    np.testing.assert_array_equal(prediction.timestamp, 123000 + np.arange(12, 16) * 300)
    assert prediction.active_vm_count.eq(3).all()
