"""Release-specific regression evidence, separate from size-independent smoke tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomity_forecasting.data import load_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release():
    data = ROOT / "challenge_data"
    if not (data / "vm_cpu_readings.csv.gz").exists() or not (data / "SHA256SUMS").exists():
        pytest.skip("Download the verified official release to run its contract regressions")
    return load_data(data)


def test_official_release_dimensions_schema_and_validity(release):
    cpu, metadata, audit = release
    assert audit["raw_rows"] == len(cpu) == 1_340_000
    assert cpu.vm_id.nunique() == metadata.vm_id.nunique() == 10_000
    assert cpu.timestamp.nunique() == 134
    assert set(cpu.columns) == {"timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"}
    assert sum(audit["missing_by_column"].values()) == 0
    assert sum(audit["metadata_missing_by_column"].values()) == 0
    assert audit["duplicate_key_rows"] == audit["invalid_cpu_or_id_rows"] == 0
    assert cpu.groupby("vm_id").size().eq(134).all()
    assert np.diff(np.sort(cpu.timestamp.unique())).tolist() == [300] * 133


def test_exact_official_frozen_prediction_contract(release):
    cpu, _, _ = release
    prediction = pd.read_csv(ROOT / "predictions.csv")
    assert prediction.columns.tolist() == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(prediction) == 24
    np.testing.assert_array_equal(prediction.timestamp, np.sort(cpu.timestamp.unique())[110:])
    assert not prediction.timestamp.duplicated().any()
    assert np.isfinite(prediction.to_numpy()).all()
    assert prediction.active_vm_count.eq(10000).all()
    np.testing.assert_allclose(prediction.aggregate_cpu_utilization, 9.16456, rtol=0, atol=1e-5)
