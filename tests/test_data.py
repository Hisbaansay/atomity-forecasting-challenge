import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomity_forecasting.data import clean_readings, construct_targets, load_data, verify_checksums
from atomity_forecasting.predict import main


def readings():
    return pd.DataFrame([
        (300, "b", 0, 100, 40), (0, "a", 0, 100, 10),
        (0, "a", 0, 100, 30), (0, "b", 0, 100, 50),
        (300, "a", 0, 100, 20),
    ], columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def test_duplicates_do_not_overweight_and_order_is_chronological():
    with pytest.warns(UserWarning):
        target = construct_targets(readings())
    assert target.timestamp.tolist() == [0, 300]
    assert target.active_vm_count.tolist() == [2, 2]
    assert target.aggregate_cpu_utilization.tolist() == [35, 30]


@pytest.mark.parametrize("column,value", [
    ("avg_cpu", np.nan), ("avg_cpu", np.inf), ("avg_cpu", -1),
    ("avg_cpu", 101), ("min_cpu", 60), ("max_cpu", 1), ("vm_id", None),
])
def test_invalid_readings_reported_and_excluded(column, value):
    frame = readings().drop_duplicates(["timestamp", "vm_id"]).astype(
        {"min_cpu": float, "max_cpu": float, "avg_cpu": float})
    frame.loc[0, column] = value
    with pytest.warns(UserWarning):
        clean, audit = clean_readings(frame)
    assert audit["invalid_cpu_or_id_rows"] == 1
    assert len(clean) == 3
    result = construct_targets(clean)
    assert result.active_vm_count.tolist() == [2, 1]
    assert result.aggregate_cpu_utilization.tolist() == [30, 20]


def test_gaps_and_empty_timestamps_fail_explicitly():
    frame = readings()
    frame.loc[frame.timestamp == 300, "timestamp"] = 600
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="gaps"):
        construct_targets(frame)
    frame = readings()
    frame.loc[frame.timestamp == 300, "avg_cpu"] = np.nan
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no valid"):
        construct_targets(frame)


def test_checksum_verification_and_tampering(tmp_path):
    # DataFrame serialization supplies a realistic small source file.
    readings().to_csv(tmp_path / "data.csv", index=False)
    digest = hashlib.sha256((tmp_path / "data.csv").read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  data.csv\n")
    assert verify_checksums(tmp_path) == {"data.csv": digest}
    readings().iloc[:1].to_csv(tmp_path / "data.csv", index=False)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(tmp_path)


def test_tiny_dataset_full_cli(tmp_path):
    cpu = pd.DataFrame([
        (t * 300, vm, 0, 100, 10 + t + vm)
        for t in range(16) for vm in range(3)
    ], columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    pd.DataFrame({"vm_id": [0, 1, 2]}).to_csv(tmp_path / "vm_metadata.csv.gz", index=False)
    output = tmp_path / "predictions.csv"
    with pytest.warns(UserWarning, match="unverified"):
        main(["--data", str(tmp_path), "--output", str(output),
              "--train-size", "12", "--horizon", "4"])
    result = pd.read_csv(output)
    assert result.columns.tolist() == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert result.timestamp.tolist() == [3600, 3900, 4200, 4500]
    assert len(result) == 4 and not result.timestamp.duplicated().any()
    assert np.isfinite(result.to_numpy()).all()
    assert result.active_vm_count.ge(0).all()
    assert result.aggregate_cpu_utilization.between(0, 100).all()
    # Population mismatches must not silently affect segment analysis.
    pd.DataFrame({"vm_id": [0, 1]}).to_csv(tmp_path / "vm_metadata.csv.gz", index=False)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="population mismatch"):
        load_data(tmp_path)


def test_release_prediction_artifact_when_available():
    path = Path(__file__).resolve().parents[1]
    if not (path / "challenge_data/vm_cpu_readings.csv.gz").exists():
        pytest.skip("Release data is downloaded separately")
    if not (path / "predictions.csv").exists():
        pytest.skip("Generate predictions.csv to verify the submission artifact")
    cpu, _, _ = load_data(path / "challenge_data")
    targets = construct_targets(cpu)
    from atomity_forecasting.predict import forecast_targets
    expected, _, _ = forecast_targets(targets)
    actual = pd.read_csv(path / "predictions.csv")
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-14)
    np.testing.assert_array_equal(actual.timestamp, targets.timestamp.iloc[110:134])
    assert len(actual) == 24
    assert not actual.timestamp.duplicated().any()
    assert np.isfinite(actual.to_numpy()).all()
    assert actual.active_vm_count.ge(0).all()
    assert actual.aggregate_cpu_utilization.between(0, 100).all()
