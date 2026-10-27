import numpy as np
import pandas as pd
import pytest

from atomity_forecasting.analysis import workload_diagnostics


def fixture():
    cpu = pd.DataFrame([(t * 300, vm, value) for vm, values in {
        "a": [90, 90, 90, 90], "b": [0, 80, 0, 80], "c": [10, 10, 10, 10],
    }.items() for t, value in enumerate(values)], columns=["timestamp", "vm_id", "avg_cpu"])
    metadata = pd.DataFrame({"vm_id": ["a", "b", "c"], "vm_category": ["x", "x", "y"],
                             "virtual_core_bucket": ["2", "2", "4"], "memory_gb_bucket": ["4", "4", "8"]})
    return cpu, metadata


def test_hot_vm_concentration_and_temporal_variability():
    cpu, metadata = fixture()
    concentration, segments = workload_diagnostics(cpu, metadata, [0, 300, 600, 900])
    assert concentration["vm_count"] == 3
    assert concentration["high_readings"] == 6
    assert concentration["vms_ever_high"] == 2
    assert concentration["vms_high_at_least_half"] == 2
    assert concentration["vms_training_mean_ge80"] == 1
    assert concentration["top_group_vm_count"] == 1
    assert concentration["top_group_share_of_high_readings"] == pytest.approx(4 / 6)
    group = segments[(segments.dimension == "vm_category") & (segments.segment == "x")].iloc[0]
    assert group.vm_count == 2 and group.timestamps == 4
    assert group.mean_cpu_pct == 65
    assert group.segment_mean_temporal_std_pp == pytest.approx(np.std([45, 85, 45, 85], ddof=1))
    assert group.largest_absolute_change_pp == 40
    assert group.mean_vm_temporal_std_pp == pytest.approx(np.std([0, 80, 0, 80], ddof=1) / 2)


def test_no_high_readings_is_undefined_concentration_not_zero_evidence():
    cpu, metadata = fixture()
    cpu["avg_cpu"] = 1
    concentration, segments = workload_diagnostics(cpu, metadata, [0, 300, 600, 900])
    assert concentration["high_readings"] == 0
    assert concentration["vms_ever_high"] == 0
    assert concentration["top_group_share_of_high_readings"] is None
    assert segments.segment_mean_temporal_std_pp.eq(0).all()


def test_future_cpu_and_source_metadata_summaries_do_not_change_diagnostics():
    cpu, metadata = fixture()
    expected, expected_segments = workload_diagnostics(cpu, metadata, [0, 300, 600])
    cpu.loc[cpu.timestamp == 900, "avg_cpu"] = 1e9
    metadata["avg_cpu"] = 1e9
    actual, segments = workload_diagnostics(cpu, metadata, [0, 300, 600])
    assert actual == expected
    pd.testing.assert_frame_equal(segments, expected_segments)


def test_missing_timestamp_fails_without_silent_imputation():
    cpu, metadata = fixture()
    with pytest.raises(ValueError, match="Every requested training timestamp"):
        workload_diagnostics(cpu, metadata, [0, 300, 1200])
