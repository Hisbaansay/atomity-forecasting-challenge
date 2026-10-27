import numpy as np
import pandas as pd
import pytest

from atomity_forecasting.cross_section import cross_sectional_diagnostics


def readings():
    return pd.DataFrame([
        (0, "a", 0, 20, 5), (0, "b", 0, 100, 10), (0, "c", 0, 100, 90),
        (300, "a", 0, 30, 6), (300, "b", 0, 100, 10), (300, "c", 0, 100, 90),
    ], columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def test_equal_weight_duplicates_counts_quantiles_and_classes():
    cpu = readings()
    # Duplicates average to 10 for b, rather than giving b additional VM weight.
    cpu.loc[1, "avg_cpu"] = 0
    cpu = pd.concat([cpu, readings().iloc[[1]].assign(avg_cpu=20)])
    original = cpu.copy(deep=True)
    with pytest.warns(UserWarning, match="cleaning"):
        pertime, profile, summary = cross_sectional_diagnostics(cpu, [0, 300])
    pd.testing.assert_frame_equal(cpu, original)
    assert pertime.timestamp.tolist() == [0, 300]
    assert pertime.reporting_vm_count.tolist() == [3, 3]
    assert pertime.workload_active_vm_count.tolist() == [2, 3]
    np.testing.assert_allclose(pertime.mean_avg_cpu, [35, 106 / 3])
    quantiles = pertime[[f"p{q}_avg_cpu" for q in (50, 75, 90, 95, 99)]
                       + ["max_avg_cpu"]]
    assert (np.diff(quantiles.to_numpy(), axis=1) >= 0).all()
    assert pertime.p50_avg_cpu.tolist() == [10, 10]
    assert pertime.max_avg_cpu.tolist() == [90, 90]
    vm = profile.set_index("vm_id")
    assert vm.observations.tolist() == [2, 2, 2]
    assert vm.mean_load_class.tolist() == ["moderate", "moderate", "high"]
    assert vm.bursty.tolist() == [True, False, False]
    assert vm.bursty_std_threshold_pp.nunique() == 1
    assert vm.loc["a", "bursty_std_threshold_pp"] == pytest.approx(np.sqrt(.5) * .8)
    exclusive = summary[summary.classification.eq("exclusive_mean")]
    assert exclusive.vm_count.tolist() == [0, 2, 1]
    assert exclusive.share_vms.sum() == pytest.approx(1)
    assert summary.set_index("class").loc["interval_max_gt80", "vm_count"] == 2
    with pytest.warns(UserWarning):
        reordered = cross_sectional_diagnostics(cpu.sample(frac=1, random_state=3), [0, 300])
    for expected, actual in zip((pertime, profile, summary), reordered):
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)


def test_boundaries_and_bursty_ties():
    cpu = pd.DataFrame([
        (t, vm, 0, maximum, avg)
        for t in (0, 300)
        for vm, avg, maximum in [("a", 4, 20), ("b", 5, 21), ("c", 50, 80),
                                 ("d", 51, 81), ("e", 80, 80), ("f", 81, 81),
                                 ("g", 6, 20)]
    ], columns=readings().columns)
    pertime, profile, summary = cross_sectional_diagnostics(cpu, [0, 300])
    assert pertime.workload_active_vm_count.tolist() == [4, 4]
    assert profile.mean_load_class.tolist() == [
        "idle_low", "moderate", "moderate", "high", "high", "high", "moderate"]
    assert profile.hot_avg_gt80.tolist() == [False] * 5 + [True, False]
    assert profile.interval_max_gt80.tolist() == [False, False, False, True, False, True, False]
    assert profile.bursty_std_threshold_pp.eq(0).all()
    assert not profile.bursty.any()
    cpu.loc[cpu.timestamp.eq(300), "avg_cpu"] -= 1
    _, profile, summary = cross_sectional_diagnostics(cpu, [0, 300])
    assert profile.bursty.all()  # Equal positive stds satisfy >= q90, even with ties.
    assert summary.set_index("class").loc["bursty", "share_vms"] == 1


@pytest.mark.parametrize("future_value", [np.nan, np.inf, -1e12, 1e12])
def test_future_values_and_identities_are_ignored(future_value):
    cpu = readings()
    expected = cross_sectional_diagnostics(cpu, [0, 300])
    future = readings().assign(timestamp=lambda x: x.timestamp + 600,
                               vm_id=None, avg_cpu=future_value,
                               min_cpu=future_value, max_cpu=future_value)
    extended = pd.concat([cpu, future], ignore_index=True)
    original = extended.copy(deep=True)
    actual = cross_sectional_diagnostics(extended, [0, 300])
    for left, right in zip(expected, actual):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    pd.testing.assert_frame_equal(extended, original)


@pytest.mark.parametrize("grid", [[], [0], [300, 0], [0, 0], [0, 600],
                                   [0., 300.], [False, 300], [-300, 0],
                                   list(range(0, 111 * 300, 300))])
def test_invalid_training_grid(grid):
    with pytest.raises(ValueError, match="Training grid"):
        cross_sectional_diagnostics(readings(), grid)


def test_missing_columns_timestamps_and_nonprefix():
    with pytest.raises(ValueError, match="Missing CPU columns"):
        cross_sectional_diagnostics(readings().drop(columns="min_cpu"), [0, 300])
    with pytest.raises(ValueError, match="Every requested"):
        cross_sectional_diagnostics(readings(), [0, 300, 600])
    with pytest.raises(ValueError, match="first input timestamp"):
        cross_sectional_diagnostics(readings(), [300, 600])


def test_full_training_grid_is_supported():
    grid = list(range(0, 110 * 300, 300))
    cpu = pd.DataFrame([(t, "a", 0, 10, 5) for t in grid], columns=readings().columns)
    pertime, profile, _ = cross_sectional_diagnostics(cpu, grid)
    assert pertime.timestamp.tolist() == grid
    assert profile.observations.tolist() == [110]


def test_single_observation_std_and_training_invalid_readings():
    cpu = readings().drop(index=3)
    _, profile, _ = cross_sectional_diagnostics(cpu, [0, 300])
    singleton = profile.set_index("vm_id").loc["a"]
    assert singleton.observations == 1
    assert pd.isna(singleton.avg_cpu_std)
    assert not singleton.bursty
    cpu.loc[cpu.timestamp.eq(300), "avg_cpu"] = np.nan
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no valid"):
        cross_sectional_diagnostics(cpu, [0, 300])
