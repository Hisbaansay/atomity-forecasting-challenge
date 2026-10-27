import json

import numpy as np
import pandas as pd
import pytest

from atomity_forecasting.backtesting import (
    backtest_origins, run_backtests, select_models, summarize_backtests,
)
from atomity_forecasting.models import CANDIDATES, TARGETS, forecast
from atomity_forecasting.predict import forecast_targets


def targets(n=134):
    t = np.arange(n)
    return pd.DataFrame({
        "timestamp": t * 300,
        "active_vm_count": 10 + t % 7,
        "aggregate_cpu_utilization": 40 + 15 * np.sin(t / 5),
    })


@pytest.mark.parametrize("model", CANDIDATES)
def test_candidates_deterministic_and_constrained(model):
    for target in TARGETS:
        result = forecast(np.arange(10) * 100 - 200, model, 24, target)
        np.testing.assert_array_equal(result, forecast(np.arange(10) * 100 - 200, model, 24, target))
        assert result.shape == (24,)
        assert np.isfinite(result).all() and (result >= 0).all()
        if target == "active_vm_count":
            np.testing.assert_array_equal(result, np.rint(result))
        else:
            assert (result <= 100).all()


def test_simple_candidate_values():
    np.testing.assert_array_equal(forecast([10, 20], "persistence", 4), [20] * 4)
    np.testing.assert_array_equal(forecast([10, 20], "mean_6", 4), [15] * 4)
    np.testing.assert_allclose(forecast([10, 20], "ses_0.2", 4), [12] * 4)
    np.testing.assert_array_equal(forecast([2.5], "persistence", 2, "active_vm_count"), [2, 2])


@pytest.mark.parametrize("replacement", [np.nan, 1e100, "not a label"])
def test_holdout_labels_never_read(replacement):
    original = targets()
    expected, expected_selection, expected_backtest = forecast_targets(original)
    changed = original.astype({target: object for target in TARGETS})
    changed.loc[110:, list(TARGETS)] = replacement
    actual, selection, backtest = forecast_targets(changed)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(expected_backtest, backtest)
    assert selection == expected_selection
    assert len(actual) == 24
    assert actual.columns.tolist() == ["timestamp", *TARGETS]
    json.dumps(selection, allow_nan=False)


def test_official_origins_and_long_schema():
    train = targets(110)
    result = run_backtests(train)
    assert backtest_origins(110) == (50, 70, 86)
    assert len(result) == 2 * len(CANDIDATES) * 3 * 24
    assert set(("target", "model", "origin", "horizon", "timestamp", "actual", "prediction", "error", "peak")) <= set(result)
    for origin in (50, 70, 86):
        for target in TARGETS:
            group = result[(result.origin == origin) & (result.target == target)]
            threshold = train[target].iloc[:origin].quantile(0.9)
            np.testing.assert_allclose(group.peak_threshold, threshold)
            np.testing.assert_array_equal(group.peak, group.actual >= threshold)
    np.testing.assert_allclose(result.error, result.prediction - result.actual)


def test_backtest_predictions_do_not_see_origin_future():
    original = targets(110)
    expected = run_backtests(original, origins=[50])
    changed = original.copy()
    changed.loc[50:, list(TARGETS)] = 999
    actual = run_backtests(changed, origins=[50])
    pd.testing.assert_frame_equal(
        expected[["prediction", "peak_threshold"]], actual[["prediction", "peak_threshold"]]
    )
    assert not expected.actual.equals(actual.actual)


def test_constant_ties_prefer_persistence_and_zero_baseline_is_null():
    frame = targets(16)
    frame[list(TARGETS)] = 10
    predictions, selection, backtest = forecast_targets(frame, 12, 4)
    assert len(predictions) == 4
    for target in TARGETS:
        chosen = selection["targets"][target]
        assert chosen["model"] == "persistence"
        assert chosen["metrics"]["MAE"] == 0
        assert chosen["metrics"]["relative_improvement"] is None
    assert all(summary is None for summary in summarize_backtests(backtest).relative_improvement)
    assert backtest_origins(9, 4) == (5,)


def test_pooled_metrics():
    frame = targets(12)
    summary = summarize_backtests(run_backtests(frame, 4))
    result = run_backtests(frame, 4)
    for row in summary.itertuples():
        group = result[(result.target == row.target) & (result.model == row.model)]
        assert row.MAE == pytest.approx(group.error.abs().mean())
        assert row.RMSE == pytest.approx(np.sqrt(np.mean(group.error ** 2)))
        assert row.bias == pytest.approx(group.error.mean())
        best_baseline = summary[(summary.target == row.target) & summary.model.isin(CANDIDATES[:4])].MAE.min()
        if best_baseline:
            assert row.relative_improvement == pytest.approx((best_baseline - row.MAE) / best_baseline)


@pytest.mark.parametrize("case", ["short", "irregular", "nan_train", "duplicate", "negative_cadence"])
def test_invalid_input(case):
    frame = targets(16)
    if case == "short":
        frame = frame.iloc[:-1]
    elif case == "irregular":
        frame.loc[15, "timestamp"] += 1
    elif case == "nan_train":
        frame.loc[5, "aggregate_cpu_utilization"] = np.nan
    elif case == "duplicate":
        frame.loc[5, "timestamp"] = frame.loc[4, "timestamp"]
    else:
        frame["timestamp"] *= -1
    with pytest.raises(ValueError):
        forecast_targets(frame, 12, 4)


def test_selection_requires_complete_internal_horizon():
    with pytest.raises(ValueError, match="five fitting rows"):
        select_models(targets(8), 4)


def test_forecast_and_backtest_share_postprocessing():
    frame = targets(16)
    frame["active_vm_count"] = np.arange(16) * 0.7
    frame["aggregate_cpu_utilization"] = np.arange(16) * 20
    result = run_backtests(frame.iloc[:12], 4)
    for (target, model, origin), group in result.groupby(["target", "model", "origin"]):
        np.testing.assert_array_equal(
            group.prediction, forecast(frame[target].iloc[:origin], model, 4, target)
        )


def test_vm_target_construction_and_future_cpu_perturbation():
    data = pytest.importorskip("atomity_forecasting.data")
    cpu = pd.DataFrame([
        {"timestamp": step * 300, "vm_id": vm, "min_cpu": 0,
         "avg_cpu": float(10 + step + vm), "max_cpu": 100}
        for step in range(16) for vm in range(2)
    ])
    aggregate = data.construct_targets(cpu)
    np.testing.assert_array_equal(aggregate.active_vm_count, 2)
    np.testing.assert_allclose(aggregate.aggregate_cpu_utilization, np.arange(16) + 10.5)
    expected, selection, backtest = forecast_targets(aggregate, 12, 4)
    changed = cpu.copy()
    changed.loc[changed.timestamp >= 12 * 300, "avg_cpu"] = 99.0
    actual, actual_selection, actual_backtest = forecast_targets(data.construct_targets(changed), 12, 4)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(backtest, actual_backtest)
    assert selection == actual_selection


def test_no_validation_peaks_reports_null():
    frame = targets(9)
    frame.loc[:4, list(TARGETS)] = 90
    frame.loc[5:, list(TARGETS)] = 1
    selection, backtest = select_models(frame, 4)
    assert not backtest.peak.any()
    for target in TARGETS:
        assert selection["targets"][target]["metrics"]["peak_MAE"] is None
    json.dumps(selection, allow_nan=False)
