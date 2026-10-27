from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomity_forecasting import backtesting, evaluation
from atomity_forecasting.backtesting import run_backtests, select_models
from atomity_forecasting.evaluation import (
    detailed_origin_metrics, diagnostic_horizon_metrics, validation_sensitivity,
)
from atomity_forecasting.models import BASELINES, CANDIDATES, TARGETS


@pytest.fixture
def train():
    t = np.arange(110)
    return pd.DataFrame({
        "timestamp": t * 300,
        "active_vm_count": 10 + t % 7,
        "aggregate_cpu_utilization": 40 + 15 * np.sin(t / 5),
    })


def test_sensitivity_schedules_order_and_read_only(train, monkeypatch):
    before = train.copy(deep=True)
    calls = []

    def record(frame, horizon, origins):
        assert frame is train
        assert all(5 <= origin and origin + horizon <= 110 for origin in origins)
        calls.append(origins)
        return run_backtests(frame, horizon, origins)

    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostics must never call model selection")

    monkeypatch.setattr(evaluation, "run_backtests", record)
    monkeypatch.setattr(backtesting, "select_models", forbidden)
    monkeypatch.setattr(evaluation, "select_models", forbidden, raising=False)
    result = validation_sensitivity(train)
    assert calls == [(50, 70, 86), (38, 62, 86), (30, 54, 78)]
    assert list(result) == [
        "schedule", "origins", "target", "model", "MAE", "RMSE", "bias",
        "peak_MAE", "relative_improvement", "preferred", "persistence_MAE",
        "training_mean", "normalized_MAE", "normalized_MAE_pct",
    ]
    assert len(result) == 3 * len(TARGETS) * len(CANDIDATES)
    for (_, target), group in result.groupby(["schedule", "target"], sort=False):
        assert target in TARGETS
        assert group.model.tolist() == list(CANDIDATES)
        assert group.preferred.sum() == 1
        assert group.loc[group.preferred, "model"].iloc[0] == group.loc[group.MAE.idxmin(), "model"]
    pd.testing.assert_frame_equal(result, validation_sensitivity(train))
    pd.testing.assert_frame_equal(train, before)


def test_sensitivity_independent_metrics(train):
    result = validation_sensitivity(train)
    for schedule, scores in result.groupby("schedule", sort=False):
        origins = tuple(map(int, scores.origins.iloc[0].split(",")))
        raw = run_backtests(train, origins=origins)
        for row in scores.itertuples():
            group = raw[(raw.target == row.target) & (raw.model == row.model)]
            error = (group.prediction - group.actual).to_numpy()
            mae = np.abs(error).mean()
            assert row.MAE == pytest.approx(mae)
            assert row.RMSE == pytest.approx(np.sqrt(np.mean(error ** 2)))
            assert row.bias == pytest.approx(error.mean())
            peaks = error[group.peak.to_numpy()]
            if len(peaks):
                assert row.peak_MAE == pytest.approx(np.abs(peaks).mean())
            else:
                assert pd.isna(row.peak_MAE)
            persistence = raw[(raw.target == row.target) & (raw.model == "persistence")]
            assert row.persistence_MAE == pytest.approx((persistence.prediction - persistence.actual).abs().mean())
            baseline = scores[(scores.target == row.target) & scores.model.isin(BASELINES)].MAE.min()
            assert row.relative_improvement == pytest.approx((baseline - mae) / baseline)
            assert row.training_mean == train[row.target].mean()
            assert row.normalized_MAE == pytest.approx(mae / train[row.target].mean())
            assert row.normalized_MAE_pct == pytest.approx(100 * mae / train[row.target].mean())


@pytest.mark.parametrize("length", [109, 114, 134])
def test_sensitivity_rejects_nontraining_lengths(train, length, monkeypatch):
    frame = train.reindex(range(length))
    monkeypatch.setattr(evaluation, "run_backtests", lambda *a, **kw: pytest.fail("Must reject before backtesting"))
    with pytest.raises(ValueError, match="exactly 110 training rows"):
        validation_sensitivity(frame)


@pytest.mark.parametrize("horizon", [0, -1, 25, 24.0, True])
def test_sensitivity_rejects_invalid_windows(train, horizon):
    with pytest.raises(ValueError):
        validation_sensitivity(train, horizon)


def test_shorter_horizon_and_invalid_targets(train):
    assert len(validation_sensitivity(train, horizon=6)) == 42
    train.loc[0, "aggregate_cpu_utilization"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validation_sensitivity(train)


def test_zero_denominator_and_stable_ties(train):
    train[list(TARGETS)] = 0
    result = validation_sensitivity(train)
    assert result.normalized_MAE.isna().all()
    assert result.normalized_MAE_pct.isna().all()
    assert result.relative_improvement.isna().all()
    assert result.loc[result.preferred, "model"].tolist() == ["persistence"] * 6
    train[list(TARGETS)] = 10
    result = validation_sensitivity(train)
    assert (result.normalized_MAE == 0).all()
    assert result.loc[result.preferred, "model"].tolist() == ["persistence"] * 6


def test_origin_and_horizon_metrics_independent_and_read_only(train):
    selection, raw = select_models(train)
    raw = raw.sample(frac=1, random_state=7)
    before = raw.copy(deep=True)
    selection_before = deepcopy(selection)
    detailed = detailed_origin_metrics(raw, selection)
    diagnostic = diagnostic_horizon_metrics(raw, selection)
    assert len(detailed) == 6
    assert len(diagnostic) == 2 * 3 * 24
    np.testing.assert_allclose(diagnostic.absolute_error, (diagnostic.prediction - diagnostic.actual).abs())
    np.testing.assert_allclose(diagnostic.error, diagnostic.prediction - diagnostic.actual)
    for row in detailed.itertuples():
        assert row.model == selection["targets"][row.target]["model"]
        group = raw[(raw.target == row.target) & (raw.model == row.model) & (raw.origin == row.origin)]
        error = group.prediction - group.actual
        assert row.MAE == pytest.approx(error.abs().mean())
        assert row.RMSE == pytest.approx(np.sqrt(np.mean(error ** 2)))
        assert row.bias == pytest.approx(error.mean())
        assert row.max_absolute_error == pytest.approx(error.abs().max())
        assert row.max_overprediction == pytest.approx(error.clip(lower=0).max())
        assert row.max_underprediction == pytest.approx((-error).clip(lower=0).max())
        assert row.peak_threshold == pytest.approx(train[row.target].iloc[:row.origin].quantile(.9))
        peaks = error[group.actual >= row.peak_threshold]
        assert row.peak_n == len(peaks)
        if len(peaks):
            assert row.peak_MAE == pytest.approx(peaks.abs().mean())
            assert row.peak_bias == pytest.approx(peaks.mean())
        else:
            assert pd.isna(row.peak_MAE) and pd.isna(row.peak_bias)
        assert row.first6_MAE == pytest.approx(error[group.horizon <= 6].abs().mean())
        assert row.last6_MAE == pytest.approx(error[group.horizon >= 19].abs().mean())
    pd.testing.assert_frame_equal(raw, before)
    assert selection == selection_before
    diagnostic.loc[:, "actual"] = -999
    pd.testing.assert_frame_equal(raw, before)
    pd.testing.assert_frame_equal(detailed, detailed_origin_metrics(raw, selection))


def test_missing_peaks_are_null(train):
    train.loc[:85, list(TARGETS)] = 90
    train.loc[86:, list(TARGETS)] = 1
    raw = run_backtests(train, origins=(86,))
    selection = {"targets": {target: {"model": "persistence"} for target in TARGETS}}
    result = detailed_origin_metrics(raw, selection)
    assert (result.peak_n == 0).all()
    assert result[["peak_MAE", "peak_bias"]].isna().all().all()


def test_sensitivity_does_not_change_selection_or_forecast(train):
    from atomity_forecasting.predict import forecast_targets

    frame = train.reindex(range(134))
    frame["timestamp"] = np.arange(134) * 300
    expected, selection, raw = forecast_targets(frame)
    validation_sensitivity(train)
    actual, after_selection, after_raw = forecast_targets(frame)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(raw, after_raw)
    assert selection == after_selection


def test_actual_training_cpu_regression():
    path = Path(__file__).resolve().parents[1] / "reports/results/targets_train.csv"
    if not path.exists():
        pytest.skip("Actual training artifact not available")
    result = validation_sensitivity(pd.read_csv(path))
    cpu = result[result.target == "aggregate_cpu_utilization"]
    for schedule, model, expected in [
        ("original", "persistence", .235310),
        ("alternative_a", "mean_12", .178385),
        ("alternative_a", "persistence", .208198),
        ("alternative_b", "damped_trend", .137625),
        ("alternative_b", "persistence", .181039),
    ]:
        value = cpu.loc[(cpu.schedule == schedule) & (cpu.model == model), "MAE"].iloc[0]
        assert value == pytest.approx(expected, abs=5e-7)
