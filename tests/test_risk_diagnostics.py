import numpy as np
import pandas as pd
import pytest

from atomity_forecasting import evaluation
from atomity_forecasting.evaluation import change_error_summary, directional_error_metrics
from atomity_forecasting.models import CANDIDATES, TARGETS


@pytest.fixture
def inputs():
    changes = np.resize([0., 1., -1., 2., -2.], 110)
    for origin in (50, 70, 86):
        changes[origin:origin + 4] = [10, 1, 0, -2]
    values = 100 + np.cumsum(changes)
    train = pd.DataFrame({"timestamp": np.arange(110) * 300,
                          TARGETS[0]: values, TARGETS[1]: values * 2})
    rows = []
    for target in TARGETS:
        for model in CANDIDATES:
            for origin in (50, 70, 86):
                for step, error in enumerate([-4., 2., 0., -2.], 1):
                    actual = train[target].iloc[origin + step - 1]
                    rows.append({"target": target, "model": model, "origin": origin,
                                 "horizon": step, "timestamp": (origin + step - 1) * 300,
                                 "actual": actual, "prediction": actual + error,
                                 "error": error, "peak": False})
    return train, pd.DataFrame(rows).sample(frac=1, random_state=4)


def test_directional_manual_metrics_and_read_only(inputs, monkeypatch):
    train, raw = inputs
    before = raw.copy(deep=True)
    monkeypatch.setattr(evaluation, "run_backtests", lambda *a, **k: pytest.fail("No refitting"))
    result = directional_error_metrics(raw)
    assert len(result) == 56
    assert result.loc[result.scope == "pooled", "origin"].isna().all()
    for row in result.itertuples():
        assert row.n == (12 if row.scope == "pooled" else 4)
        assert row.MAE == 2
        assert row.RMSE == pytest.approx(np.sqrt(6))
        assert row.bias == -1
        assert row.max_absolute_error == 4
        assert row.peak_n == 0 and pd.isna(row.peak_MAE)
        assert row.underprediction_count == row.n // 2
        assert row.underprediction_rate == .5
        assert row.mean_underprediction == 1.5
        assert row.max_underprediction == 4
        assert row.underprediction_q95 == pytest.approx(4 if row.scope == "pooled" else 3.7)
        assert row.first6_MAE == row.last6_MAE == 2
    pd.testing.assert_frame_equal(result, directional_error_metrics(raw.drop(columns="error")))
    pd.testing.assert_frame_equal(raw, before)


def test_directional_horizon_masks_and_existing_error(inputs):
    _, raw = inputs
    raw.loc[raw.horizon == 4, "horizon"] = 24
    raw.loc[raw.horizon == 24, "error"] = -8
    result = directional_error_metrics(raw)
    assert (result.first6_MAE == 2).all()
    assert (result.last6_MAE == 8).all()
    raw["error"] = 0
    raw["peak"] = True
    result = directional_error_metrics(raw)
    assert (result.peak_MAE == 0).all()
    assert (result.underprediction_count == 0).all()
    assert (result.mean_underprediction == 0).all()


def test_change_labels_manual_counts_all_candidates_and_read_only(inputs, monkeypatch):
    train, raw = inputs
    train_before, raw_before = train.copy(deep=True), raw.copy(deep=True)
    monkeypatch.setattr(evaluation, "run_backtests", lambda *a, **k: pytest.fail("No refitting"))
    steps, summary = change_error_summary(train, raw)
    assert len(steps) == len(raw) and len(summary) == 56 * 3
    pd.testing.assert_frame_equal(steps[raw.columns], raw)
    for (target, origin), group in steps.groupby(["target", "origin"]):
        differences = train[target].iloc[:origin].diff().dropna()
        threshold = np.percentile(differences[differences > 0], 90)
        assert (group.large_upward_threshold == threshold).all()
        assert group.groupby("timestamp").change_group.nunique().eq(1).all()
        scale = 1 if target == TARGETS[0] else 2
        for row in group.itertuples():
            assert row.actual_change == [10, 1, 0, -2][row.horizon - 1] * scale
            assert row.change_group == ["large_upward_change", "smaller_upward_change",
                                        "nonpositive_change", "nonpositive_change"][row.horizon - 1]
    for row in summary.itertuples():
        multiplier = 3 if row.scope == "pooled" else 1
        n, mae, mean, rate, maximum = {
            "large_upward_change": (1, 4, 4, 1, 4),
            "smaller_upward_change": (1, 2, 0, 0, 0),
            "nonpositive_change": (2, 1, 1, .5, 2),
        }[row.change_group]
        assert (row.n, row.MAE, row.mean_underprediction,
                row.underprediction_rate, row.max_underprediction) == (n * multiplier, mae, mean, rate, maximum)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(raw, raw_before)
    derived, derived_summary = change_error_summary(train, raw.drop(columns="error"))
    pd.testing.assert_frame_equal(steps.drop(columns="error"), derived)
    pd.testing.assert_frame_equal(summary, derived_summary)


def test_threshold_past_only_and_boundary(inputs):
    train, raw = inputs
    raw = raw.loc[raw.origin == 50].copy()
    before, _ = change_error_summary(train, raw)
    train.loc[50:, list(TARGETS)] += 1000
    for target in TARGETS:
        mask = raw.target == target
        raw.loc[mask, "actual"] = train[target].iloc[
            (raw.loc[mask, "origin"] + raw.loc[mask, "horizon"] - 1).to_numpy()
        ].to_numpy()
    after, _ = change_error_summary(train, raw)
    pd.testing.assert_series_equal(before.large_upward_threshold, after.large_upward_threshold)
    assert (after.loc[after.horizon == 1, "actual_change"] > 1000).all()
    # A positive change equal to the threshold is not large.
    train[list(TARGETS)] = np.arange(110)[:, None] * np.ones((1, 2))
    raw["actual"] = raw.origin + raw.horizon - 1
    steps, summary = change_error_summary(train, raw)
    assert steps.change_group.eq("smaller_upward_change").all()
    empty = summary.loc[summary.n == 0]
    assert len(empty) > 0
    assert empty[["MAE", "mean_underprediction", "underprediction_rate", "max_underprediction"]].isna().all().all()


def test_no_positive_history(inputs):
    train, raw = inputs
    raw = raw.loc[raw.origin == 50].copy()
    train[list(TARGETS)] = 10.
    train.loc[50:, list(TARGETS)] = 11.
    raw["actual"] = 11.
    steps, _ = change_error_summary(train, raw)
    assert steps.large_upward_threshold.isna().all()
    assert steps.loc[steps.horizon == 1, "change_group"].eq("smaller_upward_change").all()
    assert steps.loc[steps.horizon > 1, "change_group"].eq("nonpositive_change").all()


@pytest.mark.parametrize("fault", ["134", "109", "timestamp", "origin", "horizon", "actual", "cadence"])
def test_invalid_training_or_backtest(inputs, fault):
    train, raw = inputs
    if fault in ("134", "109"):
        train = train.reindex(range(int(fault)))
    elif fault == "timestamp":
        raw.iloc[0, raw.columns.get_loc("timestamp")] += 300
    elif fault == "origin":
        raw["origin"] = 87
    elif fault == "horizon":
        raw["horizon"] = 25
    elif fault == "actual":
        raw["actual"] += 1
    else:
        train["timestamp"] *= 2
    with pytest.raises(ValueError):
        change_error_summary(train, raw)
