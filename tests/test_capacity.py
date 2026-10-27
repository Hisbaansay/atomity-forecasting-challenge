from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomity_forecasting.capacity import horizon_headroom


@pytest.fixture
def inputs():
    rows = []
    for i, origin in enumerate((50, 70, 86)):
        for horizon in range(1, 25):
            # Early peak, smaller later errors, and entirely negative residuals.
            residual = [-4, 2, 10][i] if horizon == 1 else [-3, 1, 3][i]
            if horizon == 3:
                residual = -i - 1
            rows.append({
                "target": "aggregate_cpu_utilization", "model": "persistence",
                "origin": origin, "horizon": horizon,
                "timestamp": (origin + horizon - 1) * 300,
                "actual": 20 + residual, "prediction": 20.,
                "error": -residual,
            })
    predictions = pd.DataFrame({
        "timestamp": np.arange(110, 134) * 300,
        "active_vm_count": 10000,
        "aggregate_cpu_utilization": np.resize([0., 99., 100., 20.], 24),
    })
    return pd.DataFrame(rows), predictions


def test_manual_quantiles_envelope_and_clipping(inputs):
    backtests, predictions = inputs
    result = horizon_headroom(backtests, predictions)
    assert result.columns.tolist() == [
        "timestamp", "horizon", "minutes_ahead", "point_forecast", "validation_origins",
        "raw_buffer_p90_pp", "raw_buffer_p95_pp", "buffer_p90_pp", "buffer_p95_pp",
        "reference_p90_pct", "reference_p95_pct",
    ]
    np.testing.assert_array_equal(result.horizon, np.arange(1, 25))
    np.testing.assert_array_equal(result.minutes_ahead, np.arange(1, 25) * 5)
    np.testing.assert_array_equal(result.timestamp, predictions.timestamp)
    np.testing.assert_array_equal(result.point_forecast, predictions.aggregate_cpu_utilization)
    assert result.validation_origins.eq(3).all()
    for percentile in (90, 95):
        raw = result[f"raw_buffer_p{percentile}_pp"].to_numpy()
        expected = [np.quantile([0, 2, 10], percentile / 100)]
        expected += [np.quantile([0, 1, 3], percentile / 100), 0]
        expected += [expected[1]] * 21
        np.testing.assert_allclose(raw, expected)
        envelope = result[f"buffer_p{percentile}_pp"].to_numpy()
        np.testing.assert_array_equal(envelope, np.maximum.accumulate(raw))
        assert (envelope >= 0).all() and (np.diff(envelope) >= 0).all()
        reference = result[f"reference_p{percentile}_pct"]
        np.testing.assert_array_equal(reference, np.clip(result.point_forecast + envelope, 0, 100))
        assert reference.iloc[1] == reference.iloc[2] == 100
    assert (result.raw_buffer_p95_pp >= result.raw_buffer_p90_pp).all()
    assert (result.buffer_p95_pp >= result.buffer_p90_pp).all()
    assert result.raw_buffer_p90_pp.iloc[1] < result.raw_buffer_p90_pp.iloc[0]


def test_read_only_deterministic_selection_and_signed_error(inputs):
    backtests, predictions = inputs
    before_bt, before_preds = backtests.copy(deep=True), predictions.copy(deep=True)
    expected = horizon_headroom(backtests, predictions)
    pd.testing.assert_frame_equal(expected, horizon_headroom(backtests, predictions))
    pd.testing.assert_frame_equal(backtests, before_bt)
    pd.testing.assert_frame_equal(predictions, before_preds)
    other = backtests.assign(model="other", actual=np.nan, timestamp=999999)
    shuffled = pd.concat([backtests, other]).sample(frac=1, random_state=7)
    shuffled["error"] = 999  # Stored error is not our actual - prediction residual.
    pd.testing.assert_frame_equal(expected, horizon_headroom(shuffled, predictions))
    pd.testing.assert_frame_equal(expected, horizon_headroom(backtests.drop(columns="error"), predictions))


@pytest.mark.parametrize("fault", [
    "missing", "missing_all_origins", "origin", "fractional_origin", "duplicate",
    "horizon", "fractional_horizon", "future", "training_end", "timestamp",
    "duplicate_timestamp", "nan", "infinite", "cpu_low", "cpu_high", "no_model",
])
def test_invalid_backtests(inputs, fault):
    backtests, predictions = inputs
    if fault == "missing":
        backtests = backtests.iloc[1:]
    elif fault == "missing_all_origins":
        backtests = backtests.loc[backtests.horizon != 5]
    elif fault == "origin":
        backtests.loc[backtests.origin == 50, "origin"] = 51
    elif fault == "fractional_origin":
        backtests = backtests.astype({"origin": float})
        backtests.loc[0, "origin"] = 50.5
    elif fault == "duplicate":
        backtests = pd.concat([backtests.iloc[:-1], backtests.iloc[[0]]])
    elif fault in ("horizon", "fractional_horizon", "training_end"):
        backtests = backtests.astype({"horizon": float})
        backtests.loc[71, "horizon"] = {"horizon": 0, "fractional_horizon": 1.5, "training_end": 25}[fault]
    elif fault in ("future", "timestamp", "duplicate_timestamp"):
        backtests.loc[0, "timestamp"] = {
            "future": predictions.timestamp.iloc[0], "timestamp": 15001,
            "duplicate_timestamp": backtests.timestamp.iloc[1],
        }[fault]
    elif fault == "no_model":
        backtests["model"] = "other"
    else:
        backtests.loc[0, "prediction"] = {
            "nan": np.nan, "infinite": np.inf, "cpu_low": -1, "cpu_high": 101,
        }[fault]
    with pytest.raises(ValueError):
        horizon_headroom(backtests, predictions)


@pytest.mark.parametrize("fault", [
    "empty", "long", "reverse", "duplicate", "irregular", "nan_timestamp",
    "nan", "infinite", "low", "high", "observed", "missing_target",
])
def test_invalid_predictions(inputs, fault):
    backtests, predictions = inputs
    if fault == "empty":
        predictions = predictions.iloc[:0]
    elif fault == "long":
        predictions = pd.concat([predictions, predictions.iloc[[-1]]])
    elif fault == "reverse":
        predictions = predictions.iloc[::-1]
    elif fault in ("duplicate", "irregular", "nan_timestamp"):
        predictions.loc[1, "timestamp"] = {
            "duplicate": 33000, "irregular": 33301, "nan_timestamp": np.nan,
        }[fault]
    elif fault == "observed":
        predictions["actual"] = 20
    elif fault == "missing_target":
        predictions = predictions.drop(columns="aggregate_cpu_utilization")
    else:
        predictions.loc[0, "aggregate_cpu_utilization"] = {
            "nan": np.nan, "infinite": np.inf, "low": -1, "high": 101,
        }[fault]
    with pytest.raises(ValueError):
        horizon_headroom(backtests, predictions)


def test_single_horizon_and_training_boundary(inputs):
    backtests, predictions = inputs
    result = horizon_headroom(backtests.loc[backtests.horizon == 1], predictions.iloc[:1])
    assert len(result) == 1 and result.minutes_ahead.iloc[0] == 5
    # Origin 86 + horizon 24 = 110 is the last allowed training observation.
    assert backtests.iloc[-1].timestamp == 32700
    assert len(horizon_headroom(backtests, predictions)) == 24


def test_frozen_csv_precision():
    root = Path(__file__).resolve().parents[1]
    backtests = pd.read_csv(root / "reports/results/backtest_errors.csv", float_precision="round_trip")
    predictions = pd.read_csv(root / "predictions.csv", float_precision="round_trip")
    result = horizon_headroom(backtests, predictions)
    assert result.point_forecast.eq(9.164556174985147).all()
    np.testing.assert_array_equal(result.point_forecast, predictions.aggregate_cpu_utilization)
