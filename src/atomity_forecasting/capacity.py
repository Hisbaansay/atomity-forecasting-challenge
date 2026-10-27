"""Training-only, descriptive CPU headroom references, not capacity estimates."""

import numpy as np
import pandas as pd


def horizon_headroom(
    backtests: pd.DataFrame,
    predictions: pd.DataFrame,
    model: str = "persistence",
    target: str = "aggregate_cpu_utilization",
) -> pd.DataFrame:
    """Return uncalibrated one-sided CPU buffers in percentage points.

    Use only the selected target/model's original training backtests at prefix
    origins 50, 70 and 86. Each must contain exactly horizons 1..len(predictions),
    with origin + horizon <= 110. Numeric timestamps must follow the original
    300-second grid, with predictions starting immediately after training.
    Predictions are a wide forecast-only frame (timestamp and forecast targets),
    never future observed labels. Their provenance cannot be inferred from values;
    callers must supply forecasts, not observed CPU disguised as forecasts.

    The signed residual here is actual - prediction, opposite to the stored
    backtest `error` convention. Quantiles use its positive part, including zeros
    for overprediction, and NumPy's default linear interpolation. There are only
    THREE samples per horizon, from overlapping windows: raw P90/P95 names do not
    imply calibrated coverage, prediction intervals, or a future SLA guarantee.

    Cumulative maxima of the raw quantiles are a deliberately conservative display
    heuristic. They do not assert that empirical raw risk increases with horizon.
    References clip point forecast + envelope to [0, 100]; clipping loses headroom
    information. Mean CPU utilization is not absolute demand: this does not call
    provisioned_capacity, recommend automatic scaling, or imply a cost ratio.
    Inputs, model selection and frozen point predictions are never changed.
    """
    required = {"target", "model", "origin", "horizon", "timestamp", "actual", "prediction"}
    if not backtests.columns.is_unique or not required.issubset(backtests.columns):
        raise ValueError(f"Backtests require unique columns including {sorted(required)}")
    forecast_columns = {"timestamp", "active_vm_count", "aggregate_cpu_utilization"}
    if (
        target != "aggregate_cpu_utilization"
        or not predictions.columns.is_unique
        or not {"timestamp", target}.issubset(predictions.columns)
        or not set(predictions.columns).issubset(forecast_columns)
    ):
        raise ValueError("Predictions must contain only timestamps and forecast targets; CPU target required")
    size = len(predictions)
    if not 1 <= size <= 24:
        raise ValueError("Predictions require 1..24 horizons within the original training backtests")
    timestamps = predictions["timestamp"].to_numpy(dtype=float)
    points = predictions[target].to_numpy(dtype=float)
    if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) == 300):
        raise ValueError("Prediction timestamps must be strictly increasing on a 300-second grid")
    if not np.isfinite(points).all() or np.any((points < 0) | (points > 100)):
        raise ValueError("CPU point forecasts must be finite and within [0, 100]")

    selected = backtests.loc[(backtests["model"] == model) & (backtests["target"] == target)]
    values = selected[["origin", "horizon", "timestamp", "actual", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Selected backtest values must be finite")
    origins, horizons, times, actual, fitted = values.T
    if (
        len(selected) != 3 * size
        or set(origins) != {50, 70, 86}
        or not np.equal(horizons, np.floor(horizons)).all()
        or np.any((horizons < 1) | (horizons > size))
        or selected.duplicated(["origin", "horizon"]).any()
    ):
        raise ValueError("Require origins 50/70/86, each with exactly one row per forecast horizon")
    if np.any(origins + horizons > 110) or np.any(times >= timestamps[0]):
        raise ValueError("Backtest validation must lie entirely within the original 110 training rows")
    expected_times = timestamps[0] + (origins + horizons - 111) * 300
    if not np.array_equal(times, expected_times):
        raise ValueError("Backtest timestamps must match origin/horizon on the original training grid")
    if np.any((actual < 0) | (actual > 100) | (fitted < 0) | (fitted > 100)):
        raise ValueError("Backtest CPU actuals and predictions must be within [0, 100]")

    steps = np.arange(1, size + 1)
    positive_errors = np.maximum(actual - fitted, 0)
    raw = np.array([
        np.quantile(positive_errors[horizons == step], [0.90, 0.95])
        for step in steps
    ])
    envelope = np.maximum.accumulate(raw, axis=0)
    return pd.DataFrame({
        "timestamp": predictions["timestamp"].to_numpy(copy=True),
        "horizon": steps,
        "minutes_ahead": steps * 5,
        "point_forecast": points,
        "validation_origins": np.full(size, 3),
        "raw_buffer_p90_pp": raw[:, 0],
        "raw_buffer_p95_pp": raw[:, 1],
        "buffer_p90_pp": envelope[:, 0],
        "buffer_p95_pp": envelope[:, 1],
        "reference_p90_pct": np.clip(points + envelope[:, 0], 0, 100),
        "reference_p95_pct": np.clip(points + envelope[:, 1], 0, 100),
    })
