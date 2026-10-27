"""Read-only training diagnostics, independent of model selection and forecasting."""

import numpy as np
import pandas as pd

from .backtesting import run_backtests, summarize_backtests, validate_targets
from .models import TARGETS

VALIDATION_SCHEDULES = {
    "original": (50, 70, 86),
    "alternative_a": (38, 62, 86),
    "alternative_b": (30, 54, 78),
}


def validation_sensitivity(train: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Compare fixed schedules on exactly the caller's first 110 training rows.

    Never truncate a larger frame: callers must exclude holdout rows explicitly.
    ``preferred`` marks one minimum-MAE candidate per target/schedule, breaking
    ties in CANDIDATES order. It is diagnostic only, not a selection update.
    Normalized MAE uses the full supplied target mean, not validation means.
    """
    if len(train) != 110:
        raise ValueError("Supply exactly 110 training rows, excluding all holdout rows")
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    schedules = VALIDATION_SCHEDULES
    if any(origin < 5 or origin + horizon > len(train)
           for origins in schedules.values() for origin in origins):
        raise ValueError("All fixed origin windows must lie inside the 110 training rows")
    validate_targets(train)
    results = []
    for schedule, origins in schedules.items():
        summary = summarize_backtests(run_backtests(train, horizon=horizon, origins=origins))
        summary.insert(0, "origins", ",".join(map(str, origins)))
        summary.insert(0, "schedule", schedule)
        summary["preferred"] = False
        for target in TARGETS:
            mask = summary.target == target
            summary.loc[summary.loc[mask, "MAE"].idxmin(), "preferred"] = True
            persistence = summary.loc[mask & (summary.model == "persistence"), "MAE"].iloc[0]
            mean = float(train[target].mean())
            summary.loc[mask, "persistence_MAE"] = persistence
            summary.loc[mask, "training_mean"] = mean
            summary.loc[mask, "normalized_MAE"] = summary.loc[mask, "MAE"] / mean if mean != 0 else np.nan
        summary["normalized_MAE_pct"] = 100 * summary.normalized_MAE
        results.append(summary)
    return pd.concat(results, ignore_index=True)


def diagnostic_horizon_metrics(backtest: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """Copy selected-model backtest rows, adding absolute error.

    ``selection`` is the unchanged artifact returned by ``select_models``.
    ``error`` remains the signed error (prediction minus actual); all original
    columns, including per-origin peak thresholds, are retained.
    """
    mask = pd.Series(False, index=backtest.index)
    for target in TARGETS:
        model = selection["targets"][target]["model"]
        mask |= (backtest.target == target) & (backtest.model == model)
    result = backtest.loc[mask].copy()
    result["absolute_error"] = result.error.abs()
    return result


def detailed_origin_metrics(backtest: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """Summarize each selected model/origin without refitting or selecting.

    First/last-six metrics use horizon numbers, independent of input row order.
    Missing peaks produce null peak MAE/bias and a zero peak count.
    """
    selected = diagnostic_horizon_metrics(backtest, selection)
    rows = []
    for (target, model, origin), group in selected.groupby(["target", "model", "origin"], sort=False):
        error = group.error
        peaks = group.loc[group.peak, "error"]
        rows.append({
            "target": target, "model": model, "origin": origin,
            "MAE": float(error.abs().mean()),
            "RMSE": float(np.sqrt(np.square(error).mean())),
            "bias": float(error.mean()),
            "max_absolute_error": float(error.abs().max()),
            "max_overprediction": float(error.clip(lower=0).max()),
            "max_underprediction": float((-error).clip(lower=0).max()),
            "peak_threshold": float(group.peak_threshold.iloc[0]),
            "peak_n": len(peaks),
            "peak_MAE": float(peaks.abs().mean()) if len(peaks) else None,
            "peak_bias": float(peaks.mean()) if len(peaks) else None,
            "first6_MAE": float(group.loc[group.horizon <= 6, "absolute_error"].mean()),
            "last6_MAE": float(group.loc[group.horizon > group.horizon.max() - 6, "absolute_error"].mean()),
        })
    return pd.DataFrame(rows, columns=[
        "target", "model", "origin", "MAE", "RMSE", "bias", "max_absolute_error",
        "max_overprediction", "max_underprediction",
        "peak_threshold", "peak_n", "peak_MAE", "peak_bias", "first6_MAE", "last6_MAE",
    ])


def directional_error_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    """Describe all supplied candidates, per origin and pooled, without selection.

    Signed error is prediction minus actual. One-sided losses include zeros for
    correct/overpredicted observations, including in the 95th percentile. Pooled
    rows count overlapping validation windows separately. First/last-six masks
    are defined within each origin, not by input row order.
    """
    frame = backtest.copy()
    if "error" not in frame:
        frame["error"] = frame.prediction - frame.actual
    frame["underprediction"] = (-frame.error).clip(lower=0)
    frame["last6"] = frame.horizon > frame.groupby(
        ["target", "model", "origin"]
    ).horizon.transform("max") - 6
    rows = []
    for scope, keys in [("origin", ["target", "model", "origin"]),
                        ("pooled", ["target", "model"])]:
        for key, group in frame.groupby(keys, sort=False):
            error = group.error
            under = group.underprediction
            peaks = error[group.peak]
            rows.append({
                "target": key[0], "model": key[1], "scope": scope,
                "origin": key[2] if scope == "origin" else np.nan,
                "n": len(group), "MAE": error.abs().mean(),
                "RMSE": np.sqrt(np.square(error).mean()), "bias": error.mean(),
                "max_absolute_error": error.abs().max(), "peak_n": len(peaks),
                "max_overprediction": error.clip(lower=0).max(),
                "peak_MAE": peaks.abs().mean(),
                "underprediction_count": int((under > 0).sum()),
                "underprediction_rate": (under > 0).mean(),
                "mean_underprediction": under.mean(),
                "underprediction_q95": under.quantile(.95),
                "max_underprediction": under.max(),
                "first6_MAE": error[group.horizon <= 6].abs().mean(),
                "last6_MAE": error[group.last6].abs().mean(),
            })
    return pd.DataFrame(rows, columns=[
        "target", "model", "scope", "origin", "n", "MAE", "RMSE", "bias",
        "max_absolute_error", "peak_n", "max_overprediction", "peak_MAE", "underprediction_count",
        "underprediction_rate", "mean_underprediction", "underprediction_q95",
        "max_underprediction", "first6_MAE", "last6_MAE",
    ])


def schedule_diagnostics(train: pd.DataFrame) -> pd.DataFrame:
    """Return identical origin/pooled metrics across fixed training-only schedules.

    This reporting function does not update the final selection. Pooled rows
    score different periods across schedules and are not independent tests.
    """
    if len(train) != 110:
        raise ValueError("Supply exactly 110 training rows, excluding all holdout rows")
    rows = []
    for schedule, origins in VALIDATION_SCHEDULES.items():
        metrics = directional_error_metrics(run_backtests(train, horizon=24, origins=origins))
        metrics.insert(0, "origins", ",".join(map(str, origins)))
        metrics.insert(0, "schedule", schedule)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def change_error_summary(
    train: pd.DataFrame, backtest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label actual five-minute changes and summarize all candidate errors.

    Supply exactly 110 chronological training rows, never holdout observations.
    Each origin's threshold is the q90 of positive differences wholly within
    its training prefix. Validation changes use timestamp-matched actuals,
    including the last training actual for the first validation step. Without
    positive history the threshold is null and positive changes are smaller.

    These are noisy, retrospective, noncausal evaluation labels, not evidence
    of a causal effect or a definition of gradual trends. No future actuals are
    used for forecasts, fitting, thresholds, or selection. Pooled summaries
    count overlapping windows separately and have no pooled threshold.
    """
    if len(train) != 110:
        raise ValueError("Supply exactly 110 training rows, excluding all holdout rows")
    validate_targets(train)
    if not np.equal(np.diff(train.timestamp.to_numpy()), 300).all():
        raise ValueError("Training timestamps must have five-minute cadence")
    frame = backtest.copy()
    if not frame.target.isin(TARGETS).all():
        raise ValueError("Unknown backtest target")
    for column in ("origin", "horizon"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Backtest {column} must be finite integers")
        if frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"Backtest {column} must be integers, not booleans")
    if ((frame.origin < 5) | (frame.origin + 24 > 110)
            | (frame.horizon < 1) | (frame.horizon > 24)).any():
        raise ValueError("Each origin needs five training rows and a complete 24-step window")
    positions = (frame.origin + frame.horizon - 1).to_numpy(dtype=int)
    if not np.equal(frame.timestamp.to_numpy(), train.timestamp.to_numpy()[positions]).all():
        raise ValueError("Backtest timestamp must match its training origin/horizon")
    frame["actual_change"] = np.nan
    frame["large_upward_threshold"] = np.nan
    for (target, origin), group in frame.groupby(["target", "origin"], sort=False):
        values = train[target].to_numpy(dtype=float)
        indices = (group.origin + group.horizon - 1).to_numpy(dtype=int)
        if not np.allclose(group.actual, values[indices], rtol=1e-12, atol=1e-12):
            raise ValueError("Backtest actuals must match supplied training targets")
        history_changes = np.diff(values[:int(origin)])
        positive = history_changes[history_changes > 0]
        threshold = np.quantile(positive, .9) if len(positive) else np.nan
        frame.loc[group.index, "actual_change"] = values[indices] - values[indices - 1]
        frame.loc[group.index, "large_upward_threshold"] = threshold
    frame["change_group"] = "nonpositive_change"
    frame.loc[frame.actual_change > 0, "change_group"] = "smaller_upward_change"
    frame.loc[(frame.actual_change > 0) & (
        frame.actual_change > frame.large_upward_threshold
    ), "change_group"] = "large_upward_change"
    error = frame.error if "error" in frame else frame.prediction - frame.actual
    frame["underprediction"] = (-error).clip(lower=0)
    rows = []
    for scope, keys in [("origin", ["target", "model", "origin"]),
                        ("pooled", ["target", "model"])]:
        for key, group in frame.groupby(keys, sort=False):
            for label in ("large_upward_change", "smaller_upward_change", "nonpositive_change"):
                subset = group.loc[group.change_group == label]
                errors = subset.error if "error" in subset else subset.prediction - subset.actual
                under = subset.underprediction
                rows.append({
                    "target": key[0], "model": key[1], "scope": scope,
                    "origin": key[2] if scope == "origin" else np.nan,
                    "change_group": label, "n": len(subset),
                    "MAE": errors.abs().mean(), "mean_underprediction": under.mean(),
                    "underprediction_rate": (under > 0).mean(),
                    "max_underprediction": under.max(),
                })
    return frame, pd.DataFrame(rows, columns=[
        "target", "model", "scope", "origin", "change_group", "n", "MAE",
        "mean_underprediction", "underprediction_rate", "max_underprediction",
    ])
