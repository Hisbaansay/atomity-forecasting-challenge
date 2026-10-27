"""Expanding-window evaluation and training-only model selection."""

import numpy as np
import pandas as pd

from .models import BASELINES, CANDIDATES, TARGETS, forecast


def validate_targets(targets: pd.DataFrame) -> None:
    """Validate the timestamp grid and finite labels of a training frame."""
    required = {"timestamp", *TARGETS}
    if not required.issubset(targets.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    timestamps = targets["timestamp"].to_numpy(dtype=float)
    if len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError("At least two finite numeric timestamps are required")
    differences = np.diff(timestamps)
    if differences[0] <= 0 or not np.allclose(differences, differences[0], rtol=1e-9, atol=1e-9):
        raise ValueError("Timestamps must be strictly increasing with a regular cadence")
    if not np.isfinite(targets[list(TARGETS)].to_numpy(dtype=float)).all():
        raise ValueError("Training targets must be finite")


def backtest_origins(train_size: int, horizon: int = 24) -> tuple[int, ...]:
    """Origins are prefix lengths; every validation window lies in training."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if train_size < 5 + horizon:
        raise ValueError("Training requires at least five fitting rows plus one horizon")
    if train_size == 110 and horizon == 24:
        return (50, 70, 86)
    last = train_size - horizon
    first = max(5, min(train_size // 2, last))
    return tuple(sorted(set(np.linspace(first, last, 3, dtype=int).tolist())))


def run_backtests(train: pd.DataFrame, horizon: int = 24, origins=None) -> pd.DataFrame:
    """Return long-form fixed-origin errors (prediction minus actual)."""
    validate_targets(train)
    default_origins = backtest_origins(len(train), horizon)
    origins = default_origins if origins is None else tuple(origins)
    if not origins or any(
        not isinstance(origin, (int, np.integer)) or origin < 5 or origin + horizon > len(train)
        for origin in origins
    ):
        raise ValueError("Each origin needs five training rows and a complete validation horizon")
    rows = []
    for target in TARGETS:
        values = train[target].to_numpy(dtype=float)
        for origin in origins:
            history = values[:origin]
            actual = values[origin:origin + horizon]
            threshold = float(np.quantile(history, 0.9))
            for model in CANDIDATES:
                prediction = forecast(history, model, horizon, target)
                for step in range(horizon):
                    rows.append({
                        "target": target, "model": model, "origin": int(origin),
                        "horizon": step + 1,
                        "timestamp": train["timestamp"].iloc[origin + step],
                        "actual": actual[step], "prediction": prediction[step],
                        "error": prediction[step] - actual[step],
                        "peak": bool(actual[step] >= threshold),
                        "peak_threshold": threshold,
                    })
    return pd.DataFrame(rows)


def summarize_backtests(backtest: pd.DataFrame) -> pd.DataFrame:
    """Pool all origin/horizon errors, including overlapping validation windows."""
    rows = []
    for target in TARGETS:
        for model in CANDIDATES:
            group = backtest[(backtest.target == target) & (backtest.model == model)]
            if group.empty:
                raise ValueError(f"Missing backtest rows for {target}/{model}")
            error = group.error.to_numpy(dtype=float)
            peaks = group.loc[group.peak, "error"].to_numpy(dtype=float)
            rows.append({
                "target": target, "model": model,
                "MAE": float(np.abs(error).mean()),
                "RMSE": float(np.sqrt(np.square(error).mean())),
                "bias": float(error.mean()),
                "peak_MAE": float(np.abs(peaks).mean()) if len(peaks) else None,
            })
    summary = pd.DataFrame(rows)
    summary["relative_improvement"] = pd.Series([None] * len(summary), dtype=object)
    for target in TARGETS:
        mask = summary.target == target
        baseline = summary.loc[mask & summary.model.isin(BASELINES), "MAE"].min()
        if baseline != 0:
            summary.loc[mask, "relative_improvement"] = (
                (baseline - summary.loc[mask, "MAE"]) / baseline
            )
    return summary


def select_models(train: pd.DataFrame, horizon: int = 24) -> tuple[dict, pd.DataFrame]:
    """Select each target independently using only the supplied training prefix."""
    backtest = run_backtests(train, horizon)
    summary = summarize_backtests(backtest)
    selection = {
        "train_size": len(train), "horizon": int(horizon),
        "origins": list(backtest_origins(len(train), horizon)),
        "criterion": "pooled_backtest_MAE", "targets": {},
    }
    for target in TARGETS:
        scores = summary[summary.target == target]
        best = scores.loc[scores.MAE.idxmin()]
        baseline = scores[scores.model.isin(BASELINES)]
        selection["targets"][target] = {
            "model": best.model,
            "best_baseline": baseline.loc[baseline.MAE.idxmin(), "model"],
            "metrics": {key: (None if pd.isna(best[key]) else float(best[key]))
                        for key in ("MAE", "RMSE", "bias", "peak_MAE", "relative_improvement")},
            "candidates": [
                {key: (None if pd.isna(value) else value) for key, value in row.items()}
                for row in scores.drop(columns="target").to_dict(orient="records")
            ],
        }
    return selection, backtest
