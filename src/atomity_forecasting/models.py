"""Small, deterministic univariate forecasting candidates."""

import numpy as np

TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
CANDIDATES = (
    "persistence", "mean_6", "mean_12", "mean_24",
    "ses_0.2", "ses_0.5", "damped_trend",
)
BASELINES = CANDIDATES[:4]


def forecast(values, model: str, horizon: int = 24, target: str = "aggregate_cpu_utilization") -> np.ndarray:
    """Fit on the supplied history and forecast all steps without updating."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Training values must be a nonempty finite one-dimensional array")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    if model not in CANDIDATES:
        raise ValueError(f"Unknown model: {model}")
    if model == "persistence":
        result = np.full(horizon, y[-1])
    elif model.startswith("mean_"):
        result = np.full(horizon, y[-int(model.split("_")[1]):].mean())
    elif model.startswith("ses_"):
        alpha = float(model.split("_")[1])
        level = y[0]
        for value in y[1:]:
            level = alpha * value + (1 - alpha) * level
        result = np.full(horizon, level)
    else:
        # Fixed Holt recurrence: alpha=.3, beta=.1, damping=.9.
        level, trend = y[0], 0.0
        for value in y[1:]:
            previous = level
            level = 0.3 * value + 0.7 * (level + 0.9 * trend)
            trend = 0.1 * (level - previous) + 0.9 * 0.9 * trend
        result = level + trend * np.cumsum(0.9 ** np.arange(1, horizon + 1))
    if target == "active_vm_count":
        return np.rint(np.maximum(result, 0))
    return np.clip(result, 0, 100)
