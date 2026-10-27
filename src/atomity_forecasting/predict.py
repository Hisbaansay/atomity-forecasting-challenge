"""Training-only model selection and holdout prediction CLI."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtesting import select_models, validate_targets
from .models import TARGETS, forecast


def forecast_targets(targets: pd.DataFrame, train_size: int = 110, horizon: int = 24):
    """Return predictions, selection metadata, and training-only backtests.

    Only timestamps are inspected after train_size. Holdout labels may be NaN
    or arbitrary objects without affecting selection, fitting, or forecasts.
    """
    if not isinstance(train_size, (int, np.integer)) or train_size < 5:
        raise ValueError("train_size must be an integer of at least five")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if len(targets) < train_size + horizon:
        raise ValueError("Need at least train_size + horizon timestamps")
    train = targets.iloc[:train_size].copy()
    validate_targets(train)
    timestamps = targets["timestamp"].iloc[:train_size + horizon].to_numpy(dtype=float)
    cadence = float(train["timestamp"].iloc[1] - train["timestamp"].iloc[0])
    if not np.isfinite(timestamps).all() or not np.allclose(
        np.diff(timestamps), cadence, rtol=1e-9, atol=1e-9
    ):
        raise ValueError("Forecast timestamps must continue the regular training cadence")
    selection, backtest = select_models(train, horizon)
    predictions = pd.DataFrame({
        "timestamp": targets["timestamp"].iloc[train_size:train_size + horizon].to_numpy()
    })
    for target in TARGETS:
        predictions[target] = forecast(
            train[target].to_numpy(dtype=float), selection["targets"][target]["model"], horizon, target
        )
    predictions["active_vm_count"] = predictions["active_vm_count"].astype(np.int64)
    return predictions, selection, backtest


def main(argv=None) -> None:
    from .data import construct_targets, load_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-size", type=int, default=110)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args(argv)
    cpu, _, _ = load_data(args.data)
    predictions, selection, _ = forecast_targets(
        construct_targets(cpu), args.train_size, args.horizon
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    selection_path = args.selection_output or args.output.with_suffix(".selection.json")
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
