from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atomity_forecasting.backtesting import run_backtests
from atomity_forecasting.evaluation import schedule_diagnostics


def train_frame():
    t = np.arange(110)
    return pd.DataFrame({"timestamp": t * 300, "active_vm_count": 3,
                         "aggregate_cpu_utilization": 20 + np.sin(t / 7)})


def test_all_schedules_include_origin_and_pooled_metrics_without_input_changes():
    train = train_frame()
    before = train.copy(deep=True)
    results = schedule_diagnostics(train)
    pd.testing.assert_frame_equal(train, before)
    pd.testing.assert_frame_equal(results, schedule_diagnostics(train))
    assert len(results) == 168
    for schedule, origins in {"original": (50, 70, 86), "alternative_a": (38, 62, 86),
                              "alternative_b": (30, 54, 78)}.items():
        actual = results[(results.schedule == schedule) & (results.target == "aggregate_cpu_utilization")]
        raw = run_backtests(train, origins=origins)
        for row in actual.itertuples():
            group = raw[(raw.target == row.target) & (raw.model == row.model)]
            if row.scope == "origin":
                group = group[group.origin == row.origin]
            error = group.prediction - group.actual
            assert row.MAE == pytest.approx(error.abs().mean())
            assert row.RMSE == pytest.approx(np.sqrt((error ** 2).mean()))
            assert row.bias == pytest.approx(error.mean())
            assert row.max_absolute_error == pytest.approx(error.abs().max())


def test_schedule_diagnostics_rejects_holdout_rows():
    with pytest.raises(ValueError, match="110 training"):
        schedule_diagnostics(train_frame().reindex(range(134)))


def test_official_original_schedule_metrics_when_training_artifact_available():
    path = Path(__file__).resolve().parents[1] / "reports/results/targets_train.csv"
    if not path.exists():
        pytest.skip("Official training artifact not available")
    results = schedule_diagnostics(pd.read_csv(path))
    row = results[(results.schedule == "original") & (results.scope == "pooled") &
                  (results.target == "aggregate_cpu_utilization") & (results.model == "persistence")].iloc[0]
    assert row.MAE == pytest.approx(.23531, abs=5e-6)
    assert row.RMSE == pytest.approx(.27463, abs=5e-6)
