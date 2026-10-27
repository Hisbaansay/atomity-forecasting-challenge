"""Training-only, equal-VM-weight cross-sectional CPU descriptions."""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd

from .data import CPU_COLUMNS, clean_readings


def cross_sectional_diagnostics(
    cpu: pd.DataFrame, training_timestamps: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological per-time statistics, VM profiles, and class summaries.

    Supply the first 2--110 timestamps, strictly ordered at 300-second cadence.
    Only selected readings are cleaned; future CPU values and VM identities never
    enter validation or statistics. Valid duplicate VM/timestamp rows are averaged
    before aggregation, so each reporting VM has equal weight at each timestamp.

    All thresholds are illustrative, not incident, SLA, or starvation evidence.
    Workload-active means avg_cpu > 5 AND interval max_cpu > 20, not the official
    active_vm_count target (reporting VMs), churn, or provisioned compute capacity.
    Mean classes are <5, 5 through 50 inclusive, and >50. Hot average and hot
    interval maximum flags both use >80, but an interval maximum can be momentary.
    Bursty means sample temporal std >= the training cross-VM std 90th percentile
    AND std > 0. This retrospective descriptive cutoff is not a prediction feature;
    ties can flag more than 10% of VMs. Singleton stds remain NaN and are not bursty.

    Summary shares use all observed training VMs as denominator. Exclusive mean
    classes partition that population; nonexclusive flag rows may overlap. Summary
    means are equal-weight VM-profile means, with undefined stds excluded. CPU and
    std columns are percentages and percentage points respectively. No files or
    figures are written; the VM profile is optional detail for downstream callers.
    """
    grid = list(training_timestamps)
    if (not 2 <= len(grid) <= 110
            or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, Integral)
                   or t < 0 for t in grid)
            or any(b - a != 300 for a, b in zip(grid, grid[1:]))):
        raise ValueError("Training grid requires first 2--110 nonnegative integer timestamps, "
                         "strictly ordered at 300-second cadence")
    missing = set(CPU_COLUMNS) - set(cpu.columns)
    if missing:
        raise ValueError(f"Missing CPU columns: {sorted(missing)}")
    timestamps = pd.to_numeric(cpu.timestamp, errors="coerce")
    if timestamps.lt(grid[0]).any():
        raise ValueError("Training grid must start at the first input timestamp")
    selected = cpu.loc[timestamps.isin(grid), CPU_COLUMNS].copy()
    selected["timestamp"] = pd.to_numeric(selected.timestamp)
    if set(selected.timestamp) != set(grid):
        raise ValueError("Every requested training timestamp needs observations")
    clean, _ = clean_readings(selected)
    clean["workload_active"] = clean.avg_cpu.gt(5) & clean.max_cpu.gt(20)
    grouped = clean.groupby("timestamp", sort=True)
    pertime = grouped.agg(
        reporting_vm_count=("vm_id", "size"),
        workload_active_vm_count=("workload_active", "sum"),
        mean_avg_cpu=("avg_cpu", "mean"),
    )
    for q in (50, 75, 90, 95, 99):
        pertime[f"p{q}_avg_cpu"] = grouped.avg_cpu.quantile(q / 100)
    pertime["max_avg_cpu"] = grouped.avg_cpu.max()
    pertime = pertime.reset_index()

    profile = clean.groupby("vm_id", sort=True).agg(
        avg_cpu_mean=("avg_cpu", "mean"),
        avg_cpu_std=("avg_cpu", "std"),
        peak_avg_cpu=("avg_cpu", "max"),
        peak_interval_max_cpu=("max_cpu", "max"),
        observations=("avg_cpu", "size"),
    ).reset_index()
    profile["mean_load_class"] = np.select(
        [profile.avg_cpu_mean.lt(5), profile.avg_cpu_mean.le(50)],
        ["idle_low", "moderate"], default="high",
    )
    profile["hot_avg_gt80"] = profile.peak_avg_cpu.gt(80)
    profile["interval_max_gt80"] = profile.peak_interval_max_cpu.gt(80)
    threshold = profile.avg_cpu_std.quantile(.9)
    profile["bursty_std_threshold_pp"] = threshold
    profile["bursty"] = profile.avg_cpu_std.ge(threshold) & profile.avg_cpu_std.gt(0)

    rows = []
    for label in ("idle_low", "moderate", "high", "hot_avg_gt80",
                  "interval_max_gt80", "bursty"):
        exclusive = label in ("idle_low", "moderate", "high")
        members = profile.loc[profile.mean_load_class.eq(label) if exclusive else profile[label]]
        rows.append({
            "class": label,
            "classification": "exclusive_mean" if exclusive else "nonexclusive_flags",
            "vm_count": len(members),
            "share_vms": len(members) / len(profile),
            "mean_avg_cpu": members.avg_cpu_mean.mean(),
            "mean_std_pp": members.avg_cpu_std.mean(),
            "peak_avg_cpu": members.peak_avg_cpu.max(),
            "peak_interval_max_cpu": members.peak_interval_max_cpu.max(),
        })
    return pertime, profile, pd.DataFrame(rows)
