"""Canonical charts from computed analysis data; only save_figure exports files."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .models import TARGETS


def save_figure(fig: Figure, path: Path) -> None:
    """Reserve a separate legend band so labels never cover plotted data."""
    entries = {}
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            entries.setdefault(label, handle)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    top = 1.0
    if entries:
        columns = min(2, len(entries))
        rows = int(np.ceil(len(entries) / columns))
        legend_height = .24 * rows + .16
        top -= legend_height / fig.get_figheight()
        fig.legend(list(entries.values()), list(entries), loc="upper center",
                   bbox_to_anchor=(.5, .995), ncol=columns, frameon=False,
                   fontsize=9, borderaxespad=0)
    fig.tight_layout(rect=(.01, .01, .99, top), pad=1.2, h_pad=1.8, w_pad=1.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_cpu_coverage(per_time: pd.DataFrame, cross_section: pd.DataFrame) -> Figure:
    minutes = (per_time.timestamp - per_time.timestamp.iloc[0]) / 60
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(minutes, per_time.aggregate_cpu_utilization, label="Equal-VM aggregate mean")
    axes[0].set(ylabel="CPU (%)", title="Training aggregate CPU, first 110 timestamps")
    axes[1].plot(minutes, per_time.active_vm_count, label="Reporting VMs (required target)")
    axes[1].plot(minutes, cross_section.workload_active_vm_count, label="Workload-active (illustrative rule)")
    axes[1].set(ylabel="VM count", xlabel="Minutes since training start", title="Reporting coverage differs from CPU activity")
    return fig


def plot_cross_sectional_fan(cross_section: pd.DataFrame) -> Figure:
    minutes = (cross_section.timestamp - cross_section.timestamp.iloc[0]) / 60
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bands = [("p50_avg_cpu", "p75_avg_cpu", "#d9e8ed", "P50-P75"),
             ("p75_avg_cpu", "p90_avg_cpu", "#b5d2dc", "P75-P90"),
             ("p90_avg_cpu", "p95_avg_cpu", "#80afc2", "P90-P95"),
             ("p95_avg_cpu", "p99_avg_cpu", "#4986a0", "P95-P99")]
    for lower, upper, color, label in bands:
        ax.fill_between(minutes, cross_section[lower], cross_section[upper], color=color, label=label)
    ax.plot(minutes, cross_section.mean_avg_cpu, color="#c35f26", label="Mean VM avg CPU", linewidth=1.8)
    ax.plot(minutes, cross_section.max_avg_cpu, color="#31434e", linestyle=":", label="Max VM avg CPU")
    ax.set(xlabel="Minutes since training start", ylabel="VM average CPU (%)", ylim=(0, 105),
           title="Training cross-sectional quantiles: population spread, not forecast uncertainty")
    return fig


def plot_horizon_headroom(capacity_schedule: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    for percentile, color in (("p95", "#bdd2de"), ("p90", "#76a2ba")):
        ax.fill_between(capacity_schedule.minutes_ahead, capacity_schedule.point_forecast,
                        capacity_schedule[f"reference_{percentile}_pct"], color=color,
                        label=f"Empirical {percentile.upper()} envelope (not calibrated)")
    ax.plot(capacity_schedule.minutes_ahead, capacity_schedule.point_forecast, color="#24556b", label="Frozen point forecast")
    ax.set(xlabel="Forecast horizon (minutes)", ylabel="Equal-VM mean CPU reference (%)",
           title="Illustrative horizon headroom: only three residuals per horizon")
    return fig


def plot_validation_schedules(sensitivity: pd.DataFrame, comparison: pd.DataFrame) -> Figure:
    matrix = sensitivity[sensitivity.target == "aggregate_cpu_utilization"].pivot(
        index="model", columns="schedule", values="MAE").reindex(
            index=comparison[comparison.target == "aggregate_cpu_utilization"].model,
            columns=["original", "alternative_a", "alternative_b"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(3), ["A: 50/70/86", "B: 38/62/86", "C: 30/54/78"])
    ax.set_yticks(range(len(matrix)), matrix.index)
    for row in range(len(matrix)):
        for column in range(3):
            ax.text(column, row, f"{matrix.iloc[row, column]:.4f}", ha="center", va="center",
                    color="white" if matrix.iloc[row, column] > .25 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="CPU MAE (percentage points)")
    ax.set_title("Validation schedule sensitivity: different periods, not comparable test sets")
    return fig


def plot_vm_utilization(vm_means: pd.Series, bins: np.ndarray, time_summary: pd.DataFrame) -> Figure:
    minutes = (time_summary.timestamp - time_summary.timestamp.iloc[0]) / 60
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(vm_means, bins=bins, label="Training mean per VM")
    axes[0].set(xlabel="Mean CPU (%)", ylabel="VM count", title="Distribution of VM training means")
    axes[1].plot(minutes, time_summary.ge80 * 100, label="avg CPU >=80%")
    axes[1].plot(minutes, time_summary.ge95 * 100, label="avg CPU >=95%")
    axes[1].set(xlabel="Minutes since training start", ylabel="Readings (%)", title="High-utilization training readings")
    return fig


def plot_segments(segment: pd.DataFrame, dimensions: list[str]) -> Figure:
    fig, axes = plt.subplots(len(dimensions), 2, figsize=(13, 4 * len(dimensions)), squeeze=False)
    for row, column in enumerate(dimensions):
        group = segment[segment.dimension == column]
        for ax, metric, unit in ((axes[row, 0], "mean_cpu_pct", "Mean CPU (%)"),
                                 (axes[row, 1], "vm_count", "VM count")):
            ax.bar(group.segment, group[metric], label="Training population")
            ax.set(ylabel=unit, title=f"{column}: {unit}")
            ax.tick_params(axis="x", rotation=0)
    return fig


def plot_autocorrelation(acf: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(acf.lag_minutes, acf.autocorrelation, width=3, label="Descriptive Pearson autocorrelation")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set(xlabel="Lag (minutes)", ylabel="Correlation (unitless)", title="Training aggregate CPU autocorrelation; no significance inference")
    return fig


def plot_backtest_cpu(bt: pd.DataFrame, selection: dict[str, Any], start_timestamp: int) -> Figure:
    cpu_bt = bt[bt.target == "aggregate_cpu_utilization"]
    fig, axes = plt.subplots(len(selection["origins"]), 1, figsize=(10, 6.5), squeeze=False)
    cpu_choice = selection["targets"]["aggregate_cpu_utilization"]
    for ax, origin in zip(axes[:, 0], selection["origins"]):
        group = cpu_bt[cpu_bt.origin == origin]
        actual = group[group.model == cpu_choice["model"]]
        x = (actual.timestamp - start_timestamp) / 60
        ax.plot(x, actual.actual, label="Actual validation CPU")
        shared = cpu_choice["model"] == cpu_choice["best_baseline"]
        label = "Selected = baseline: persistence" if shared else f"Selected: {cpu_choice['model']}"
        ax.plot(x, actual.prediction, label=label, linestyle="--")
        if not shared:
            baseline = group[group.model == cpu_choice["best_baseline"]]
            ax.plot(x, baseline.prediction, linestyle=":", label=f"Baseline: {cpu_choice['best_baseline']}")
        ax.set(xlabel="Minutes since training start", ylabel="Aggregate CPU (%)",
               title=f"Training-only backtest, origin prefix {origin}")
    return fig


def plot_model_comparison(comparison: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    group = comparison[comparison.target == "aggregate_cpu_utilization"]
    ax.barh(group.model, group.MAE, color=["#24556b" if selected else "#9caeb8" for selected in group.selected])
    ax.invert_yaxis()
    ax.set_ylim(len(group) + 1.3, -.7)
    ax.set(xlabel="CPU MAE (percentage points)", title="CPU candidate comparison: predefined origins 50, 70, 86")
    ax.text(.98, .04, "Active VM count: all seven candidates have MAE = RMSE = 0 VMs.\nNo evidence of learned churn.",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=9,
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#d0d0d0"})
    return fig


def plot_horizon_errors(horizon: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, target in zip(axes, TARGETS):
        group = horizon[horizon.target == target]
        ax.plot(group.horizon * 5, group.MAE, label="Selected model MAE across origins")
        ax.set(xlabel="Forecast horizon (minutes)", ylabel="MAE (VMs)" if target == "active_vm_count" else "MAE (percentage points)", title=target)
    return fig


def plot_residual_bias_peak_errors(origin_metrics: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metrics = origin_metrics[origin_metrics.target == "aggregate_cpu_utilization"]
    positions = np.arange(len(metrics))
    axes[0].bar(positions, metrics.bias, label="Mean signed error")
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set(xlabel="Origin prefix length", ylabel="Bias (percentage points)", title="Selected CPU residual bias")
    axes[1].bar(positions, metrics.peak_MAE, label="Peak MAE (past-only q90 rule)")
    for position, row in zip(positions, metrics.itertuples()):
        if row.peak_n == 0:
            axes[1].text(position, 0.02, "N/A\n0 cases", ha="center", va="bottom")
    axes[1].set(xlabel="Origin prefix length", ylabel="MAE (percentage points)", title="Selected CPU peak error (N/A if no cases)")
    for ax in axes:
        ax.set_xticks(positions, metrics.origin.astype(str))
        ax.set_xlim(-.6, len(metrics) - .4)
    return fig


def plot_data_quality(issues: pd.Series, coverage: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(issues.index, issues, label="Full structural audit")
    axes[0].set(xlabel="Rows or VMs (as named)", title="Data-quality violations; zero bars mean none")
    axes[1].plot((coverage.timestamp - coverage.timestamp.iloc[0]) / 60, coverage.vm_count,
                 label="Structural coverage only, no CPU labels")
    axes[1].set(xlabel="Minutes since first timestamp", ylabel="VM count", title="Full-window structural coverage")
    return fig


PLOTTERS: dict[str, Callable[..., Figure]] = {
    "training_cpu_coverage": plot_training_cpu_coverage,
    "cross_sectional_fan": plot_cross_sectional_fan,
    "horizon_headroom": plot_horizon_headroom,
    "validation_schedules": plot_validation_schedules,
    "vm_utilization": plot_vm_utilization,
    "segments": plot_segments,
    "autocorrelation": plot_autocorrelation,
    "backtest_cpu": plot_backtest_cpu,
    "model_comparison": plot_model_comparison,
    "horizon_errors": plot_horizon_errors,
    "residual_bias_peak_errors": plot_residual_bias_peak_errors,
    "data_quality": plot_data_quality,
}
