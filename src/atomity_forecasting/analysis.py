"""Reproducible training-only analysis; final 24 labels are never scored or plotted."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from .backtesting import summarize_backtests
from .data import construct_targets, load_data
from .capacity import horizon_headroom
from .cross_section import cross_sectional_diagnostics
from .evaluation import (
    change_error_summary, detailed_origin_metrics, diagnostic_horizon_metrics,
    directional_error_metrics, schedule_diagnostics, validation_sensitivity,
)
from .models import TARGETS
from .predict import forecast_targets
from .plotting import PLOTTERS, save_figure
from .typography import times_new_roman_files


def json_safe(value):
    """Normalize numpy/pandas scalars and undefined descriptive statistics."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def distribution(values):
    values = pd.Series(values, dtype=float).dropna()
    return {"n": len(values), "min": values.min(), "max": values.max(),
            "mean": values.mean(), "median": values.median(),
            "std_ddof1": values.std(),
             **{f"p{q}": values.quantile(q / 100) for q in (10, 25, 75, 90, 95, 99)}}


def workload_diagnostics(cpu, metadata, training_timestamps):
    """Describe hot-VM concentration and temporal variability on an explicit prefix.

    No metadata utilization summaries enter these calculations. The top group is
    retrospectively ranked by high-reading count, not a predictive feature.
    """
    selected = cpu[cpu.timestamp.isin(training_timestamps)].copy()
    if selected.empty or set(selected.timestamp) != set(training_timestamps):
        raise ValueError("Every requested training timestamp needs observations")
    selected["high80"] = selected.avg_cpu.ge(80)
    vm = selected.groupby("vm_id").agg(
        observations=("avg_cpu", "size"), high80=("high80", "sum"),
        mean_cpu=("avg_cpu", "mean"), temporal_std=("avg_cpu", "std"),
    )
    top_count = max(1, int(np.ceil(.01 * len(vm))))
    high_total = int(vm.high80.sum())
    concentration = {
        "scope": "CPU readings at explicitly supplied training timestamps only",
        "vm_count": len(vm), "training_timestamp_count": len(set(training_timestamps)),
        "high_threshold_pct": 80, "high_readings": high_total,
        "vms_ever_high": int(vm.high80.gt(0).sum()),
        "vms_high_at_least_half": int(vm.high80.div(vm.observations).ge(.5).sum()),
        "vms_training_mean_ge80": int(vm.mean_cpu.ge(80).sum()),
        "top_group_fraction": .01, "top_group_vm_count": top_count,
        "top_group_high_readings": int(vm.high80.nlargest(top_count).sum()),
        "top_group_share_of_high_readings": (
            float(vm.high80.nlargest(top_count).sum() / high_total) if high_total else None
        ),
        "mean_vm_temporal_std_pp": float(vm.temporal_std.mean()),
        "top_group_note": "Retrospective count ranking; descriptive concentration, not incident prediction",
    }
    joined = selected.merge(metadata, on="vm_id", validate="many_to_one", suffixes=("", "_metadata"))
    rows = []
    for dimension in ("vm_category", "virtual_core_bucket", "memory_gb_bucket"):
        if dimension not in joined:
            continue
        for label, group in joined.groupby(dimension, dropna=False, observed=True):
            means = group.groupby("timestamp").avg_cpu.mean().sort_index()
            ids = group.vm_id.unique()
            rows.append({
                "dimension": dimension, "segment": str(label), "vm_count": len(ids),
                "timestamps": len(means), "mean_cpu_pct": float(means.mean()),
                "segment_mean_temporal_std_pp": float(means.std()),
                "largest_absolute_change_pp": float(means.diff().abs().max()),
                "mean_vm_temporal_std_pp": float(vm.loc[ids, "temporal_std"].mean()),
            })
    return concentration, pd.DataFrame(rows, columns=[
        "dimension", "segment", "vm_count", "timestamps", "mean_cpu_pct",
        "segment_mean_temporal_std_pp", "largest_absolute_change_pp", "mean_vm_temporal_std_pp",
    ])


def main(argv: list[str] | None = None, *, render_figures: bool = True) -> dict:
    fonts = times_new_roman_files()
    if fonts:
        for font in fonts.values():
            font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Times New Roman" if fonts else "STIXGeneral",
                         "mathtext.fontset": "stix"})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cpu, metadata, audit = load_data(args.data)
    full = construct_targets(cpu)
    if len(full) != 134:
        raise ValueError("This analysis requires exactly 110 training and 24 holdout timestamps")

    # Freeze selection before inspecting any distributions or validation errors.
    predictions, selection, bt = forecast_targets(full, 110, 24)
    capacity_schedule = horizon_headroom(bt, predictions)
    frozen_selection = json.dumps(json_safe(selection), sort_keys=True, allow_nan=False)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    figures = output / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    artifacts = {"tables": {}, "summaries": {}, "figures": {}, "plot_inputs": {}}

    def write_json(name, value):
        artifacts["summaries"][name] = json_safe(value)
        (results / f"{name}.json").write_text(
            json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n"
        )

    def csv(name, frame):
        artifacts["tables"][name] = frame.copy()
        frame.to_csv(results / f"{name}.csv", index=False)

    def figure(name, *inputs):
        artifacts["plot_inputs"][name] = inputs
        if render_figures:
            fig = PLOTTERS[name](*inputs)
            save_figure(fig, figures / f"{name}.png")
            artifacts["figures"][name] = fig

    write_json("selection", selection)
    csv("capacity_schedule", capacity_schedule)
    checks = []
    for perturbation in ("extreme_finite", "all_nan"):
        changed = full.copy()
        for i, target in enumerate(TARGETS):
            changed[target] = changed[target].astype(float)
            changed.loc[changed.index[110:], target] = (
                np.nan if perturbation == "all_nan" else
                np.linspace(-1e9, 1e9, 24) * (i + 1)
            )
        other_predictions, other_selection, other_bt = forecast_targets(changed, 110, 24)
        pd.testing.assert_frame_equal(predictions, other_predictions, check_exact=True)
        pd.testing.assert_frame_equal(bt, other_bt, check_exact=True)
        assert json.dumps(json_safe(other_selection), sort_keys=True, allow_nan=False) == frozen_selection
        checks.append({"perturbation": perturbation, "changed_rows": 24,
                       "targets": list(TARGETS), "predictions_exactly_equal": True,
                       "selection_exactly_equal": True, "backtests_exactly_equal": True})
    write_json("holdout_isolation_check", {"passed": True, "checks": checks,
               "holdout_scored": False, "holdout_labels_exported": False})

    train = full.iloc[:110].copy()
    del full, predictions
    train_cpu = cpu[cpu.timestamp.isin(train.timestamp)].copy()
    cross_section, vm_profiles, vm_classes = cross_sectional_diagnostics(cpu, train.timestamp.to_numpy())
    csv("cross_sectional_quantiles", cross_section)
    csv("vm_classification_summary", vm_classes)
    csv("targets_train", train)
    audit["scope"] = "Full-window structure only; CPU-value EDA uses first 110 timestamps"
    audit["metadata_schema"] = {c: str(t) for c, t in metadata.dtypes.items()}
    observations = cpu.groupby("vm_id").timestamp.agg(["size", "min", "max"])
    audit["observations_per_vm"] = distribution(observations["size"])
    audit["population_matches"] = set(cpu.vm_id) == set(metadata.vm_id)
    cadence = cpu.sort_values(["vm_id", "timestamp"]).groupby("vm_id").timestamp.diff().dropna()
    audit["within_vm_cadence_seconds"] = distribution(cadence)
    csv("observations_per_vm", observations["size"].value_counts().sort_index()
        .rename_axis("observations").reset_index(name="vm_count"))
    coverage_rows = []
    previous = None
    for timestamp, group in cpu.groupby("timestamp", sort=True):
        current = set(group.vm_id)
        coverage_rows.append({"timestamp": timestamp, "vm_count": len(current),
                              "entered_since_previous": None if previous is None else len(current - previous),
                              "left_since_previous": None if previous is None else len(previous - current),
                              "scope": "structural_only"})
        previous = current
    coverage = pd.DataFrame(coverage_rows)
    csv("coverage", coverage)
    audit["coverage"] = {"min_vm_count": coverage.vm_count.min(),
                         "max_vm_count": coverage.vm_count.max(),
                         "total_entries_after_first": coverage.entered_since_previous.sum(),
                         "total_exits_after_first": coverage.left_since_previous.sum()}
    missingness = [{"table": table, "column": column, "missing_rows": count,
                    "total_rows": audit["raw_rows"] if table == "cpu_raw" else len(metadata)}
                   for table, counts in (("cpu_raw", audit["missing_by_column"]),
                                         ("metadata", audit["metadata_missing_by_column"]))
                   for column, count in counts.items()]
    csv("missingness", pd.DataFrame(missingness))
    quality_keys = ["raw_rows", "clean_rows", "duplicate_key_rows", "invalid_cpu_or_id_rows",
                    "invalid_cpu_range_rows", "invalid_min_mean_max_rows",
                    "vms_missing_metadata", "metadata_without_readings"]
    csv("data_quality", pd.DataFrame({"metric": quality_keys, "count": [audit[k] for k in quality_keys]}))

    # Source metadata statistics are audited separately, never model features.
    meta_stats = []
    for column in metadata.columns:
        if "cpu" in column.lower():
            values = pd.to_numeric(metadata[column], errors="coerce")
            meta_stats.append({"column": column, **distribution(values),
                               "fraction_gt100": values.gt(100).mean(),
                               "fraction_negative": values.lt(0).mean(),
                               "non_numeric_or_missing": values.isna().sum()})
    csv("metadata_cpu_statistics", pd.DataFrame(meta_stats))
    audit["metadata_cpu_statistics"] = meta_stats
    audit["metadata_cpu_warning"] = "Source metadata CPU units/window may differ; not predictive inputs or training CPU summaries."
    creation = next((c for c in metadata if "created" in c.lower()), None)
    deletion = next((c for c in metadata if "deleted" in c.lower()), None)
    lifetime = {"creation_column": creation, "deletion_column": deletion,
                 "boundary_convention": "Inclusive endpoint comparison is an audit assumption, not a source guarantee",
                 "span_note": "Deletion minus creation is a recorded metadata span; clipping/censoring semantics are unverified"}
    if creation and deletion:
        bounds = observations.join(metadata.set_index("vm_id")[[creation, deletion]])
        created = pd.to_numeric(bounds[creation], errors="coerce")
        deleted = pd.to_numeric(bounds[deletion], errors="coerce")
        timed = cpu[["vm_id", "timestamp"]].merge(
            metadata[["vm_id", creation, deletion]], on="vm_id", validate="many_to_one")
        before = timed.timestamp < pd.to_numeric(timed[creation], errors="coerce")
        after = timed.timestamp > pd.to_numeric(timed[deletion], errors="coerce")
        lifetime.update(creation_seconds=distribution(created), deletion_seconds=distribution(deleted),
                        lifetime_seconds=distribution(deleted - created),
                        negative_lifetime_vms=(deleted < created).sum(),
                        missing_bounds_vms=(created.isna() | deleted.isna()).sum(),
                        observations_before_creation=before.sum(), observations_after_deletion=after.sum(),
                        vms_with_observations_outside_lifetime=timed.loc[before | after, "vm_id"].nunique())
        del timed
    audit["lifetimes"] = lifetime
    write_json("audit", audit)

    values = train_cpu.avg_cpu
    vm = train_cpu.groupby("vm_id").agg(cpu_mean=("avg_cpu", "mean"),
               share_ge80=("avg_cpu", lambda x: x.ge(80).mean()),
               share_ge95=("avg_cpu", lambda x: x.ge(95).mean()))
    cpu_summary = {**distribution(values), "share_avg_ge80": values.ge(80).mean(),
                   "share_avg_ge95": values.ge(95).mean(),
                   "vm_mean_distribution": distribution(vm.cpu_mean)}
    bins = np.linspace(0, 100, 21)
    hist, edges = np.histogram(vm.cpu_mean, bins=bins)
    csv("vm_mean_distribution", pd.DataFrame({"lower_cpu_pct": edges[:-1],
        "upper_cpu_pct": edges[1:], "vm_count": hist}))
    per_time = train_cpu.groupby("timestamp").avg_cpu.quantile([.1, .5, .9]).unstack()
    per_time.columns = ["p10", "p50", "p90"]
    per_time = train.merge(per_time.reset_index(), on="timestamp")
    high = train_cpu.assign(ge80=values.ge(80), ge95=values.ge(95)).groupby("timestamp")[
        ["ge80", "ge95"]].mean().reset_index()
    per_time = per_time.merge(high, on="timestamp")
    csv("training_time_summary", per_time)
    joined = vm.join(metadata.set_index("vm_id"), rsuffix="_metadata")
    segments = []
    segment_columns = [c for c in metadata if "category" in c.lower() or "bucket" in c.lower()]
    for column in segment_columns:
        for label, group in joined.groupby(column, dropna=False, observed=True):
            segments.append({"dimension": column, "segment": str(label), "vm_count": len(group),
                             "mean_cpu_pct": group.cpu_mean.mean(),
                             "share_readings_ge80": group.share_ge80.mean(),
                             "share_readings_ge95": group.share_ge95.mean(),
                             "share_vm_mean_ge80": group.cpu_mean.ge(80).mean()})
    segment = pd.DataFrame(segments)
    csv("segment_summary", segment)
    concentration, segment_variability = workload_diagnostics(train_cpu, metadata, train.timestamp)
    write_json("workload_concentration", concentration)
    csv("segment_variability", segment_variability)
    aggregate = train.aggregate_cpu_utilization
    delta = aggregate.diff().dropna()
    acf = pd.DataFrame({"lag_steps": range(1, 13), "lag_minutes": np.arange(1, 13) * 5,
                        "autocorrelation": [aggregate.autocorr(lag) for lag in range(1, 13)]})
    csv("autocorrelation", acf)

    comparison = summarize_backtests(bt)
    training_means = train[list(TARGETS)].mean()
    comparison["training_mean"] = comparison.target.map(training_means)
    comparison["normalized_MAE"] = comparison.MAE / comparison.training_mean.replace(0, np.nan)
    comparison["normalized_MAE_pct"] = comparison.normalized_MAE * 100
    comparison["selected"] = [row.model == selection["targets"][row.target]["model"]
                               for row in comparison.itertuples()]
    csv("model_comparison", comparison)
    csv("backtest_errors", bt)
    sensitivity = validation_sensitivity(train)
    csv("validation_sensitivity", sensitivity)
    csv("schedule_diagnostics", schedule_diagnostics(train))
    csv("origin_horizon_errors", diagnostic_horizon_metrics(bt, selection))
    directional = directional_error_metrics(bt)
    csv("directional_errors", directional)
    change_steps, change_summary = change_error_summary(train, bt)
    csv("change_errors", change_steps)
    csv("change_error_summary", change_summary)
    chosen = bt[[row.model == selection["targets"][row.target]["model"]
                 for row in bt.itertuples()]].copy()
    chosen["absolute_error"] = chosen.error.abs()
    chosen["underprediction"] = (-chosen.error).clip(lower=0)
    horizon = chosen.groupby(["target", "model", "horizon"], as_index=False).agg(
        MAE=("absolute_error", "mean"), bias=("error", "mean"), n=("error", "size"))
    csv("horizon_errors", horizon)
    origin_metrics = detailed_origin_metrics(bt, selection)
    csv("origin_metrics", origin_metrics)
    selected_cpu = chosen[chosen.target == "aggregate_cpu_utilization"]
    residual = {"error_convention": "prediction minus actual; CPU errors are percentage points",
                "pooled_validation_rows": len(selected_cpu),
                "unique_validation_timestamps": selected_cpu.timestamp.nunique(),
                "overlap_note": "Origins overlap; pooled empirical errors are not independent or calibrated prediction intervals.",
                "underprediction_definition": "max(actual - prediction, 0), including zero for overpredictions",
                "underprediction_q90_pp": selected_cpu.underprediction.quantile(.9),
                "underprediction_q95_pp": selected_cpu.underprediction.quantile(.95),
                "underprediction_max_pp": selected_cpu.underprediction.max(),
                "first6_MAE_pp": selected_cpu.loc[selected_cpu.horizon.le(6), "absolute_error"].mean(),
                "last6_MAE_pp": selected_cpu.loc[selected_cpu.horizon.gt(18), "absolute_error"].mean(),
                "peak_rule": "actual >= q90 of aggregate CPU in that origin's fitting prefix",
                "per_origin": origin_metrics.to_dict(orient="records")}
    summary = {"scope": {"eda_timestamps": 110, "holdout_timestamps": 24,
                          "holdout_scored": False, "selection_frozen_before_analysis": True,
                          "training_start": train.timestamp.min(), "training_end": train.timestamp.max()},
               "units": {"cpu": "percent", "cpu_error": "percentage points",
                         "time": "trace-relative seconds", "count": "VMs"},
               "metadata_roles": {"segments": segment_columns, "core_memory": "categorical labels only, not numerical capacities",
                                  "creation": creation, "deletion": deletion,
                                  "source_cpu_columns": [r["column"] for r in meta_stats]},
               "training_cpu": cpu_summary, "training_aggregate": distribution(aggregate),
               "peak_time": train.loc[aggregate.idxmax()].to_dict(),
               "low_time": train.loc[aggregate.idxmin()].to_dict(),
               "volatility": {"aggregate_std_pp": aggregate.std(), "difference_std_pp": delta.std(),
                              "max_absolute_delta_pp": delta.abs().max(), "lag1_autocorrelation": aggregate.autocorr(1)},
               "autocorrelation_note": "Descriptive Pearson correlations at lags 1..12; no significance or seasonality claim.",
                "segments": segments, "selected_validation": residual,
                "workload_concentration": concentration,
                "cross_section": {
                    "reporting_count_min": int(cross_section.reporting_vm_count.min()),
                    "reporting_count_max": int(cross_section.reporting_vm_count.max()),
                    "workload_active_count_min": int(cross_section.workload_active_vm_count.min()),
                    "workload_active_count_max": int(cross_section.workload_active_vm_count.max()),
                    "workload_active_rule": "avg_cpu > 5 AND max_cpu > 20; illustrative, not the required target",
                    "bursty_std_threshold_pp": float(vm_profiles.bursty_std_threshold_pp.iloc[0]),
                    "classes": vm_classes.to_dict(orient="records"),
                },
                "horizon_headroom": {
                    "residual": "max(actual - prediction, 0), original training backtests only",
                    "samples_per_horizon": 3,
                    "envelope": "cumulative maximum of raw per-horizon q90/q95; display heuristic",
                    "interpretation": "CPU reference, not absolute provisioned capacity or calibrated coverage",
                    "buffer_p95_first_pp": float(capacity_schedule.buffer_p95_pp.iloc[0]),
                    "buffer_p95_last_pp": float(capacity_schedule.buffer_p95_pp.iloc[-1]),
                },
                "segment_variability": segment_variability.to_dict(orient="records"),
                "change_diagnostics_note": "Retrospective validation labels using each fitting-prefix positive-change q90; no selection/policy update",
                "selected_cpu_change_errors": change_summary[
                    (change_summary.target == "aggregate_cpu_utilization") &
                    (change_summary.model == selection["targets"]["aggregate_cpu_utilization"]["model"]) &
                    (change_summary.scope == "pooled")].to_dict(orient="records"),
                "normalization": {"definition": "MAE / mean of the complete 110-row training prefix",
                                  "training_means": training_means.to_dict(),
                                  "role": "Post-hoc reporting scale, not a model input or selection criterion"},
                "sensitivity": {"role": "Read-only diagnostics; final model and original schedule unchanged",
                                "preferred": sensitivity[sensitivity.preferred].to_dict(orient="records")},
               "selection": selection, "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                                       "matplotlib": matplotlib.__version__}}
    write_json("analysis_summary", summary)

    figure("training_cpu_coverage", per_time, cross_section)
    figure("cross_sectional_fan", cross_section)
    figure("horizon_headroom", capacity_schedule)
    figure("validation_schedules", sensitivity, comparison)
    figure("vm_utilization", vm.cpu_mean, bins, per_time)
    if segment_columns:
        figure("segments", segment, segment_columns)
    figure("autocorrelation", acf)
    figure("backtest_cpu", bt, selection, train.timestamp.iloc[0])
    figure("model_comparison", comparison)
    figure("horizon_errors", horizon)
    figure("residual_bias_peak_errors", origin_metrics)
    figure("data_quality", pd.Series({k: audit[k] for k in quality_keys[2:]}), coverage)
    print(json.dumps(json_safe({"output": str(output), "training_cpu": cpu_summary,
                               "selected_validation": residual,
                                "files": sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())}),
                      indent=2, allow_nan=False))
    return artifacts


if __name__ == "__main__":
    main()
