"""Validated ingestion and exact employer target definitions."""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CPU_COLUMNS = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]


def verify_checksums(path: Path) -> dict[str, str]:
    """Verify supplied release manifest; synthetic smoke inputs need no manifest."""
    manifest = path / "SHA256SUMS"
    if not manifest.exists():
        warnings.warn("No SHA256SUMS: input is unverified (allowed for smoke data).")
        return {}
    results = {}
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if Path(name).name != name:
            raise ValueError("Checksum manifest must contain simple filenames")
        with (path / name).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        results[name] = digest
    return results


def clean_readings(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = set(CPU_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing CPU columns: {sorted(missing)}")
    frame = raw[CPU_COLUMNS].copy()
    audit = {"raw_rows": len(frame), "missing_by_column": frame.isna().sum().to_dict(),
             "raw_schema": {c: str(t) for c, t in frame.dtypes.items()},
             "duplicate_key_rows": int(frame.duplicated(["timestamp", "vm_id"]).sum()),
             "input_globally_time_sorted": bool(frame.timestamp.is_monotonic_increasing)}
    for col in ["timestamp", "min_cpu", "max_cpu", "avg_cpu"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame.timestamp.isna().any() or not np.isfinite(frame.timestamp).all():
        raise ValueError("Missing/nonfinite timestamp: cannot determine forecast grid")
    if not np.equal(frame.timestamp, np.floor(frame.timestamp)).all():
        raise ValueError("Timestamps must be integer trace-relative seconds")
    frame["timestamp"] = frame.timestamp.astype("int64")
    valid = frame.vm_id.notna() & frame.vm_id.astype(str).str.strip().ne("")
    cpu = frame[["min_cpu", "avg_cpu", "max_cpu"]]
    in_range = cpu.ge(0).all(axis=1) & cpu.le(100).all(axis=1)
    ordered = frame.min_cpu.le(frame.avg_cpu) & frame.avg_cpu.le(frame.max_cpu)
    valid &= in_range & ordered
    audit["invalid_cpu_or_id_rows"] = int((~valid).sum())
    audit["invalid_cpu_range_rows"] = int((~in_range).sum())
    audit["invalid_min_mean_max_rows"] = int((~ordered).sum())
    # Average valid duplicate readings within each VM first: no duplicate VM weight.
    clean = frame.loc[valid].groupby(["timestamp", "vm_id"], as_index=False)[
        ["min_cpu", "max_cpu", "avg_cpu"]].mean()
    audit["valid_rows_before_dedup"] = int(valid.sum())
    audit["clean_rows"] = len(clean)
    if (~valid).any() or audit["duplicate_key_rows"]:
        warnings.warn(f"CPU cleaning applied; inspect audit: {audit}")
    grid = np.sort(frame.timestamp.unique())
    if len(grid) < 2 or not np.equal(np.diff(grid), 300).all():
        raise ValueError("Timestamp gaps/irregular cadence: no silent resampling or imputation")
    if not np.array_equal(grid, np.sort(clean.timestamp.unique())):
        raise ValueError("A timestamp has no valid measurements; cannot form target")
    return clean, audit


def construct_targets(cpu: pd.DataFrame) -> pd.DataFrame:
    """Idempotent cleaning also protects direct callers with raw duplicate rows."""
    clean, _ = clean_readings(cpu)
    return clean.groupby("timestamp", as_index=False).agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    ).sort_values("timestamp").reset_index(drop=True)


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = Path(path)
    checksums = verify_checksums(path)
    raw = pd.read_csv(path / "vm_cpu_readings.csv.gz")
    metadata = pd.read_csv(path / "vm_metadata.csv.gz")
    clean, audit = clean_readings(raw)
    if "vm_id" not in metadata or metadata.vm_id.isna().any() or metadata.vm_id.duplicated().any():
        raise ValueError("Metadata requires unique, nonmissing vm_id")
    observed, described = set(clean.vm_id), set(metadata.vm_id)
    audit.update(checksums=checksums, metadata_rows=len(metadata),
                 metadata_missing_by_column=metadata.isna().sum().to_dict(),
                 vms_missing_metadata=len(observed - described),
                 metadata_without_readings=len(described - observed),
                 unique_vms=len(observed), unique_timestamps=int(clean.timestamp.nunique()),
                 timestamp_start=int(clean.timestamp.min()), timestamp_end=int(clean.timestamp.max()),
                 cadence_seconds=300)
    if observed != described:
        raise ValueError(f"Metadata population mismatch: {audit}")
    return clean, metadata, audit
