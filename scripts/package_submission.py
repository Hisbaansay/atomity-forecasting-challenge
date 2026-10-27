"""Build a reviewer archive without local environments, caches, or raw data."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_archive(root: Path, output: Path) -> Path:
    files = [root / name for name in (
        "README.md", "pyproject.toml", "constraints.txt", "predictions.csv",
        "predictions.selection.json", ".gitignore", "ASSIGNMENT.md", "DATASET.md",
        "SUBMISSION.md", "DATA_RELEASE.md", "SOURCES.md", "LICENSE",
        "challenge_data/DATA_CARD.md", "challenge_data/SHA256SUMS",
        "notebooks/workload_analysis.ipynb", "reports/final_report.md", "reports/final_report.pdf",
        "src/atomity_forecasting/predict.py",
        "src/atomity_forecasting/plotting.py",
        "src/atomity_forecasting/capacity.py", "src/atomity_forecasting/cross_section.py",
        "reports/results/capacity_schedule.csv", "reports/results/cross_sectional_quantiles.csv",
        "reports/results/schedule_diagnostics.csv", "reports/results/vm_classification_summary.csv",
        "reports/figures/cross_sectional_fan.png", "reports/figures/horizon_headroom.png",
        "reports/figures/validation_schedules.png",
    )]
    local_only = {"reports/revision_record.md", "reports/results/revision_integrity.json",
                  "scripts/verify_revision.py"}
    for directory, suffixes in (
        ("src", {".py"}), ("tests", {".py"}), ("scripts", {".py"}),
        ("notebooks", {".ipynb"}), ("reports", {".md", ".pdf", ".png", ".csv", ".json"}),
    ):
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in suffixes
                     and p.relative_to(root).as_posix() not in local_only
                     and not any(part.startswith(".") or part == "__pycache__"
                                 or part.endswith(".egg-info")
                                 for part in p.relative_to(root / directory).parts))
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(set(files))
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"Required submission file missing: {file}")
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, Path("atomity-forecasting-challenge") / file.relative_to(root))
        if archive.testzip() is not None:
            raise ValueError("Archive integrity check failed")
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist/atomity-forecasting-challenge.zip")
    args = parser.parse_args()
    output = build_archive(root, args.output)
    with ZipFile(output) as archive:
        print(f"Packaged {len(archive.namelist())} files: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
