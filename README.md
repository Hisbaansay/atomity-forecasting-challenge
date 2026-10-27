# Cloud Workload Forecasting

Submission for Atomity's AI/ML Engineering Internship challenge. The project analyzes Azure VM workloads and forecasts the next two hours of reporting VM count and aggregate mean CPU utilization.

**Review:** [Technical report (PDF)](reports/final_report.pdf) | [Report source](reports/final_report.md) | [Executed notebook](notebooks/workload_analysis.ipynb) | [Predictions](predictions.csv)

## Dataset And Targets

The exact [Atomity `data-v1` release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) contains **10,000 VMs, 134 five-minute timestamps, and 1,340,000 CPU readings**, curated from [Microsoft Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset). See [source attribution](SOURCES.md) and the [dataset contract](DATASET.md).

- `active_vm_count`: distinct VM IDs with a valid CPU observation at each timestamp.
- `aggregate_cpu_utilization`: arithmetic mean of valid VM-level `avg_cpu` readings, in percent, with equal weight per VM.

The first **110 timestamps** are used for development. The final **24 timestamps** form the two-hour holdout; their observed target values are not used for fitting, features, tuning, or model selection.

## Installation

Requires **Python 3.11+**. Run from the repository root:

```bash
git clone https://github.com/Hisbaansay/atomity-forecasting-challenge.git
cd atomity-forecasting-challenge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebook,report]'
```

On Windows PowerShell, create the environment with `py -3 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`; the subsequent `python` commands are the same. If activation is blocked, use `.\.venv\Scripts\python.exe` instead.

For the tested CPython **3.13.9** environment, install with `python -m pip install -c constraints.txt -e '.[dev,notebook,report]'`. On other Python versions, use the unconstrained installation above and run the tests.

## Dataset Setup

Download [data-v1.tar.gz](https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz) from the official release. Extract it into `challenge_data/`:

```bash
tar -xzf /path/to/data-v1.tar.gz -C challenge_data
python -c "from pathlib import Path; from atomity_forecasting.data import verify_checksums; print(verify_checksums(Path('challenge_data')))"
```

The directory must contain `vm_cpu_readings.csv.gz`, `vm_metadata.csv.gz`, `DATA_CARD.md`, and `SHA256SUMS`. All three checksum entries must verify. The loader repeats verification before reading the data. Raw datasets are intentionally excluded from Git.

## Generate Predictions

```bash
python -m atomity_forecasting.predict --data challenge_data --output predictions.csv
```

This generates the root-level [predictions.csv](predictions.csv) and a selection-metadata sidecar. The CSV has exactly **24 chronologically ordered rows** and these columns, with no index:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

Counts are nonnegative integers using nearest-integer rounding with ties to even; CPU forecasts are bounded to `[0,100]`. Forecasts are generated programmatically, without manual CSV edits. See the official [submission contract](SUBMISSION.md).

## Reproduce Analysis And Tests

```bash
python -m atomity_forecasting.analysis --data challenge_data --output reports
python scripts/execute_notebook.py
python -m pytest -q
```

Analysis exports tables to `reports/results/` and figures to `reports/figures/`. The notebook calculates results from raw data in memory and contains executable plotting cells; it does not load report CSVs or PNGs as analytical inputs. To run interactively in VS Code/Jupyter, select the project `.venv` kernel and **Run All**.

The notebook also regenerates the PDF. Alternatively, run `python scripts/render_report.py`. The supplied PDF embeds Times New Roman; rerendering without that installed font uses a documented serif fallback. Set `ATOMITY_FONT_DIR` to a complete local Times New Roman font directory to reproduce its typography.

The suite covers aggregation, invalid/missing readings, duplicate handling, chronological validation, holdout independence, reproducibility, prediction contracts, and alternative input sizes. [Verification evidence](reports/verification.md) documents the executed checks.

### Reviewer Smoke Inputs

The default prediction command supports a small VM population with **134 timestamps**, and is tested with **1, 3, and 5 VMs**, different CPU values, and shifted timestamps. For shorter inputs, specify the split and horizon explicitly:

```bash
python -m atomity_forecasting.predict --data /path/to/smoke_data --output smoke_predictions.csv --train-size 12 --horizon 4
python -m pytest tests/test_reviewer_smoke.py -q
```

This example needs 16 timestamps. Supply the same input filenames/schema and five-minute cadence, with matching metadata IDs. Use a separate directory with no checksum manifest or one generated for those files; a stale public-release manifest correctly fails verification. Only prediction is size-configurable; the full notebook/report analysis uses the official 110/24 layout.

## Methodology And Results

Valid readings satisfy `0 <= min_cpu <= avg_cpu <= max_cpu <= 100`. Valid duplicates are averaged within VM/timestamp before cross-VM aggregation. Invalid readings generate warnings and audit counts. Entire timestamp gaps fail explicitly; there is no silent imputation. Metadata utilization summaries are not predictors.

Seven candidates share expanding-window origins **50, 70, and 86**, each forecasting 24 steps without intermediate actuals: persistence, trailing means over 6/12/24 observations, SES with alpha 0.2/0.5, and fixed damped Holt trend. Selection minimizes pooled validation MAE separately per target. Persistence was selected for both.

| Target | Selected / best baseline | Validation MAE | Validation RMSE | Forecast |
|---|---|---:|---:|---:|
| Reporting VM count | Persistence / persistence | 0 VMs | 0 VMs | 10,000 VMs |
| Aggregate CPU | Persistence / persistence | 0.235310 pp | 0.274629 pp | 9.164556174985147% |

These are **training-period validation scores**, not holdout accuracy. CPU MAE improvement over the selected baseline is 0%; count improvement is undefined because baseline error is zero. The [complete comparison](reports/results/model_comparison.csv) includes peak error and bias.

Alternative training-only schedules favor Mean-12 at **38/62/86** and damped trend at **30/54/78**. They score different periods and demonstrate selection sensitivity, not grounds to change the original final model. See the [sensitivity results](reports/results/validation_sensitivity.csv).

## Findings And Limitations

- Complete-trace curation fixes reporting count at 10,000; zero count error does not demonstrate real VM churn prediction.
- The low aggregate mean masks high-utilization VM subsets. Per-timestamp P99 CPU ranges from 82.54% to 91.04%; aggregate mean CPU is not absolute compute demand or a sufficient downsizing signal.
- Only 110 aggregate development timestamps are available. Three overlapping validation windows provide limited independent evidence, and all original CPU peak cases occur in one window.
- Empirical headroom allowances use only three residuals per horizon. They are illustrative monitoring references, not calibrated prediction intervals, capacity estimates, or autonomous scaling policies.
- Daily/weekly seasonality, service-level protection, and external generalization are not established. The public holdout remains unscored.

## Repository Structure

```text
src/atomity_forecasting/  ingestion, forecasting, evaluation, analysis, plotting
src/cwfc/                preserved employer starter utilities
notebooks/               executed analysis with plotting code cells
reports/                 PDF/Markdown report, methodology, verification
reports/figures/          generated charts
reports/results/          supporting CSV/JSON evidence
tests/                   automated checks and reviewer smoke tests
scripts/                 notebook execution, report rendering, reproduction
predictions.csv          required forecast output
```

Official instructions are retained in [ASSIGNMENT.md](ASSIGNMENT.md), [DATASET.md](DATASET.md), and [SUBMISSION.md](SUBMISSION.md). Raw data, environments, caches, and local-only files are ignored by Git.
