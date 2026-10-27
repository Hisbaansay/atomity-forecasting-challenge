# Cloud Workload Forecasting

## 60-Second Overview

Training-only analysis and two-hour forecasts for the exact Azure `data-v1` challenge: **10,000 VMs, 134 five-minute timestamps, 110 training points, and 24 future points**. The 1,340,000 readings are not 1,340,000 independent temporal examples. Persistence is the frozen selection for both targets; the public holdout is not scored or plotted as an observed CPU outcome.

| Target | Frozen model | Forecast at every future step | Original validation MAE | Original validation RMSE |
|---|---|---:|---:|---:|
| Reporting VM count (`active_vm_count`) | Persistence | 10,000 VMs | 0 VMs | 0 VMs |
| Equal-VM mean CPU (`aggregate_cpu_utilization`) | Persistence | 9.16456% | 0.23531 pp | 0.27463 pp |

CPU **9.16456% is display rounding only**: the frozen value is **9.164556174985147%** in `predictions.csv`. Errors above are training-only validation scores at origins **50/70/86**, not holdout accuracy. All seven count candidates tie at zero; this is fixed reporting coverage, not learned churn. Selected persistence is also the best original-schedule baseline, so CPU MAE improvement is 0.00%; count improvement is undefined because baseline error is zero.

Four findings matter:

- **Schedule sensitivity:** alternative training-only periods favor Mean-12 or damped trend. Local plateaus, ramps, and reversals affect rankings; they do not justify retrospective reselection.
- **The mean hides the tail:** per-timestamp cross-VM P95 ranges from **34.2264% to 41.1702%**, and P99 from **82.5398% to 91.0373%**. The fan describes population dispersion, not forecast uncertainty or observed SLA violations.
- **Headroom is empirical:** horizon-specific P90/P95 diagnostics have only **three overlapping validation samples per horizon**. Their monotone display envelopes are not calibrated coverage or absolute provisioning requirements.
- **Reporting is not workload activity:** reporting count is always **10,000**, while the illustrative `avg_cpu > 5 AND max_cpu > 20` rule counts **2,561-2,951** workload-active VMs. It does not redefine the forecast target.

Read the [report](reports/final_report.md) or [PDF](reports/final_report.pdf) for the findings and the [decision log](reports/decision_log.md) for modeling choices. The retained [assignment](ASSIGNMENT.md), [dataset contract](DATASET.md), and [submission contract](SUBMISSION.md) are authoritative.

## Quickstart

Clone the submission repository, then follow dataset setup below:

```bash
git clone https://github.com/Hisbaansay/atomity-forecasting-challenge.git
cd atomity-forecasting-challenge
```

**Prerequisite:** run from the repository root with the exact `data-v1` dataset already extracted into `challenge_data/` and its manifest verified, as described below. Use the tested **CPython 3.13.9** environment for `constraints.txt`; the package minimum is Python 3.11, but constrained pins may need a newer interpreter.

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -c constraints.txt -e '.[dev,notebook,report]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/execute_notebook.py
```

The third command is the primary end-to-end workflow. It uses real Jupyter `nbconvert` in a clean kernel, saves the executed notebook's **22 code cells and 12 figures**, and runs analysis, prediction checks, tests, and PDF rendering. Selection and frozen predictions are unchanged. Dependencies are declared in `pyproject.toml`.

All 12 diagrams have **real executable notebook cells** containing their Python definitions and drawing calls. Running a cell produces its chart immediately below. `scripts/materialize_notebook_plots.py` synchronizes those definitions from [`plotting.py`](src/atomity_forecasting/plotting.py), so they do not become separately maintained implementations. The notebook computes plot inputs from raw data, draws each figure once, and exports PNGs for the report; it never loads them as inputs. These are code cells, not rendered screenshots or HTML source blocks.

Tables also come directly from calculated in-memory DataFrames. The notebook never reads report CSVs or JSON files to obtain analytical results. CSVs are still generated for review and reuse, but are outputs only. Table text/background colors are explicitly set for readable results in both light and dark VS Code themes.

## Dataset Setup

Download only [atomityhq/forecasting-challenge, release data-v1](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1), specifically [data-v1.tar.gz](https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz). The challenge subset is curated from [Microsoft Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset); see [sources and attribution](SOURCES.md). Do not substitute the full upstream trace.

With GitHub CLI (`gh`) installed:

```bash
download_dir="$(mktemp -d)"
gh release download data-v1 --repo atomityhq/forecasting-challenge --pattern data-v1.tar.gz --dir "$download_dir"
mkdir -p challenge_data
tar -xzf "$download_dir/data-v1.tar.gz" -C challenge_data
```

`challenge_data/` must directly contain `vm_cpu_readings.csv.gz`, `vm_metadata.csv.gz`, `DATA_CARD.md`, and `SHA256SUMS`. After installing the package, verify every manifest entry with Python; the loader repeats this check without a Unix checksum utility:

```bash
.venv/bin/python -c "from pathlib import Path; from atomity_forecasting.data import verify_checksums; print(verify_checksums(Path('challenge_data')))"
```

For the first setup, install the environment before this verification, then satisfy the verified-data prerequisite before tests and end-to-end execution. Complete-trace eligibility fixes observation coverage; recorded creation/deletion spans have uncertain censoring semantics and do not establish predictable real fleet churn.

## Windows And VS Code

PowerShell, with the Python launcher and modern Windows `tar`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c constraints.txt -e ".[dev,notebook,report]"
$downloadDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $downloadDir | Out-Null
gh release download data-v1 --repo atomityhq/forecasting-challenge --pattern data-v1.tar.gz --dir "$downloadDir"
New-Item -ItemType Directory -Force -Path challenge_data | Out-Null
tar -xzf "$downloadDir/data-v1.tar.gz" -C challenge_data
.\.venv\Scripts\python.exe -c "from pathlib import Path; from atomity_forecasting.data import verify_checksums; print(verify_checksums(Path('challenge_data')))"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/execute_notebook.py
```

Activation is optional (`.\.venv\Scripts\Activate.ps1`); explicit interpreter paths work even if activation is blocked. On other Python versions, unconstrained `-m pip install -e ".[dev,notebook,report]"` can resolve compatible dependencies, but run verification in that environment rather than claiming the tested constrained environment.

In VS Code, open this folder, install Microsoft's **Python** and **Jupyter** extensions, and open [workload_analysis.ipynb](notebooks/workload_analysis.ipynb). Select `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`) as the kernel and **Run All** after dataset verification. Keep the local dataset and environment for interactive work; exclude both from the archive and version control. A fresh kernel does not depend on stored outputs.

## Reproduction Details

Optional direct execution after activating the environment (activation is important because Jupyter dispatches subcommands through `PATH`):

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/workload_analysis.ipynb --ExecutePreprocessor.timeout=600
```

Individual entry points, when only part of the workflow is needed:

```bash
.venv/bin/python -m atomity_forecasting.predict --data challenge_data --output predictions.csv
.venv/bin/python -m atomity_forecasting.analysis --data challenge_data --output reports
.venv/bin/python scripts/render_report.py
.venv/bin/python scripts/verify_clean_copy.py
```

Prediction selects using training-only backtests, refits on all 110 training points, and writes root `predictions.csv` plus `predictions.selection.json`. The CSV has exactly `timestamp,active_vm_count,aggregate_cpu_utilization`, no index, and 24 ordered future rows. Counts use `np.rint` (nearest integer, ties to even), bounded below by zero; CPU is clipped to `[0,100]`. Do not manually round or edit the CSV.

Analysis regenerates `reports/results/` and `reports/figures/`, not hand-written report prose or the decision log. There are **12 figures**, including the new cross-sectional fan, validation-schedule comparison, and horizon-headroom diagnostic. The PDF has **six main pages plus a variable-length appendix**: the main figures are the fan on page 2, schedules on page 4, and backtests on page 5. Identical selected/baseline forecasts share one labeled curve.

The included PDF embeds Times New Roman and uses justified body text. Charts use the same font family. To reproduce that appearance, install Times New Roman locally or set `ATOMITY_FONT_DIR` to a directory containing its regular, bold, italic, and bold-italic fonts. Font files are not redistributed. Without them, rendering remains functional using PDF Times and Matplotlib's bundled STIX serif fallback; forecast values do not depend on fonts. Plain Markdown previews use the viewer's own typography.

New evidence includes `capacity_schedule.csv` (24 rows), `cross_sectional_quantiles.csv` (110 rows), `vm_classification_summary.csv`, and `schedule_diagnostics.csv` (168 origin/pooled rows). Existing sensitivity, origin/horizon, directional-error, concentration, and segment exports remain supplementary diagnostics, not new selection criteria. See [verification](reports/verification.md) and [clean-copy evidence](reports/results/clean_copy_verification.json) for the actual verification run. Isolated reproduction needs separately supplied release data and package-index/cache access and checks byte-identical predictions; do not infer success from an old artifact or memorize test counts.

The prediction CLI also supports synthetic smoke inputs:

```bash
.venv/bin/python -m atomity_forecasting.predict --data /path/to/smoke_data --output /tmp/smoke_predictions.csv --train-size 12 --horizon 4
```

This needs 16 timestamps total; horizon 4 requires at least 9 training rows. Preserve schema, filenames, five-minute cadence, and unique matching metadata IDs. Synthetic inputs may omit the manifest with an explicit warning. Full analysis requires the official 110/24 layout.

For reviewer smoke inputs, the default prediction command needs **134 timestamps**, not 10,000 VMs. It is tested with **1, 3, and 5 VMs**, different CPU values, and shifted trace-relative timestamps. Shorter series use the explicit split/horizon flags above; the split is never guessed silently. Synthetic data should be supplied in its own directory, without the public release's checksum manifest, or with a manifest generated for those input files. A stale public manifest must fail integrity verification rather than be bypassed. Only prediction is size-configurable; the full release-specific notebook/report analysis intentionally uses 110/24.

Run just these public-command smoke checks without downloading Azure data:

```bash
.venv/bin/python -m pytest tests/test_reviewer_smoke.py -q
```

These checks invoke the CLI as a subprocess from outside the repository, remove `PYTHONPATH`, verify output timestamps/ranges, and perturb only future synthetic CPU labels to confirm identical forecasts and selection.

## Method And Findings

Validate finite CPU percentages with `0 <= min_cpu <= avg_cpu <= max_cpu <= 100`; exclude invalid records with warnings. Average valid duplicate VM/timestamp rows before cross-VM aggregation. Missing readings contribute neither count nor CPU; whole-timestamp gaps fail rather than being imputed. `active_vm_count` counts valid represented VMs, and aggregate CPU weights each VM's `avg_cpu` equally, not by cores.

Seven candidates were fixed before EDA: persistence, trailing means of 6/12/24 points, SES alpha 0.2/0.5, and damped Holt alpha/beta/damping 0.3/0.1/0.9. Selection minimizes pooled MAE separately per target on origins 50/70/86. The 72 errors cover only 60 distinct timestamps. Deterministic ties favor persistence; no numerical coefficient optimization, metadata predictors, or new models were added. Holdout-label perturbations must leave forecasts, selection, and backtests unchanged.

| Training-only schedule | CPU diagnostic favorite | Favorite MAE (pp) | Persistence MAE (pp) |
|---|---|---:|---:|
| Original 50/70/86 | Persistence | 0.235310 | 0.235310 |
| Alternative A 38/62/86 | Mean-12 | 0.178385 | 0.208198 |
| Alternative B 30/54/78 | Damped trend | 0.137625 | 0.181039 |

Different schedules score different periods, so compare candidates within a row, not pooled scores across rows. With a 24-step horizon equal to **21.8%** of 110 training points, local regimes strongly influence rankings. Original CPU peak MAE is 0.306193 pp; all 20 peak cases come from one origin. CPU MAE divided by the first-110 training mean (9.41408901953901%) is **2.499548%**, not MAPE or baseline improvement.

Training VM means partition into **4,635 low (<5%)**, **5,145 moderate (5%-50%, inclusive)**, and **220 high (>50%)**. Hot-average (>80%), hot-interval-maximum (>80%), and burstiness flags are separate, overlapping descriptions, not additional exclusive classes. Burstiness means positive within-VM temporal standard deviation at least the cross-VM training q90; interval maxima may be momentary. None is an incident, SLA, or starvation label.

## Headroom And Limits

`capacity.py` uses only selected-model original validation residuals: `max(actual - prediction, 0)`, including zeros for overprediction. It computes raw empirical q90/q95 per horizon from **three samples**, then applies cumulative maxima to create monotone display envelopes. Raw risk need not rise with horizon; monotonicity is imposed, not discovered.

| Horizon | P90 envelope buffer (pp) | P95 envelope buffer (pp) | Frozen point + P95 mean-CPU reference |
|---|---:|---:|---:|
| 5 minutes | 0.042290517 | 0.047576832 | 9.212133006664125% |
| 120 minutes | 0.388657142 | 0.437239284 | 9.601795459225082% |

References add the envelope to the frozen point and clip to `[0,100]`; they are not absolute capacity, prediction intervals, or guaranteed P90/P95 coverage. The original **0.50 pp illustrative review buffer remains unchanged**. The dynamic schedule is an additional diagnostic, not policy calibration or model selection. Underprediction could create service-dependent resource risk during scaling delay; overprediction could waste resources or review effort. These traces establish neither actual SLA violations/starvation nor a universal cost ratio. Low equal-VM mean CPU alone cannot justify downsizing.

Collect at least **seven days (2,016 five-minute samples)** before testing an **equal-weight persistence + Mean-12 ensemble or adaptive windows as candidates to validate**. Neither is implemented or validated here, and seven days is a collection milestone, not a guarantee of sufficient evidence or permission to deploy. Use independent, non-overlapping evaluation, service-level signals, scaling latency, and trustworthy capacity data before changing policy. Daily seasonality, real churn, calibrated coverage, and external generalization remain unestablished.

## Repository Map

```text
src/atomity_forecasting/     data, forecasting, evaluation, analysis, and plotting
src/cwfc/                   preserved employer starter utilities
tests/                      implementation and contract tests
notebooks/                  saved executed package-backed notebook
reports/                    report source/PDF, decision log, verification
reports/figures/            generated figures
reports/results/            generated CSV/JSON evidence
scripts/                    execution, rendering, verification, packaging
predictions.csv             frozen 24-row forecast
predictions.selection.json  prediction-selection sidecar
```

The GitHub repository contains the implementation, tests, executed notebook, report, relevant evidence, and official contracts/attribution. Raw datasets, environments, caches, ZIP archives, local editor settings, interview notes, and revision-history files are ignored. These local-only files remain in the workspace where useful. Download and verify the release data separately before running the project. `scripts/package_submission.py` is an optional export utility also used by the isolated reproduction check; no ZIP is required to review the repository.
