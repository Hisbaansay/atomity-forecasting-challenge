# Submission Verification

## Reproduce

Tested with CPython 3.13.9 on macOS arm64. Download the exact release and verify its manifest as described in the README, then run:

```bash
python -m pip install -c constraints.txt -e '.[dev,notebook,report]'
python scripts/execute_notebook.py
python -m pytest -q
python scripts/verify_clean_copy.py
python scripts/package_submission.py
```

The notebook constructs targets, freezes selection, generates training-only analyses and plots, writes forecasts, runs tests, and renders the PDF. The execution script invokes real nbconvert using the selected environment. `results/clean_copy_verification.json` records reproduction from an extracted archive in a fresh environment, with the release data supplied separately.

## Checks

| Area | Evidence |
|---|---|
| Dataset integrity | All three release checksums; `results/audit.json` |
| Target definitions and cleaning | Distinct valid VMs, equal-VM CPU mean; duplicate/missing/invalid-data tests |
| Chronological protocol | First 110 timestamps; 24-step forecasts; origins 50/70/86 |
| Holdout isolation | Extreme/missing-label perturbations preserve selection, backtests, and forecasts |
| Model sensitivity | `results/validation_sensitivity.csv` and `schedule_diagnostics.csv` |
| Error analysis | Origin, horizon, peak, directional, and change-group exports; absent peak metrics are N/A |
| CPU heterogeneity | Training-only quantiles, concentration, segment variability, and VM classes |
| Headroom | 24-row empirical schedule; three samples/horizon; monotone envelopes are not calibrated coverage |
| Reporting versus activity | Required reporting count is distinct from the illustrative CPU-activity rule |
| Notebook figures | 12 executable plotting cells with ordinary Python definitions/calls; synchronized with the package; no image inputs |
| Report | Six main pages plus six appendix pages; Times New Roman embedded; justified body paragraphs |
| Predictions | Exact schema, 24 timestamps, finite/range checks, unchanged SHA-256 |
| Reviewer smoke data | Three-VM CLI test with configurable training size and horizon |
| Packaging | Required-file checks, ZIP integrity checks, and exclusions tested |

The current suite passes **153 tests** (workspace: 8.32s), including a check that segment bucket labels remain horizontal. Before the four additional reviewer-smoke tests, fresh extracted-archive reproduction passed 148 tests in 6.79s. The notebook executes all **22 code cells** and generates **12 figures** without errors. Tests cover numerical correctness, leakage, executable chart definitions, typography, nonoverlapping layouts, and size-independent CLI execution. The PDF embeds Times New Roman regular/bold and uses justified body text.

## Clean GitHub Checkout And Reviewer Smoke Inputs

The public repository was cloned into a separate temporary directory and installed into a newly created virtual environment using the documented constrained installation. Import-path inspection confirmed that the prediction module came from that clone, not the working project. The four new subprocess tests passed against that published code (**4 passed in 9.81s**), without downloading or reading the official Azure CSVs:

- Default 110/24 prediction command with **1, 3, and 5 VMs**, 134 timestamps, shifted trace-relative times, and different CPU values.
- A **three-VM, 16-timestamp** series with `--train-size 12 --horizon 4`.
- For each default-split input, perturbing only the final synthetic CPU values preserves forecast CSV bytes and selection metadata.

Each CLI subprocess runs from its temporary input directory with `PYTHONPATH` removed. Input-specific checksums are verified where provided; a checksum-free synthetic input is accepted with an explicit warning. Replacing raw files while retaining the public release's old checksum manifest correctly fails integrity verification. A shorter series requires explicit split/horizon settings; the protocol is never inferred from its target values. These synthetic inputs exercise execution only and do not affect the submission forecasts or validation scores.

The plotting cleanup moved chart implementations out of the analysis orchestrator and removed three duplicate notebook implementations. All figure creation now lives in `plotting.py`; it accepts computed data and returns Figure objects, while `save_figure` handles layout/export. Analysis returns its calculated DataFrames, JSON-ready summaries, and live Figure objects. The notebook uses those objects directly, without reading report CSV/JSON/PNG exports. Tables set dark text and light backgrounds explicitly to avoid theme-dependent low contrast. No model, split, threshold, target definition, or statistical calculation changed.

The final notebook repair removes stale `on_figure` calls and report-file reads. Every table, including model selection, metadata summaries, and forecasts, uses explicit high-contrast colors. Plot definitions now reside in genuine code cells: the synchronization utility inserts the package definition and the cell executes it against in-memory plot inputs. Analysis skips drawing in notebook mode, so diagrams are not drawn twice. Report CSV/JSON files remain output-only exports, and no HTML code-display blocks remain.

## Forecast Preservation

The data, forecasting, original model selection, and prediction modules remain unchanged. All original backtest rows and comparison metrics are preserved. The forecast contains 24 rows with count 10,000 and full-precision CPU 9.164556174985147. SHA-256:

```text
e224a7d8e711b4d7d47da25f0bcb1a52e24ff8eb04f76b39b321d0ce04c9da5f
```

Independent reviews reconstructed the training targets, candidate forecasts, sensitivity scores, and supplementary diagnostics using direct pandas/NumPy calculations. No substantive numerical or leakage defect was found. Plot layout and typography changes do not change forecasts or metrics.

## Submission Contents And Limits

The submission repository contains the source, tests, executed notebook, report in Markdown/PDF, relevant figures/result tables, methodology notes, dependency specifications, official contracts, and attribution. Local interview notes, revision journals, baseline-comparison utilities, editor settings, environments, raw data, ZIPs, caches, and backups are ignored and excluded. Licensed font binaries are not redistributed; the PDF embeds font subsets. A machine without Times New Roman uses documented serif fallbacks when rerendering.

The public holdout remains unscored. Short history, overlapping windows, schedule-sensitive rankings, and sparse peak examples limit generalization. Headroom is an empirical diagnostic, not an SLA guarantee or capacity estimate. Windows instructions and the VS Code UI were not automated. Repository publication is authorized for candidate review and sharing; no email or formal application submission is sent by this project.
