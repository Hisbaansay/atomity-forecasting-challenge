# Decision Log

## Latest Follow-Up: Evidence Before Features

Record the gaps and approve bounded diagnostics before interpreting their results. No new model, candidate feature, selection rule, or operational policy is introduced. The original 50/70/86 selection, seven candidates, frozen count 10,000, and frozen CPU **9.164556174985147%** remain unchanged; 9.16456% is display rounding only. Original CPU MAE/RMSE 0.23531/0.27463 pp are validation scores, not holdout performance.

| Evidence missing before this follow-up | Approved diagnostic | Risks and boundary |
|---|---|---|
| Sections 5-7 lacked a plotted all-candidate schedule comparison and consistent per-origin/pooled maximum-error evidence across alternatives | `evaluation.schedule_diagnostics`, 168 origin/pooled rows in `results/schedule_diagnostics.csv`, plus `validation_schedules.png` | Different schedules evaluate different periods; no favorable-schedule reselection or independent-test claim |
| Requested horizon-dependent headroom was not explicit | `capacity.py`, `results/capacity_schedule.csv`, and `horizon_headroom.png`, retaining raw positive-part residual quantiles and cumulative-max envelopes | Only three overlapping samples per horizon; no calibrated coverage, absolute provisioning, buffer retuning, or policy validation |
| Fleet mean concealed the actual per-timestamp P95/P99 distribution | `cross_section.py`, `results/cross_sectional_quantiles.csv`, and `cross_sectional_fan.png` | Population dispersion is not uncertainty around a forecast; high CPU alone cannot prove starvation or SLA violations |
| Reporting coverage was easily mistaken for workload activity; activity cuts were previously illustrative rather than an exported comparison | Adopt illustrative `avg_cpu > 5 AND interval max_cpu > 20`, export activity counts and `results/vm_classification_summary.csv` | Diagnostic only, not the official target, churn, future model input, incident definition, or capacity estimate |

**Headroom construction:** use the selected original CPU backtests and `max(actual - prediction, 0)`, with zeros for overprediction. Raw q90/q95 use linear interpolation on three samples at each horizon; cumulative maxima form a monotone **display envelope**, not calibrated P90/P95 coverage or a finding that raw risk must increase. At five minutes the P90/P95 envelopes are 0.042290517/0.047576832 pp; at 120 minutes they are 0.388657142/0.437239284 pp. Adding the P95 envelope to the frozen point gives clipped mean-CPU references 9.212133006664125%-9.601795459225082%, not provisioned capacity. The original **0.50 pp illustrative buffer remains unchanged**; dynamic headroom is additional diagnostic evidence, not model selection or policy calibration.

**Cross-sectional findings after implementation:** first-110 per-timestamp P95 spans 34.2264%-41.1702%, P99 82.5398%-91.0373%. Reporting count remains 10,000, while the adopted illustrative workload-active rule counts 2,561-2,951. VM training means partition into 4,635 low (<5%), 5,145 moderate (5%-50% inclusive), and 220 high (>50%). Separate >80% hot-average and interval-maximum flags, and positive temporal std at least its cross-VM training q90 for burstiness, overlap the classes and each other. They must not be summed as exclusive populations; interval maxima may be momentary. No future CPU enters these diagnostics or forecasts.

**Interpretation and future work:** the 24-step horizon is 21.8% of 110 training points. Local plateaus, ramps, and reversals plausibly explain ranking instability: alternative A favors Mean-12 (0.178385 versus persistence 0.208198 pp), and B favors damped trend (0.137625 versus 0.181039 pp). Do not compare pooled scores across schedules as identical test instances. Underprediction may pose service-dependent risk during scaling delay and overprediction may waste resources, but no observed service outcome or universal cost ratio is established. Recommend an equal-weight persistence/Mean-12 ensemble or adaptive windows only as **candidates to validate after collecting at least seven days (2,016 five-minute samples)**. Neither is implemented, validated, or automatically deployable after seven days; that milestone does not guarantee sufficient evidence. Independent evaluation and service-level policy validation remain necessary.

**Presentation and reproduction:** README now leads with the frozen results and an exactly three-command quickstart after the extracted-and-verified-data prerequisite. The primary workflow is `scripts/execute_notebook.py`, using real Jupyter `nbconvert` and saving 10 code cells with 12 figures; direct nbconvert remains optional and uses the same package. Three new figures bring the total to 12. The PDF retains six main pages with the fan on page 2, schedules on page 4, and backtests on page 5, followed by a variable-length appendix. Raw data/release archives and environments stay out of the submission but remain local for VS Code. These documentation and diagnostic changes add no models and do not score the public holdout. Verification outcomes belong to the final verification run, not a guessed test count.

## Comprehensive Follow-Up: Value Before Complexity

The then-current implementation and metrics were preserved in an external ZIP before this earlier follow-up; the forecast hash was unchanged. No new forecasting model was justified. The following diagnostics were approved before examining their results:

| Missing evidence / question | Value and output | Risks and boundaries | Placement |
|---|---|---|---|
| Are high-CPU readings spread widely or concentrated in repeatedly hot VMs? | Training-only counts of VMs ever >=80%, >=80% for half their readings, and the top 1% of VMs' share of >=80% readings | Descriptive hottest group selected retrospectively; threshold/1% are transparent summaries, not incident definitions | Brief report finding and supplementary JSON |
| Which segments fluctuate over time, not just differ in average load? | Temporal standard deviation and largest five-minute change of category/core/memory segment means, with VM counts | Segment means can hide VM bursts; small categories are not representative; no capacity inference | Compact existing segment table, no additional figure |
| Do candidates differ in operational downside? | Same per-origin/pooled underprediction frequency, zero-inclusive mean, q95, and maximum for all original candidates | Overlapping errors are dependent; descriptive only; do not recalibrate selection or buffer | Appendix CSV and concise notebook/report interpretation |
| Does persistence err more on abrupt upward changes? | Compare errors after upward steps above the fitting-prefix positive-change q90 with smaller upward steps | Small/non-independent subgroups and confounding by origin/horizon prevent causal or significance claims; cutoff comes only from each fitting prefix | Supplementary table and qualified report sentence |
| How does an operator apply the illustrative rule? | Explicit hypothetical arithmetic plus service/segment review | No invented Atomity threshold, cost, SLA, or automatic capacity action | Main operational section |

All new CPU summaries exclude the final 24 readings. Recorded metadata spans and inclusive endpoint comparisons are audit conventions, not proven uncensored VM lifetimes. The 1.34 million raw records yield only 134 aggregate timestamps, 110 for development; row volume is not independent temporal training evidence.

Notebook follow-up: added an executed Jupyter workflow that calls the same package, preserves model selection before EDA, exports predictions, and runs tests. Optional notebook/PDF dependencies do not change the minimal prediction runtime. Added a Markdown-derived PDF with six main pages and a figure appendix, plus isolated source-copy reproduction. These are review/reproducibility improvements, not new modeling experiments; holdout CPU remains unscored.

Final review: added tested dependency constraints after an independent audit flagged minimum-only requirements. The portable Python 3.11+ install remains supported through dependency resolution; exact constraints describe the tested CPython 3.13.9 environment. Integer source timestamps are preserved in CSV serialization. Neither change affects model selection or forecasts.

Submission cleanup: reorganized README navigation and generated artifacts into `figures/` (PNG) and `results/` (CSV/JSON). The report/PDF stay in `reports/`; `predictions.csv` and `predictions.selection.json` stay at the repository root. This is layout-only cleanup: no new models, selection changes, or holdout scoring.

This records methodological order, not invented wall-clock timestamps. Numerical evidence is in the generated [selection](results/selection.json), [audit](results/audit.json), and [analysis summary](results/analysis_summary.json). Model and policy choices are distinguished from later descriptive findings.

## 1. Establish The Contract

Read the complete root `ASSIGNMENT.md` and `SUBMISSION.md`, with target definitions in `DATASET.md`. Use the exact GitHub `data-v1` release and verify its manifest. Construct both full timestamp-level targets before splitting the first 110 observations from the final 24. Preserve the supplied five-minute grid and produce the required three-column CSV without an index. Root official source documents remain retained; the original README is preserved in ignored `.official/`.

## 2. Fix Data Rules

Require finite integer trace-relative timestamps and a regular 300-second cadence. Require nonblank VM IDs; all three CPU fields must be finite percentages in `[0,100]` and satisfy `min_cpu <= avg_cpu <= max_cpu`. Do not auto-convert ambiguous `[0,1]` CPU values to percentages. A value's small magnitude alone does not establish fractional units.

Exclude invalid records with warnings and audit counts. Average valid duplicate records per VM/timestamp before aggregating across VMs, avoiding duplicate VM weight. Count each valid represented VM once. Average VM `avg_cpu` equally, without treating bucket labels as exact capacity. Do not impute individual missing readings or silently resample; a whole-timestamp gap, irregular cadence, or timestamp lacking valid measurements raises an error. Require unique matching metadata IDs. These are fixed ingestion policies, not adjustments made after observing model scores.

## 3. Fix Candidates And Validation Before EDA

Use seven candidates in deterministic order: persistence; means over 6, 12, and 24 observations; SES alpha 0.2 and 0.5; fixed damped Holt alpha/beta/damping 0.3/0.1/0.9. Recursively update states from the fitting history; do not optimize the coefficients numerically. Use no metadata predictors, calendar assumptions, or holdout-derived features.

Set fitting-prefix origins to 50, 70, and 86, followed by 24-step forecasts without within-horizon updates. Select independently per target by pooled MAE; retain the first candidate on ties. Apply identical validation to baselines and alternatives. The pool contains 72 forecast errors but only 60 distinct timestamps because windows overlap; do not describe these as independent samples. Count forecasts use `np.rint` (nearest integer, ties to even) after bounding below by zero; CPU forecasts are clipped to `[0,100]`.

## 4. Select, Then Freeze Before Analysis

`analysis.py` calls `forecast_targets` and serializes the selection before computing training distributions or diagnostic plots. On the predefined 50/70/86 schedule, persistence wins CPU MAE at 0.23530966539049836 pp. All seven count candidates have zero MAE, RMSE, bias, and peak MAE; persistence wins the tie. Relative improvement is undefined for counts because baseline error is zero. CPU persistence has zero improvement over itself; every alternative has negative MAE improvement versus the best baseline on this schedule.

Do not revise the candidate set after EDA, promote damped trend for its smaller bias, or imply a complex model beat the baseline. The selected model is then applied using all 110 training observations for the final 24 predictions.

## 5. Verify Holdout Isolation

Never score or plot observed holdout CPU targets, nor use holdout labels for fitting, feature construction, tuning, or selection. Full-window schema, validity, cadence, and population checks are structural, not predictive evaluation. The full target series exists to satisfy construction/split requirements; the forecasting boundary reads only the training labels and future timestamps.

The generated isolation check replaces both final target columns with extreme finite values and, separately, all missing values. Forecasts, selection, and backtests remain exactly equal. This supports implementation-level isolation, not a holdout-performance claim. Analysis uses only the first 110 timestamps for CPU distributions, temporal patterns, segments, and forecast diagnostics; no holdout CPU labels are exported.

## 6. Interpret Training Evidence Without New Predictors

Constant population and complete coverage explain the count result. CPU segment differences motivate monitoring, not post-hoc model features. Source creation/deletion and CPU summary availability at an operational forecast origin is unknown, so these remain descriptive. Audit lifetime quantiles rather than equating the short observed window with complete VM life histories.

Inspect potential above-100 metadata statistics separately from interval CPU validation. The current audit reports maxima of 100.0, 99.19946388736228, and 99.519121830877 for metadata `max_cpu`, `avg_cpu`, and `p95_max_cpu`, respectively, and no above-100 values. Do not invent exceedances or equate source summaries with training-window measurements. Use categorical core/memory labels for EDA only, never as numerical capacity weights.

## 7. Diagnose Errors After Selection

Report MAE, RMSE, prediction-minus-actual bias, peak MAE, and improvement relative to the best baseline. Define peaks using each origin's fitting-prefix q90. All 20 CPU peak cases occur at origin 50; later empty peak groups are not zero-error successes. Examine per-origin bias and horizon errors because a small pooled bias can hide underprediction followed by overprediction. Keep this diagnostic interpretation separate from model selection.

## 8. Choose An Illustrative Operational Policy

After selection, inspect `max(actual - prediction, 0)`, including zero for overpredictions. Its empirical q95 is 0.42116067217222297 pp and observed maximum 0.4858214269332599 pp, approximately 0.49 pp. Round the maximum upward to **0.50 pp** for the labeled illustrative buffer. Do not claim that 0.49 strictly exceeds the observed maximum.

Neither the q95 nor the rounded maximum is calibrated future coverage: validation windows overlap and also selected the model, creating selection optimism. Compare forecast plus buffer with a hypothetical team-configured threshold, then request review of service and segment signals. Do not assume an actual 80% aggregate alert threshold, infer spare physical capacity from the unweighted fleet mean, or promise safety. Validate any real scaling policy with independent data and service-specific costs.

## 9. Reproduction And Scope

The README exposes installation, `python -m pytest`, prediction, and analysis commands. The source analysis module is the notebook equivalent. Record the running environment rather than copying versions from earlier generated evidence. Document the suite command rather than a test count that can become stale.

The prediction CLI supports tiny datasets via `--train-size` and `--horizon`; `--train-size 12 --horizon 4` needs 16 timestamps total. With horizon 4, validation requires at least 9 training rows. The full analysis intentionally requires the official 110/24 layout. Keep large archives/raw traces, environments, and caches out of version control.

## 10. Final Revision: Diagnose After Freeze, Without Retuning

The new `evaluation.py` adds read-only training diagnostics after the original selection has been serialized. It accepts exactly 110 training rows for schedule sensitivity, preventing accidental inclusion of the public holdout. The original 50/70/86 schedule, candidate definitions, selected persistence models, and final forecasts remain unchanged. Earlier selection statements describe those historical choices, not a claim that persistence is universally best.

On alternative origins 38/62/86, CPU `mean_12` has MAE 0.178384788340 pp versus persistence 0.208197957831 pp. On 30/54/78, `damped_trend` has MAE 0.137625318042 pp versus persistence 0.181039110054 pp. These schedules evaluate different periods, so pooled scores across schedules are not directly comparable. The changed preferences illustrate schedule sensitivity; they are not permission to choose a favorable schedule retrospectively or revise submitted predictions.

Report normalized MAE as MAE divided by the complete first-110 training mean, then multiply by 100 for percent presentation. CPU mean 9.41408901953901% and unrounded MAE 0.23530966539049836 pp give 2.499547910606%. This post-hoc reporting scale is not a model input, MAPE, or relative improvement over a baseline. CPU baseline improvement remains 0.00%; count improvement remains undefined because baseline MAE is zero. Zero count errors do not demonstrate learned churn.

The generated `results/validation_sensitivity.csv` contains 42 rows (three schedules, two targets, seven candidates). `results/origin_metrics.csv` adds maximum absolute error and first/last-six diagnostics; empty peak groups remain missing, not successful zero-error cases. `results/origin_horizon_errors.csv` preserves 144 selected-model rows (two targets, three origins, 24 steps), including signed and absolute errors. These exports expose error structure without refitting or changing selection.

Documentation includes PowerShell installation/data extraction and Python-based cross-platform checksum verification. A separate learning guide is retained locally, outside the submission archive. Three figures appear in the six-page main report, with supporting figures in the appendix; identical selected/baseline forecasts use one curve. These changes do not introduce production-performance claims or holdout evaluation. Current verification evidence and test totals are recorded in the verification document.
