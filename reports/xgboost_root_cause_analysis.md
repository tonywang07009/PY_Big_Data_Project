# XGBoost Root Cause Analysis

Legacy diagnostic report. This report explains earlier pipeline experiments and is not the formal project result. The formal project version is the Gate 7 county-month ensemble documented in `reports/final_project_report.md`.

## Data Profile

- Raw rows: `110,373`
- County-month groups: `931`
- Avg rows per county-month: `118.55`
- donation_ratio zero-row ratio: `0.0148`
- donation_count zero-row ratio: `0.0148`

## Pipeline Results

| pipeline | target | transform | train_rows | valid_rows | feature_count | rmse | mae | smape | epsilon_mape | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_row_baseline | donation_count | identity | 47346 | 48460 | 118 | 249590.301350 | 120502.593750 | 38.881973 | 30.697399 | 0.946285 |
| county_month_baseline | donation_count | identity | 361 | 456 | 16 | 267281.761862 | 99280.296875 | 39.757405 | 44.604463 | 0.914751 |
| hybrid_county_month | donation_count | identity | 361 | 456 | 17 | 289546.723055 | 102552.156250 | 44.309699 | 36.761642 | 0.899957 |
| county_month_baseline | donation_count | log1p | 361 | 456 | 16 | 503785.414128 | 153899.562500 | 66.022990 | 40.076400 | 0.697141 |
| raw_row_baseline | donation_count | log1p | 47346 | 48460 | 118 | 602477.657677 | 203662.703125 | 66.862197 | 41.702688 | 0.687016 |
| hybrid_county_month | donation_count | log1p | 361 | 456 | 17 | 511866.142502 | 165624.968750 | 70.270706 | 43.415825 | 0.687347 |
| county_month_baseline | donation_ratio | identity | 361 | 456 | 17 | 0.003560 | 0.001901 | 37.353059 | 27.270068 | 0.482633 |
| hybrid_county_month | donation_ratio | identity | 361 | 456 | 17 | 0.003727 | 0.002030 | 42.940660 | 30.021711 | 0.432929 |
| raw_row_baseline | donation_ratio | identity | 47346 | 48460 | 118 | 0.004530 | 0.002430 | 50.164703 | 33.111466 | 0.350392 |

## Stepwise Decision Table

| Step | Question | Evidence | Decision |
| --- | --- | --- | --- |
| 1 | Should we use raw-row directly? | Raw data repeats the same county-month target across ~118.6 rows on average. Raw-row pipelines did not dominate the best metric rows consistently. | Do not default to raw-row direct modeling; treat it as a weighting-heavy alternative. |
| 2 | Which target is more stable for XGBoost? | Best `donation_ratio` pipeline: `county_month_baseline` with sMAPE `37.3531`. Best `donation_count` pipeline: `raw_row_baseline` with sMAPE `38.8820`. | Prefer the target with the lower and more stable error across multiple pipelines. |
| 3 | Is hybrid county-month worth it? | Hybrid keeps county-month target semantics while adding raw-derived structure. Compare its result rows against plain county-month and raw-row. | If hybrid beats both baseline variants on the chosen target, use hybrid as the next implementation baseline. |
| 4 | Should normalization be a first-order concern? | All tested pipelines use XGBoost, which is usually insensitive to monotonic scaling. | Focus on grain, target, lag design, and structural features before normalization. |
| 5 | Should we split by time first or by group? | A single full-data time split keeps comparisons fair across target/grain choices. | Finalize a full-data baseline first, then revisit grouping only if it improves holdout metrics. |

## Recommended Next Step

- This diagnostic step has been superseded by the Gate 7 formal ensemble.
- Use `reports/final_project_report.md` and `model_outputs/donation_count_xgboost_ensemble/` for the current project result.
- Keep this report only as historical evidence for why the raw-row repeated-target approach was not selected as the final project line.
