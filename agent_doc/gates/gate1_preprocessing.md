# Gate 1 - Preprocessing and Split Construction

## Purpose

Build the county-month modeling frame and construct monthly expanding walk-forward folds for the formal Gate 7 ensemble.

## Formal Micro-Tasks

- T1-A: aggregate raw county x industry x carrier-type rows into one county-month row
- T1-B: add `year`, `month_num`, `month_sin`, `month_cos`, and `time_index`
- T1-C: preserve raw-derived structure features such as carrier usage, industry concentration, and row count
- T1-D: create monthly expanding folds from `2023-01` through `2024-12`
- T1-E: write the county-month matrix under `model_outputs/donation_count_xgboost_ensemble/`

## Required Evidence

- Validation months are never used to fit train-fold preprocessing state.
- Each validation fold contains one month only.
- Training rows are strictly earlier than the validation month.

## Expected Outputs

- `model_outputs/donation_count_xgboost_ensemble/county_month_matrix.csv`
- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward/`
- `model_outputs/gate1_verdict.json`
