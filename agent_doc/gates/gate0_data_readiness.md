# Gate 0 - Data Readiness

## Purpose

Confirm that the encoded source data is present, readable, and suitable for the formal Gate 7 county-month `donation_count` ensemble.

## Formal Micro-Tasks

- T0-A: read `data/raw/encoded_ml_dataset.csv`
- T0-B: confirm required columns for county-month aggregation and ensemble training
- T0-C: confirm all available `county_label` values and month coverage
- T0-D: write `model_outputs/gate0_verdict.json`

## Required Evidence

- Source file exists and is readable.
- Target column is `donation_count`.
- Formal scope is all available county labels.
- No synthetic county rows are added.

## Expected Outputs

- `model_outputs/gate0_verdict.json`
- `model_outputs/donation_count_xgboost_ensemble/county_month_matrix.csv`
- `reports/final_project_report.md`
