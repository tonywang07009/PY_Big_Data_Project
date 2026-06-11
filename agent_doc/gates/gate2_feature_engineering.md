# Gate 2 - Feature Engineering

## Purpose

Prepare model-ready economic, structural, seasonal, and target-history features for the formal Gate 7 ensemble.

## Formal Micro-Tasks

- T2-A: compute invoice scale and value features
- T2-B: compute carrier and industry structure features
- T2-C: compute date-safe `donation_count_lag1/2/3`
- T2-D: compute rolling 3-month target summary features
- T2-E: standardize numeric features per `county_label` using train-fold rows only
- T2-F: one-hot encode `county_label` and align validation columns to train columns

## Required Evidence

- `county_numeric_scalers.json` exists for each fold.
- Validation feature columns exactly match train feature columns.
- Target `donation_count` is not standardized.
- Lag features are joined by date and do not use future rows.

## Expected Outputs

- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward/fold_*/county_numeric_scalers.json`
- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward/fold_*/metrics.json`
- `model_outputs/gate2_verdict.json`
