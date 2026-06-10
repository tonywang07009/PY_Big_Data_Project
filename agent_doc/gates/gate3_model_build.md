# Gate 3 - Model Build

## Purpose

Train the formal donation-ratio models for the scoped county-month matrix and compare candidate regressors under a fixed time-based split.

## Current Main Implementation

- Entrypoint: `run_all.py`
- Main code path:
  - `src/model.run()`
- Train policy:
  - `year <= 2022`
- Test policy:
  - `year >= 2023`

## Current Model Set

- Linear Regression
- Ridge
- Lasso
- Random Forest
- XGBoost

## Current Build Flow

1. Drop rows missing required feature columns.
2. Split train/test by year.
3. Apply:
   - z-score scaling + one-hot encoding for linear models
   - passthrough numeric features + one-hot encoding for tree models
4. Train all candidate models.
5. Compare test RMSE and test R2.
6. Select the best model.
7. Export:
   - model comparison table
   - comparison plot
   - predicted-vs-actual scatter for the best model
   - Ridge coefficients
   - priority scores

## Current Main Output Artifacts

- `model_outputs/metrics/model_comparison.csv`
- `model_outputs/metrics/model_comparison.png`
- `model_outputs/metrics/prediction_scatter.png`
- `model_outputs/metrics/ridge_coefficients.csv`
- `model_outputs/priority_scores.csv`
- `model_outputs/gate3_verdict.json`

## Supplementary Workflow Added In This Chat

- Script: `train_donation_ratio_county_scaled_xgboost.py`
- Purpose:
  - train XGBoost on `data/county_zscore_split/train_scaled_all_counties.csv`
  - validate on `data/county_zscore_split/test_scaled_all_counties.csv`
  - use `donation_ratio` as target
- Output root:
  - `model_outputs/donation_ratio_county_scaled_xgboost/`

This branch is an auxiliary modeling path and does not replace the formal `run_all.py` model-comparison stage.

## Current Status

- Implemented and active.
- The formal delivery still treats `src/model.py` as the canonical Gate 3 path.
