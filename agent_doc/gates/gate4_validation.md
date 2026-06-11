# Gate 4 - Validation

## Purpose

Evaluate whether the county-scaled `donation_ratio` models generalize under both the fixed 2024 holdout and the rolling year-validation workflow.

## Current Validation Paths

### Fixed train/test validation

- Used by:
  - `train_donation_ratio_county_scaled_xgboost.py`
  - `compare_donation_ratio_county_scaled_models.py`
- Policy:
  - train: `year < 2024`
  - test: `year >= 2024`

### Rolling year validation

- Used by:
  - `train_donation_ratio_county_scaled_year_cv.py`
- Policy:
  - for each validation year, train on all earlier years
  - recompute county-level scaling inside the fold

## Current Fold Sequence

- `2020`
- `2022`
- `2023`
- `2024`

## Current Validation Metrics

- `RMSE`
- `MAE`
- `sMAPE`
- `epsilon-MAPE`
- `R2`

## Current Output Artifacts

- Fixed holdout:
  - `model_outputs/donation_ratio_county_scaled_xgboost/performance_result.png`
  - `model_outputs/donation_ratio_county_scaled_model_comparison/model_comparison.png`
  - `model_outputs/donation_ratio_county_scaled_model_comparison/prediction_scatter.png`
- Rolling validation:
  - `model_outputs/donation_ratio_county_scaled_year_cv/metrics_by_fold.csv`
  - `model_outputs/donation_ratio_county_scaled_year_cv/metrics_summary.json`
  - `model_outputs/donation_ratio_county_scaled_year_cv/validation_predictions_by_fold.csv`
  - `model_outputs/donation_ratio_county_scaled_year_cv/performance_result.png`

## Current Status

- Implemented and active.
- The current validation story is centered on the county-scaled `donation_ratio` branch, not the retired formal project pipeline.
