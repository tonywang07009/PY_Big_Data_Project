# Gate 3 - Model Build

## Purpose

Train the current `donation_ratio` models on the county-scaled train/test datasets and compare multiple regressors on the exact same data split.

## Current Implementations

### Single-model XGBoost path

- Script:
  - `train_donation_ratio_county_scaled_xgboost.py`
- Input:
  - `data/county_zscore_split/train_scaled_all_counties.csv`
  - `data/county_zscore_split/test_scaled_all_counties.csv`
- Target:
  - `donation_ratio`
- Output root:
  - `model_outputs/donation_ratio_county_scaled_xgboost/`

### Multi-model comparison path

- Script:
  - `compare_donation_ratio_county_scaled_models.py`
- Same input data and target as above
- Output root:
  - `model_outputs/donation_ratio_county_scaled_model_comparison/`

## Current Model Set

- Linear Regression
- Ridge
- Lasso
- Random Forest
- XGBoost

## Current Build Flow

1. Load the already-scaled train/test CSVs.
2. Add `month_num`, `month_sin`, and `month_cos`.
3. Separate numeric and categorical feature groups.
4. Use:
   - standardized numeric + one-hot categorical preprocessing for linear models
   - passthrough numeric + one-hot categorical preprocessing for tree models
5. Fit each model on the same train set.
6. Predict on the same test set.
7. Rank models by validation R2.

## Current Output Artifacts

- Single-model path:
  - `model_outputs/donation_ratio_county_scaled_xgboost/metrics.json`
  - `model_outputs/donation_ratio_county_scaled_xgboost/validation_predictions.csv`
  - `model_outputs/donation_ratio_county_scaled_xgboost/xgboost_model.json`
- Multi-model path:
  - `model_outputs/donation_ratio_county_scaled_model_comparison/model_comparison.csv`
  - `model_outputs/donation_ratio_county_scaled_model_comparison/metrics.json`
  - `model_outputs/donation_ratio_county_scaled_model_comparison/validation_predictions.csv`

## Current Status

- Implemented and active.
- The current best-performing model on this exact split is now determined by the comparison script, not by `run_all.py`.
