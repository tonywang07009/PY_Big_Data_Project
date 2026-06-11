# Gate 6 - Final Delivery

## Purpose

Package the current county-scaled `donation_ratio` results into stable machine-readable outputs and reviewable figures for comparison and inspection.

## Current Delivery Scope

The active delivery scope is no longer the old report/dashboard stack. The current deliverables are the generated model result folders under `model_outputs/`.

## Current Delivery Bundles

### County-scaled XGBoost bundle

- `model_outputs/donation_ratio_county_scaled_xgboost/metrics.json`
- `model_outputs/donation_ratio_county_scaled_xgboost/validation_predictions.csv`
- `model_outputs/donation_ratio_county_scaled_xgboost/performance_result.png`
- `model_outputs/donation_ratio_county_scaled_xgboost/feature_importance.csv`
- `model_outputs/donation_ratio_county_scaled_xgboost/feature_importance.png`
- `model_outputs/donation_ratio_county_scaled_xgboost/xgboost_model.json`

### County-scaled multi-model comparison bundle

- `model_outputs/donation_ratio_county_scaled_model_comparison/model_comparison.csv`
- `model_outputs/donation_ratio_county_scaled_model_comparison/model_comparison.png`
- `model_outputs/donation_ratio_county_scaled_model_comparison/prediction_scatter.png`
- `model_outputs/donation_ratio_county_scaled_model_comparison/validation_predictions.csv`
- `model_outputs/donation_ratio_county_scaled_model_comparison/metrics.json`

### County-scaled rolling year-CV bundle

- `model_outputs/donation_ratio_county_scaled_year_cv/metrics_by_fold.csv`
- `model_outputs/donation_ratio_county_scaled_year_cv/metrics_summary.json`
- `model_outputs/donation_ratio_county_scaled_year_cv/validation_predictions_by_fold.csv`
- `model_outputs/donation_ratio_county_scaled_year_cv/performance_result.png`

## Current Status

- Implemented and active.
- Current delivery is centered on reproducible experiment folders, not on a single final report/dashboard package.
