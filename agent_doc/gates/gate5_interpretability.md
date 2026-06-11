# Gate 5 - Interpretability

## Purpose

Explain the current county-scaled `donation_ratio` models with feature-importance outputs that are directly tied to the generated prediction artifacts.

## Current Implementation

### XGBoost feature importance

- Script:
  - `train_donation_ratio_county_scaled_xgboost.py`
- Outputs:
  - `model_outputs/donation_ratio_county_scaled_xgboost/feature_importance.csv`
  - `model_outputs/donation_ratio_county_scaled_xgboost/feature_importance.png`

## Current Interpretation Flow

1. Train XGBoost on the county-scaled train set.
2. Export model-native feature importance scores.
3. Save the ranked CSV.
4. Render a horizontal bar chart from the feature-importance table.

## Current Interpretation Scope

- This gate currently explains the XGBoost branch through feature importance.
- It does not currently compute SHAP values.
- It does not currently generate coefficient reports for the linear-model comparison branch.

## Current Status

- Implemented and active for the XGBoost branch.
- Interpretability is lighter-weight than the older SHAP-based main project flow.
