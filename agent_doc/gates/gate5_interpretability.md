# Gate 5 - Interpretability and Audit

## Purpose

Explain the selected formal Gate 7 XGBoost component and produce county-level validation audits for the final ensemble.

## Formal Micro-Tasks

- T5-A: generate SHAP importance for the selected root fold
- T5-B: export SHAP plots and top-feature JSON
- T5-C: aggregate final ensemble predictions to county-month and county levels
- T5-D: identify largest absolute and percent county errors

## Required Evidence

- SHAP feature importance table is non-empty.
- County comparison table is non-empty.
- County-month comparison table is non-empty.
- Component predictions are retained for auditability.

## Expected Outputs

- `model_outputs/donation_count_xgboost_ensemble/shap_importance.csv`
- `model_outputs/donation_count_xgboost_ensemble/shap_summary.json`
- `model_outputs/donation_count_xgboost_ensemble/county_validation_comparison.csv`
- `model_outputs/donation_count_xgboost_ensemble/county_month_validation_comparison.csv`
- `model_outputs/gate5_verdict.json`
