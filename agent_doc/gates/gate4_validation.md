# Gate 4 - Monthly Walk-Forward Validation

## Purpose

Validate the formal Gate 7 ensemble and its component models with monthly expanding Walk-Forward Analysis.

## Formal Micro-Tasks

- T4-A: evaluate monthly folds from `2023-01` through `2024-12`
- T4-B: summarize RMSE, MAE, sMAPE, epsilon-MAPE, and R2 for each component
- T4-C: compare log-XGBoost, calibrated XGBoost, historical baseline, and final ensemble
- T4-D: write aggregate validation summaries and comparison chart

## Required Evidence

- No shuffle is used.
- Each validation fold uses only one calendar month.
- Scalers, encoders, calibration, and models are fit from train-fold data only.
- Model comparison chart exists and is readable.

## Expected Outputs

- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward_summary.csv`
- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward_summary.json`
- `model_outputs/donation_count_xgboost_ensemble/model_comparison.csv`
- `model_outputs/donation_count_xgboost_ensemble/model_comparison.png`
- `reports/formal_model_comparison.png`
- `model_outputs/gate4_verdict.json`
