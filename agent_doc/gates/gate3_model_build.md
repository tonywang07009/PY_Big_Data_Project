# Gate 3 - Formal Model Build

## Purpose

Train the formal Gate 7 ensemble components for each monthly walk-forward fold and preserve selected root artifacts.

## Formal Micro-Tasks

- T3-A: train global XGBoost on `log1p(donation_count)`
- T3-B: build county historical baseline predictions
- T3-C: fit county calibration parameters from train-fold months only
- T3-D: blend `0.6 * calibrated_xgb + 0.4 * historical_baseline`
- T3-E: export fold-level model, calibration, feature importance, and component prediction files

## Required Evidence

- `xgboost_model.json` exists for each fold.
- `calibration_params.json` exists for each fold.
- Component predictions are retained for auditability.
- Root `metrics.json` identifies the selected root fold.

## Expected Outputs

- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward/fold_*/xgboost_model.json`
- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward/fold_*/calibration_params.json`
- `model_outputs/donation_count_xgboost_ensemble/component_predictions.csv`
- `model_outputs/gate3_verdict.json`
