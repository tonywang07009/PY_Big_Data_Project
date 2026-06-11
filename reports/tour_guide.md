# MecDonate Tour Guide

## Formal Mainline

- Target: `donation_count`.
- Grain: county-month.
- Model A: `log1p(donation_count)` XGBoost.
- Model B: county historical baseline from lag and rolling features.
- Model C: train-only county calibration.
- Final prediction: `0.6 * calibrated_xgb + 0.4 * historical_baseline`.
- Validation: monthly expanding Walk-Forward Analysis.
- Gate 7 verdict: `PASS`.

## Gate Checklist

- Gate 0 `PASS`: Data readiness.
- Gate 1 `PASS`: Preprocessing and split construction.
- Gate 2 `PASS`: Feature engineering.
- Gate 3 `PASS`: Formal model build.
- Gate 4 `PASS`: Monthly walk-forward validation.
- Gate 5 `PASS`: Interpretability and audit.
- Gate 6 `PASS`: Final delivery.
- Gate 7 `PASS`: Formal ensemble acceptance.

## Continuation Marker

- Selected root fold: `fold_2024-12`.
- Primary output root: `model_outputs/donation_count_xgboost_ensemble/`.
- Model comparison chart: `reports/formal_model_comparison.png`.
- Acceptance evidence: `model_outputs/donation_count_xgboost_ensemble/acceptance_summary.json`.
