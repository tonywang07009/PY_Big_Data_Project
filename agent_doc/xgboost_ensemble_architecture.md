# XGBoost Ensemble Architecture

## Purpose

This is the formal Gate 7 MecDonate architecture. It predicts county-month `donation_count` while controlling county-level scale bias through a log-target XGBoost model, county historical baseline, and county calibration layer.

## Main Design

- **Target**: `donation_count`
- **Scope**: all available `county_label` values in `data/raw/encoded_ml_dataset.csv`
- **Modeling grain**: one row per county-month
- **Validation**: monthly expanding Walk-Forward Analysis
- **Fold range**: `2023-01` through `2024-12`
- **No leakage rule**: each fold uses only months earlier than the validation month for training, scaling, calibration, and model fitting

## Component Models

### Model A - Log-Target XGBoost

- Train target: `log1p(donation_count)`
- Prediction transform: `expm1(raw_model_output)`
- Goal: reduce scale domination from high-volume counties.

### Model B - County Historical Baseline

- Preferred prediction: `donation_count_roll3_mean`
- Fallback order: `donation_count_roll3_mean` -> `donation_count_lag1` -> train-fold county median -> train-fold global median
- Goal: preserve each county's historical donation-count level.

### Model C - County Calibration

- Calibration shape: `calibrated = a_county * xgb_prediction + b_county`
- Calibration source: recent train-fold months only
- Fallback: global calibration when a county has insufficient calibration samples
- Goal: correct persistent county-level high/low bias.

### Final Ensemble

- Formula: `final_prediction = 0.6 * calibrated_xgb + 0.4 * historical_baseline`
- Final predictions are clipped at zero.
- Every output row keeps component predictions for auditability.

## Gate Mapping

- Gate 0: source data readiness
- Gate 1: county-month preprocessing and monthly split construction
- Gate 2: feature engineering
- Gate 3: formal model build
- Gate 4: monthly walk-forward validation and model comparison
- Gate 5: SHAP interpretability and county audit
- Gate 6: final delivery
- Gate 7: formal ensemble acceptance

## Acceptance Focus

Gate 7 accepts the formal ensemble when:

- mean monthly ensemble sMAPE is at or below `20%`
- mean county absolute percent difference is at or below `10%`
- median county absolute percent difference is at or below `10%`
- SHAP, county validation, county-month validation, and model comparison artifacts exist
