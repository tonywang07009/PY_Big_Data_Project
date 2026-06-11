# Gate 1 - Preprocessing

## Purpose

Prepare a county-level, year-aware train/test split and standardize numeric features per county without leaking test information into the training fit.

## Current Implementation

- Entrypoint:
  - `split_scale_by_county.py`
- Input:
  - `data/raw/encoded_ml_dataset.csv`
- Output root:
  - `data/county_zscore_split/`

## Current Processing Flow

1. Read the raw encoded dataset.
2. Derive `date` and `year` from the existing `month` column.
3. Split rows into:
   - train: `year < 2024`
   - test: `year >= 2024`
4. Group both splits by `county_label`.
5. Detect numeric columns to scale while excluding:
   - `month`
   - `date`
   - `year`
   - `county_label`
   - `industry_label`
   - `carrier_type_label`
   - `donation_ratio`
6. Fit one `StandardScaler` per county on the train group only.
7. Transform the matching test county with the already-fitted scaler.
8. Save combined datasets, per-county datasets, and per-county scaler files.

## Current Output Artifacts

- `data/county_zscore_split/train_scaled_all_counties.csv`
- `data/county_zscore_split/test_scaled_all_counties.csv`
- `data/county_zscore_split/by_county/county_<label>/train_scaled.csv`
- `data/county_zscore_split/by_county/county_<label>/test_scaled.csv`
- `data/county_zscore_split/scalers/county_<label>_standard_scaler.joblib`

## Current Status

- Implemented and active.
- This is the actual preprocessing entrypoint for the current repository results.
