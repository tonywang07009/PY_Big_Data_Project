# Gate 0 - Data Readiness

## Purpose

Confirm that the raw encoded dataset and the generated county-scaled datasets are readable before any preprocessing or model execution starts.

## Current Implementation

- Raw source:
  - `data/raw/encoded_ml_dataset.csv`
- Derived inputs checked by downstream scripts:
  - `data/county_zscore_split/train_scaled_all_counties.csv`
  - `data/county_zscore_split/test_scaled_all_counties.csv`
- Main code paths that depend on this gate:
  - `split_scale_by_county.py`
  - `train_donation_ratio_county_scaled_xgboost.py`
  - `train_donation_ratio_county_scaled_year_cv.py`
  - `compare_donation_ratio_county_scaled_models.py`

## What This Gate Checks

- The raw encoded CSV can be read.
- `month` and `county_label` exist in the raw data.
- The county-scaled train/test CSVs exist after preprocessing.
- The current train/test split is year-based:
  - train: `year < 2024`
  - test: `year >= 2024`

## Current Output Evidence

- `data/county_zscore_split/train_scaled_all_counties.csv`
- `data/county_zscore_split/test_scaled_all_counties.csv`
- `data/county_zscore_split/by_county/`
- `data/county_zscore_split/scalers/`

## Current Status

- Implemented and active through the county-scaled `donation_ratio` workflow.
- No separate `gate0_verdict.json` generator exists in the current experimental path.
