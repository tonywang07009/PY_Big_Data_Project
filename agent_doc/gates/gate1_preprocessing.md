# Gate 1 - Preprocessing

## Purpose

Transform the encoded raw invoice rows into stable, time-aware modeling inputs without crossing the dataset’s coverage gap or leaking future information.

## Current Main Implementation

- Entrypoint: `run_all.py`
- Main code path:
  - `src/features.build_county_month()`
  - `src/project_config.filter_scope_counties()`
- Primary output:
  - `model_outputs/county_month_matrix.csv`

## Current Processing Flow

1. Read `data/raw/encoded_ml_dataset.csv`.
2. Group raw rows to county-month level.
3. Preserve county-month constant indicators such as:
   - `housing_burden`
   - `price_income_ratio`
   - `total_county_invoice_count`
   - `total_county_invoice_amount`
   - `donation_count`
   - `donation_ratio`
4. Build structural features from row-level invoice behavior:
   - carrier usage ratio
   - active industry count
   - industry HHI
5. Add time fields and cyclic month features.
6. Fit STL per county for seasonal/trend components.
7. Join lag features by calendar date so lags do not silently cross the 2020 to 2022 coverage gap.
8. Filter to formal in-scope counties before downstream clustering and modeling.

## Supplementary Workflow Added In This Chat

- Standalone script: `split_scale_by_county.py`
- Purpose:
  - derive `year` from `month`
  - split rows into `year < 2024` train and `year >= 2024` test
  - fit one `StandardScaler` per `county_label`
  - save scaled combined datasets plus per-county CSVs and scaler files
- Output root:
  - `data/county_zscore_split/`

This supplementary workflow is not a dependency of `run_all.py`, but it is now part of the repository’s preprocessing toolbox.

## Current Output Artifacts

- Main pipeline:
  - `model_outputs/county_month_matrix.csv`
- Supplementary county-scaling workflow:
  - `data/county_zscore_split/train_scaled_all_counties.csv`
  - `data/county_zscore_split/test_scaled_all_counties.csv`
  - `data/county_zscore_split/by_county/`
  - `data/county_zscore_split/scalers/`

## Current Status

- Implemented and active.
- The main pipeline and the county-scaling branch are intentionally separate.
