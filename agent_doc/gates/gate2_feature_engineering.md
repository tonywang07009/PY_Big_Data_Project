# Gate 2 - Feature Engineering

## Purpose

Add the modeling features required by the current county-scaled `donation_ratio` experiments without changing the train/test split or the county-scaling outputs.

## Current Implementation

- Main code paths:
  - `train_donation_ratio_county_scaled_xgboost.py`
  - `compare_donation_ratio_county_scaled_models.py`
  - `train_donation_ratio_county_scaled_year_cv.py`

## Current Feature Logic

### Base inputs from preprocessing

- `industry_invoice_count`
- `industry_invoice_amount`
- `housing_burden`
- `price_income_ratio`
- `total_county_invoice_count`
- `total_county_invoice_amount`
- `donation_count`
- `industry_label`
- `carrier_type_label`
- `county_label`
- `year`

### Added time features

- `month_num`
- `month_sin`
- `month_cos`

### Modeling feature groups

- Numeric features:
  - all numeric columns except `donation_ratio` and categorical label columns
- Categorical features:
  - `county_label`
  - `industry_label`
  - `carrier_type_label`

## Current Processing Role

- `train_donation_ratio_county_scaled_xgboost.py` and `compare_donation_ratio_county_scaled_models.py`
  - read the county-scaled combined CSVs
  - add cyclic month features
  - one-hot encode categorical labels during model preparation
- `train_donation_ratio_county_scaled_year_cv.py`
  - rebuild the same feature contract inside each rolling fold
  - recompute county-level scaling from raw data per fold before encoding

## Current Status

- Implemented and active.
- Clustering-oriented feature engineering is now legacy relative to the current results path.
