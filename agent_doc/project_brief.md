# Project Brief

## Purpose

MecDonate predicts monthly invoice donation counts at the county level in Taiwan and audits county-level prediction bias for downstream donation-resource planning.

## Source Material

- Main proposal: `source_materials/group_six.md`
- PDF proposal: `source_materials/group_six.pdf`
- Existing transformed data utility: `source_materials/data_transformer.py`

## Formal Scope

- Counties: all `county_label` values available in `data/raw/encoded_ml_dataset.csv`
- Target: `donation_count`
- Modeling grain: county-month
- Formal model: Gate 7 XGBoost ensemble
- Validation: monthly expanding Walk-Forward Analysis
- Scaling: numeric features standardized per county using train-fold rows only

## Formal Model Contract

- Model A: global XGBoost trained on `log1p(donation_count)`
- Model B: county historical baseline from lag and rolling features
- Model C: county calibration layer fit from train-fold months only
- Final prediction: `0.6 * calibrated_xgb + 0.4 * historical_baseline`

## Non-Negotiable Constraints

- Do not shuffle time-series rows.
- Do not fit scalers, encoders, calibration, or models on validation rows.
- Do not modify files under `data/raw/`.
- Ask the human in Traditional Chinese when a high-impact ambiguity appears.
- Keep persisted project documents in English.

## Legacy Context

Older documents and scripts used `donation_ratio`, six-county filtering, group models, or raw-row XGBoost experiments. Those paths are retained only as legacy or diagnostic material and must not be presented as the formal project mainline.
