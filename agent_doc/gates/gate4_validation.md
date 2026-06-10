# Gate 4 - Validation

## Purpose

Validate that the model evaluation policy is time-aware, leakage-resistant, and explicit about which years belong to training and which belong to future evaluation.

## Current Main Implementation

- Entrypoint: `run_all.py`
- Main code path:
  - `src/model.split()`
  - gate evidence assembled by `src/final_delivery.write_gate_verdicts()`

## Current Formal Validation Policy

- No shuffling.
- Train on `2019-2022`.
- Test on `2023-2024`.
- Use RMSE and R2 as the primary formal comparison metrics.

## Current Main Output Artifacts

- `model_outputs/gate4_verdict.json`
- evaluation evidence embedded in:
  - `model_outputs/metrics/model_comparison.csv`
  - `model_outputs/metrics/model_results.json`
  - `reports/final_project_report.md`
  - `reports/tour_guide.md`

## Supplementary Validation Workflow Added In This Chat

- Script: `train_donation_ratio_county_scaled_year_cv.py`
- Purpose:
  - perform rolling year validation for the county-scaled `donation_ratio` workflow
  - recompute county-level scaling separately inside each fold
  - avoid future leakage by fitting each county scaler only on fold-train years
- Validation folds currently produced:
  - `2020`
  - `2022`
  - `2023`
  - `2024`
- Output root:
  - `model_outputs/donation_ratio_county_scaled_year_cv/`

## Current Supplementary Output Artifacts

- `model_outputs/donation_ratio_county_scaled_year_cv/metrics_by_fold.csv`
- `model_outputs/donation_ratio_county_scaled_year_cv/metrics_summary.json`
- `model_outputs/donation_ratio_county_scaled_year_cv/validation_predictions_by_fold.csv`
- `model_outputs/donation_ratio_county_scaled_year_cv/performance_result.png`

## Current Status

- The formal project pipeline uses one fixed future holdout.
- The rolling year CV workflow now exists as an additional validation branch for the county-scaled `donation_ratio` experiment.
