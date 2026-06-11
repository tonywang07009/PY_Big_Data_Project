# Gate 7 - Formal Ensemble Acceptance

## Purpose

Accept or reject the formal Gate 7 county-month XGBoost ensemble using direct performance thresholds and required artifact checks.

## Formal Micro-Tasks

- T7-A: load ensemble evidence from `model_outputs/donation_count_xgboost_ensemble`
- T7-B: validate monthly error metrics
- T7-C: validate county-level absolute percent difference metrics
- T7-D: validate SHAP, county audit, county-month audit, and model comparison artifacts
- T7-E: emit `PASS`, `CONDITIONAL`, or `FAIL`

## Required Evidence

- Ensemble monthly walk-forward summary exists.
- Ensemble county validation comparison exists.
- Ensemble county-month validation comparison exists.
- SHAP outputs exist.
- Model comparison chart exists.
- Gate 7 verdict records thresholds and actual values.

## Acceptance Rule

Gate 7 is `PASS` when:

- mean monthly ensemble sMAPE is at or below `20%`
- mean county absolute percent difference is at or below `10%`
- median county absolute percent difference is at or below `10%`
- required artifacts are present and non-empty

Gate 7 is `CONDITIONAL` when the overall monthly error passes but one county-level threshold requires human review.

Gate 7 is `FAIL` when the ensemble misses the monthly error threshold or lacks required formal artifacts.

## Expected Outputs

- `model_outputs/gate7_verdict.json`
- `model_outputs/donation_count_xgboost_ensemble/acceptance_summary.json`
- `model_outputs/donation_count_xgboost_ensemble/model_comparison.png`
- `reports/final_project_report.md`
- `reports/dashboard.html`
