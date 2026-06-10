# Gate 5 - Interpretability

## Purpose

Explain the model behavior with both linear-model coefficients and SHAP-based tree explanations.

## Current Main Implementation

- Entrypoint: `run_all.py`
- Main code paths:
  - `src/model.ridge_coefficients()`
  - `src/shap_analysis.run()`

## Current Interpretability Flow

1. Export standardized Ridge coefficients from the trained Ridge pipeline.
2. Build a tree-compatible design matrix for the XGBoost explanation path.
3. Fit an XGBoost regressor on the formal train split.
4. Compute SHAP values on the formal test split.
5. Save tabular feature importance and visual explanation artifacts.

## Current Main Output Artifacts

- `model_outputs/metrics/ridge_coefficients.csv`
- `model_outputs/shap/shap_importance.csv`
- `model_outputs/shap/shap_summary.png`
- `model_outputs/shap/shap_importance.png`
- `model_outputs/gate5_verdict.json`

## Interpretation Scope

- Ridge coefficients explain the direction and magnitude of standardized linear effects.
- SHAP explains nonlinear feature impact for the XGBoost path.
- The final report and dashboard consume both views to support human review.

## Current Status

- Implemented and active.
- This gate currently does not include per-row waterfall plots in the main project pipeline.
