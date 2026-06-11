# QFD Matrix

Use this matrix to connect the research proposal and Gate 7 formal project direction to engineering work. Gate details are routed to `agent_doc/gates/`.

| Research Need | Engineering Contract | Gate | Evidence |
|---|---|---:|---|
| Trustworthy source data | Read `encoded_ml_dataset.csv` and use all available county labels | 0 | `model_outputs/gate0_verdict.json` |
| Correct modeling grain | Aggregate raw rows into one county-month row | 1 | `model_outputs/donation_count_xgboost_ensemble/county_month_matrix.csv` |
| Leakage prevention | Use monthly expanding Walk-Forward Analysis with train-only scalers | 1 | `monthly_walk_forward_summary.csv` |
| County heterogeneity | Standardize numeric features per county and encode `county_label` | 2 | `county_numeric_scalers.json` |
| Time-aware history | Build date-safe `donation_count` lag and rolling features | 2 | fold `metrics.json` |
| Nonlinear prediction | Train XGBoost on `log1p(donation_count)` | 3 | `xgboost_model.json` |
| County-level calibration | Fit county calibration using train-fold months only | 3 | `calibration_params.json` |
| Model comparison | Compare log-XGBoost, calibrated XGBoost, historical baseline, and final ensemble | 4 | `model_comparison.csv`, `model_comparison.png` |
| Interpretability | Generate SHAP and feature importance for selected root fold | 5 | `shap_importance.csv` |
| County-level auditability | Export county-month and county validation comparisons | 5 | `county_validation_comparison.csv` |
| Final delivery | Generate gate verdicts, final report, dashboard, tour guide, and chart | 6 | `reports/` |
| Formal ensemble acceptance | Validate Gate 7 thresholds for sMAPE and county error | 7 | `model_outputs/gate7_verdict.json` |
| Human traceability | Record decisions, gate criteria, reports, and agent-readable routing | all | `agent_doc/`, `reports/` |
