# QFD Matrix

Use this matrix to connect the research proposal to engineering work. Gate details are routed to `agent_doc/gates/`.

| Research Need | Engineering Contract | Gate | Evidence |
|---|---|---:|---|
| Trustworthy source data | Discover files, profile schema, detect nulls, dtypes, outliers, and semantic issues before modeling | 0 | `model_outputs/gate0_verdict.json` |
| Mixed-frequency alignment | Align monthly invoice data with annual income and housing indicators | 1 | `model_outputs/gate1_verdict.json` |
| Seasonal noise control | Apply STL to monthly high-frequency features only | 1 | `model_outputs/preprocessing_summary.json` |
| County heterogeneity | Encode county identity and build K-Means cluster features | 2 | `model_outputs/cluster_profile.json` |
| Baseline accountability | Compare linear regression, Ridge, and Lasso before nonlinear models | 3 | `model_outputs/metrics/baseline_metrics.json` |
| Nonlinear prediction | Compare XGBoost and Random Forest | 3 | `model_outputs/metrics/model_comparison.json` |
| Leakage prevention | Use time-aware validation and never use shuffled K-Fold | 4 | `model_outputs/gate4_verdict.json` |
| Interpretability | Generate SHAP values and plots for the selected model | 5 | `model_outputs/shap/shap_values.csv` |
| Actionable output | Compute priority scores for final recommendations | 6 | `model_outputs/priority_scores.csv` |
| Human traceability | Record decisions, gate criteria, reports, and agent memory | all | `agent_doc/`, `reports/` |

