# Source Module Guide

## Formal Mainline

- `run_all.py` orchestrates the formal project pipeline.
- `train_donation_count_xgboost_ensemble.py` trains the Gate 7 county-month `donation_count` ensemble with monthly walk-forward validation.
- `analyze_donation_count_xgboost_ensemble.py` generates SHAP outputs, model comparison charts, and Gate 7 acceptance evidence.
- `src/final_delivery.py` writes Gate 0 through Gate 7 verdicts, the final report, dashboard, tour guide, and formal model comparison chart from Gate 7 artifacts.

## Legacy or Exploratory Modules

- `train_donation_count_raw_xgboost.py` and `analyze_donation_count_raw_xgboost.py` are retained as diagnostic history, not the formal mainline.
- `src/features.py`, `src/model.py`, and `src/shap_analysis.py` are retained for the older `donation_ratio` comparison path.
- `src/clustering.py` is retained for exploratory county-profile clustering.
- `group_xgboost_models.py`, `filtered_xgboost_validation.py`, `xgboost_root_cause_analysis.py`, and `train_donation_count_raw_xgboost_enhanced.py` are experiments or diagnostics, not the formal Gate 0-7 path.
