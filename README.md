# PY_Big_Data_Project

Formal MecDonate pipeline for county-level monthly invoice `donation_count` prediction.

The formal version is the Gate 7 ensemble:

- Model A: global XGBoost trained on `log1p(donation_count)`
- Model B: county historical baseline from lag and rolling features
- Model C: train-only county calibration layer
- Final prediction: `0.6 * calibrated_xgb + 0.4 * historical_baseline`
- Validation: monthly expanding Walk-Forward Analysis
- Scope: all available `county_label` values in `data/raw/encoded_ml_dataset.csv`

Run:

```bash
python run_all.py
```

Formal artifacts are written under:

- `model_outputs/donation_count_xgboost_ensemble/`
- `model_outputs/gate0_verdict.json` through `model_outputs/gate7_verdict.json`
- `reports/final_project_report.md`
- `reports/dashboard.html`
- `reports/formal_model_comparison.png`

Legacy raw-row and ratio-model scripts are retained only for historical diagnosis. They are not the formal project mainline.
