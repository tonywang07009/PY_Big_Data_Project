"""
Analyze the MecDonate XGBoost ensemble outputs.

This script:
  - generates SHAP importance for the selected latest monthly fold
  - validates the formal ensemble against Gate 7 acceptance thresholds
  - writes acceptance evidence for Gate 7
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plotting_setup import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train_donation_count_xgboost_ensemble import (
    FOLD_DIR,
    NUMERIC_FEATURES,
    OUTDIR,
    TARGET,
    build_county_month_frame,
    build_model,
    clean_frame,
    encode_features,
    monthly_walk_forward_splits,
    write_json,
)

MAX_MEAN_MONTHLY_SMAPE = 20.0
MAX_MEAN_COUNTY_ABS_PCT_DIFF = 10.0
MAX_MEDIAN_COUNTY_ABS_PCT_DIFF = 10.0


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_shap_outputs() -> dict:
    metrics = read_json(OUTDIR / "metrics.json")
    selected_fold = metrics["selected_root_fold"]
    cm = build_county_month_frame()
    fold_specs = {item["fold"]: item for item in monthly_walk_forward_splits(cm)}
    if selected_fold not in fold_specs:
        raise KeyError(f"Selected fold {selected_fold} was not found in rebuilt fold specs.")

    fold_spec = fold_specs[selected_fold]
    train = clean_frame(fold_spec["train"])
    valid = clean_frame(fold_spec["valid"])
    _train_x, valid_x, model_columns, _ = encode_features(train, valid)

    model = build_model()
    model.load_model(str(FOLD_DIR / selected_fold / "xgboost_model.json"))
    shap_x = valid_x[model_columns].reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_x)

    imp = (
        pd.DataFrame({"feature": model_columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(OUTDIR / "shap_importance.csv", index=False)
    imp.to_csv(FOLD_DIR / selected_fold / "shap_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    top = imp.head(20).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"], color="tab:green")
    plt.title("Ensemble log-XGBoost SHAP importance")
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(OUTDIR / "shap_importance.png", dpi=130)
    plt.savefig(FOLD_DIR / selected_fold / "shap_importance.png", dpi=130)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, shap_x, feature_names=model_columns, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUTDIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.savefig(FOLD_DIR / selected_fold / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    summary = {
        "target": TARGET,
        "selected_fold": selected_fold,
        "shap_rows": len(shap_x),
        "numeric_features": NUMERIC_FEATURES,
        "top_features": imp.head(15).to_dict(orient="records"),
    }
    write_json(OUTDIR / "shap_summary.json", summary)
    write_json(FOLD_DIR / selected_fold / "shap_summary.json", summary)
    return summary


def write_model_comparison_plot() -> list[dict]:
    monthly = pd.read_csv(OUTDIR / "monthly_walk_forward_summary.csv")
    components = [
        ("Log-XGBoost", "xgb_log"),
        ("Calibrated XGBoost", "calibrated_xgb"),
        ("Historical Baseline", "historical_baseline"),
        ("Final Ensemble", "ensemble"),
    ]
    rows = []
    for label, prefix in components:
        rows.append(
            {
                "model": label,
                "rmse": float(monthly[f"{prefix}_rmse"].mean()),
                "mae": float(monthly[f"{prefix}_mae"].mean()),
                "smape": float(monthly[f"{prefix}_smape"].mean()),
                "r2": float(monthly[f"{prefix}_r2"].mean()),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison_sorted = comparison.sort_values("r2", ascending=False).reset_index(drop=True)
    comparison_sorted.to_csv(OUTDIR / "model_comparison.csv", index=False)

    by_r2 = comparison.sort_values("r2", ascending=True)
    by_rmse = comparison.sort_values("rmse", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].barh(by_r2["model"], by_r2["r2"], color="tab:blue")
    axes[0].set_title("Monthly Walk-Forward R2 by model")
    axes[0].set_xlabel("Mean R2")
    axes[1].barh(by_rmse["model"], by_rmse["rmse"], color="tab:orange")
    axes[1].set_title("Monthly Walk-Forward RMSE by model")
    axes[1].set_xlabel("Mean RMSE")
    fig.tight_layout()
    fig.savefig(OUTDIR / "model_comparison.png", dpi=140)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(reports_dir / "formal_model_comparison.png", dpi=140)
    plt.close(fig)
    return comparison_sorted.to_dict(orient="records")


def accept_formal_ensemble(shap_summary: dict) -> dict:
    ensemble_metrics = read_json(OUTDIR / "metrics.json")
    ensemble_county = pd.read_csv(OUTDIR / "county_validation_comparison.csv")
    ensemble_county_month = pd.read_csv(OUTDIR / "county_month_validation_comparison.csv")

    ensemble_mean_county = float(ensemble_county["abs_pct_diff"].mean())
    ensemble_median_county = float(ensemble_county["abs_pct_diff"].median())
    ensemble_smape = float(ensemble_metrics["mean_monthly_ensemble_smape"])
    comparison_csv_exists = (OUTDIR / "model_comparison.csv").exists()
    comparison_png_exists = (OUTDIR / "model_comparison.png").exists()
    comparison_report_png_exists = (ROOT / "reports" / "formal_model_comparison.png").exists()

    smape_pass = ensemble_smape <= MAX_MEAN_MONTHLY_SMAPE
    county_mean_pass = ensemble_mean_county <= MAX_MEAN_COUNTY_ABS_PCT_DIFF
    county_median_pass = ensemble_median_county <= MAX_MEDIAN_COUNTY_ABS_PCT_DIFF
    artifact_pass = (
        not ensemble_county.empty
        and not ensemble_county_month.empty
        and shap_summary.get("shap_rows", 0) > 0
        and len(shap_summary.get("top_features", [])) > 0
        and comparison_csv_exists
        and comparison_png_exists
        and comparison_report_png_exists
    )

    if smape_pass and county_mean_pass and county_median_pass and artifact_pass:
        verdict = "PASS"
    elif smape_pass and (county_mean_pass or county_median_pass) and artifact_pass:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"

    payload = {
        "gate": 7,
        "name": "Formal ensemble acceptance",
        "verdict": verdict,
        "ensemble": {
            "pipeline": ensemble_metrics.get("pipeline"),
            "validation": ensemble_metrics.get("validation"),
            "mean_monthly_smape": ensemble_smape,
            "mean_county_abs_pct_diff": ensemble_mean_county,
            "median_county_abs_pct_diff": ensemble_median_county,
            "selected_root_fold": ensemble_metrics.get("selected_root_fold"),
        },
        "checks": [
            {
                "name": "mean monthly sMAPE threshold",
                "status": "PASS" if smape_pass else "FAIL",
                "threshold": MAX_MEAN_MONTHLY_SMAPE,
                "actual": ensemble_smape,
            },
            {
                "name": "mean county absolute percent difference threshold",
                "status": "PASS" if county_mean_pass else "FAIL",
                "threshold": MAX_MEAN_COUNTY_ABS_PCT_DIFF,
                "actual": ensemble_mean_county,
            },
            {
                "name": "median county absolute percent difference threshold",
                "status": "PASS" if county_median_pass else "FAIL",
                "threshold": MAX_MEDIAN_COUNTY_ABS_PCT_DIFF,
                "actual": ensemble_median_county,
            },
            {
                "name": "formal artifact completeness",
                "status": "PASS" if artifact_pass else "FAIL",
                "artifacts": [
                    "model_outputs/donation_count_xgboost_ensemble/county_validation_comparison.csv",
                    "model_outputs/donation_count_xgboost_ensemble/county_month_validation_comparison.csv",
                    "model_outputs/donation_count_xgboost_ensemble/shap_importance.csv",
                    "model_outputs/donation_count_xgboost_ensemble/shap_summary.json",
                    "model_outputs/donation_count_xgboost_ensemble/model_comparison.csv",
                    "model_outputs/donation_count_xgboost_ensemble/model_comparison.png",
                    "reports/formal_model_comparison.png",
                ],
            },
        ],
        "thresholds": {
            "max_mean_monthly_smape": MAX_MEAN_MONTHLY_SMAPE,
            "max_mean_county_abs_pct_diff": MAX_MEAN_COUNTY_ABS_PCT_DIFF,
            "max_median_county_abs_pct_diff": MAX_MEDIAN_COUNTY_ABS_PCT_DIFF,
        },
        "shap_summary": shap_summary,
    }
    write_json(OUTDIR / "acceptance_summary.json", payload)
    write_json(ROOT / "model_outputs" / "gate7_verdict.json", payload)
    return payload


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shap_summary = write_shap_outputs()
    write_model_comparison_plot()
    acceptance = accept_formal_ensemble(shap_summary)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
