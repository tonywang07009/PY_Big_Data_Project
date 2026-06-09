"""
Interpret and audit the formal donation_count raw-row XGBoost pipeline.

Outputs:
  - SHAP importance table and plots
  - county-level actual vs predicted donation_count comparison on validation rows
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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

from train_donation_count_raw_xgboost import (
    OUTDIR,
    TARGET,
    build_model,
    build_raw_row_frame,
    encode_features,
    split_frames,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CSV_COLS,
)

COUNTY_NAMES = {
    0: "南投縣",
    1: "台中市",
    2: "台北市",
    3: "台南市",
    4: "台東縣",
    5: "嘉義市",
    6: "嘉義縣",
    7: "基隆市",
    8: "宜蘭縣",
    9: "屏東縣",
    10: "彰化縣",
    11: "新北市",
    12: "新竹市",
    13: "新竹縣",
    14: "桃園市",
    15: "花蓮縣",
    16: "苗栗縣",
    17: "雲林縣",
    18: "高雄市",
}


def load_or_rebuild() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], object]:
    frame = build_raw_row_frame()
    train_raw, valid_raw = split_frames(frame)
    train = train_raw.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]).copy()
    valid = valid_raw.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]).copy()
    train_x, valid_x, model_columns = encode_features(train, valid)

    model = build_model()
    model_path = OUTDIR / "xgboost_model.json"
    if model_path.exists():
        model.load_model(str(model_path))
    else:
        model.fit(train_x, train[TARGET].to_numpy())
    return train, valid, train_x, valid_x, model_columns, model


def write_shap_outputs(valid: pd.DataFrame, valid_x: pd.DataFrame, model_columns: list[str], model) -> dict:
    sample_n = min(5000, len(valid_x))
    sample_pos = valid_x.sample(n=sample_n, random_state=42).index.to_numpy()
    shap_x = valid_x.iloc[sample_pos].reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_x)

    imp = (
        pd.DataFrame({"feature": model_columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(OUTDIR / "shap_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="tab:green")
    plt.title("donation_count raw-row SHAP importance")
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(OUTDIR / "shap_importance.png", dpi=130)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, shap_x, feature_names=model_columns, show=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    summary = {
        "target": TARGET,
        "shap_sample_rows": sample_n,
        "top_features": imp.head(10).to_dict(orient="records"),
    }
    (OUTDIR / "shap_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_county_comparison(valid: pd.DataFrame, valid_x: pd.DataFrame, model_columns: list[str], model) -> dict:
    preds = model.predict(valid_x[model_columns])
    scored = valid[CSV_COLS].copy()
    scored["predicted_donation_count"] = preds
    scored["county_name"] = scored["county_label"].map(COUNTY_NAMES)
    scored["count_diff"] = scored["predicted_donation_count"] - scored[TARGET]
    scored["abs_count_diff"] = np.abs(scored["count_diff"])

    county = (
        scored.groupby(["county_label", "county_name"], sort=True)
        .agg(
            actual_donation_count=(TARGET, "sum"),
            predicted_donation_count=("predicted_donation_count", "sum"),
            mean_abs_row_error=("abs_count_diff", "mean"),
            validation_rows=(TARGET, "size"),
        )
        .reset_index()
    )
    county["count_diff"] = county["predicted_donation_count"] - county["actual_donation_count"]
    county["abs_count_diff"] = county["count_diff"].abs()
    county["pct_diff"] = county["count_diff"] / np.maximum(county["actual_donation_count"], 1e-6) * 100.0
    county["abs_pct_diff"] = county["pct_diff"].abs()
    county = county.sort_values("abs_count_diff", ascending=False).reset_index(drop=True)
    county.to_csv(OUTDIR / "county_validation_comparison.csv", index=False)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(county))
    width = 0.4
    plt.bar(x - width / 2, county["actual_donation_count"], width=width, label="Actual")
    plt.bar(x + width / 2, county["predicted_donation_count"], width=width, label="Predicted")
    plt.xticks(x, county["county_name"], rotation=35, ha="right")
    plt.ylabel("donation_count sum on validation rows")
    plt.title("County-level actual vs predicted donation_count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "county_validation_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()

    summary = {
        "largest_abs_count_diff_county": county.iloc[0][["county_name", "abs_count_diff", "pct_diff"]].to_dict(),
        "largest_abs_pct_diff_county": county.sort_values("abs_pct_diff", ascending=False).iloc[0][["county_name", "abs_pct_diff", "count_diff"]].to_dict(),
    }
    (OUTDIR / "county_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    valid, valid_x = None, None
    train, valid, train_x, valid_x, model_columns, model = load_or_rebuild()
    shap_summary = write_shap_outputs(valid, valid_x, model_columns, model)
    county_summary = write_county_comparison(valid, valid_x, model_columns, model)

    print("Top SHAP features:")
    for row in shap_summary["top_features"][:8]:
        print(f"- {row['feature']}: {row['mean_abs_shap']:.4f}")
    print("County comparison summary:")
    print(json.dumps(county_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
