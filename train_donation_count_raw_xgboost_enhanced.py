"""
Enhanced donation_count raw-row XGBoost workflow.

This keeps the original encoded raw-row grain, adds county identity,
county-month structural features, target lag/rolling features, compares
weighted and unweighted training, then writes model diagnostics.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

RAW = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
DATA_DIR = ROOT / "data" / "donation_count_raw_xgboost_enhanced"
OUTDIR = ROOT / "model_outputs" / "donation_count_raw_xgboost_enhanced"
BASELINE_OUTDIR = ROOT / "model_outputs" / "donation_count_raw_xgboost"
TARGET = "donation_count"
EPSILON = 1e-6

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

BASE_NUMERIC_FEATURES = [
    "housing_burden",
    "price_income_ratio",
    "total_county_invoice_count",
    "total_county_invoice_amount",
    "industry_invoice_count",
    "industry_invoice_amount",
    "year",
    "month_num",
    "month_sin",
    "month_cos",
]

DERIVED_ROW_FEATURES = [
    "industry_count_share",
    "industry_amount_share",
    "log_industry_invoice_count",
    "log_industry_invoice_amount",
    "log_total_county_invoice_count",
    "log_total_county_invoice_amount",
]

STRUCTURE_FEATURES = [
    "carrier_usage_ratio",
    "n_active_industries",
    "industry_hhi",
    "industry_entropy",
    "top1_industry_share",
    "top3_industry_share",
    "row_count_per_county_month",
]

TARGET_HISTORY_FEATURES = [
    "donation_count_lag1",
    "donation_count_lag2",
    "donation_count_lag3",
    "donation_count_roll3_mean",
    "donation_count_roll3_std",
    "donation_count_roll3_min",
    "donation_count_roll3_max",
]

NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES
    + DERIVED_ROW_FEATURES
    + STRUCTURE_FEATURES
    + TARGET_HISTORY_FEATURES
)
CATEGORICAL_FEATURES = ["county_label", "industry_label", "carrier_type_label"]
CSV_COLS = ["county_label", "month", TARGET] + NUMERIC_FEATURES + [
    "industry_label",
    "carrier_type_label",
]


def build_model(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        **params,
    )


def model_candidates() -> list[dict]:
    return [
        {
            "name": "baseline_like",
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 1,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "name": "conservative",
            "params": {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "max_depth": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "deeper",
            "params": {
                "n_estimators": 600,
                "learning_rate": 0.04,
                "max_depth": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_alpha": 0.0,
                "reg_lambda": 2.0,
            },
        },
        {
            "name": "regularized",
            "params": {
                "n_estimators": 900,
                "learning_rate": 0.025,
                "max_depth": 4,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "min_child_weight": 8,
                "reg_alpha": 0.5,
                "reg_lambda": 8.0,
            },
        },
    ]


def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["month"] + "-01")
    out["year"] = out["date"].dt.year
    out["month_num"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
    return out


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    count_denom = out["total_county_invoice_count"].replace(0, np.nan)
    amount_denom = out["total_county_invoice_amount"].replace(0, np.nan)
    out["industry_count_share"] = out["industry_invoice_count"] / count_denom
    out["industry_amount_share"] = out["industry_invoice_amount"] / amount_denom
    out["log_industry_invoice_count"] = np.log1p(out["industry_invoice_count"])
    out["log_industry_invoice_amount"] = np.log1p(out["industry_invoice_amount"])
    out["log_total_county_invoice_count"] = np.log1p(out["total_county_invoice_count"])
    out["log_total_county_invoice_amount"] = np.log1p(out["total_county_invoice_amount"])
    return out


def county_month_aggregate(sub: pd.DataFrame) -> pd.Series:
    total_count = sub["industry_invoice_count"].sum()
    carrier_count = sub.loc[sub["carrier_type_label"] == 0, "industry_invoice_count"].sum()
    carrier_usage_ratio = carrier_count / total_count if total_count else np.nan

    by_industry = sub.groupby("industry_label")["industry_invoice_count"].sum()
    if by_industry.sum() == 0:
        shares = pd.Series(dtype=float)
    else:
        shares = by_industry / by_industry.sum()

    sorted_shares = shares.sort_values(ascending=False)
    return pd.Series(
        {
            "carrier_usage_ratio": carrier_usage_ratio,
            "n_active_industries": int((by_industry > 0).sum()),
            "industry_hhi": float((shares**2).sum()) if len(shares) else np.nan,
            "industry_entropy": float(-(shares * np.log(shares + EPSILON)).sum())
            if len(shares)
            else np.nan,
            "top1_industry_share": float(sorted_shares.iloc[0]) if len(sorted_shares) else np.nan,
            "top3_industry_share": float(sorted_shares.head(3).sum()) if len(sorted_shares) else np.nan,
            "row_count_per_county_month": int(len(sub)),
        }
    )


def build_county_month_structure(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["county_label", "month"], sort=True)
        .apply(county_month_aggregate)
        .reset_index()
    )


def add_target_history(county_month: pd.DataFrame) -> pd.DataFrame:
    out = county_month.copy()
    key = out[["county_label", "date", TARGET]].copy()
    for lag in (1, 2, 3):
        shifted = key.copy()
        shifted["date"] = shifted["date"] + pd.DateOffset(months=lag)
        shifted = shifted.rename(columns={TARGET: f"{TARGET}_lag{lag}"})
        out = out.merge(
            shifted[["county_label", "date", f"{TARGET}_lag{lag}"]],
            on=["county_label", "date"],
            how="left",
        )

    lag_cols = [f"{TARGET}_lag{lag}" for lag in (1, 2, 3)]
    out["donation_count_roll3_mean"] = out[lag_cols].mean(axis=1)
    out["donation_count_roll3_std"] = out[lag_cols].std(axis=1, ddof=0)
    out["donation_count_roll3_min"] = out[lag_cols].min(axis=1)
    out["donation_count_roll3_max"] = out[lag_cols].max(axis=1)
    return out


def build_raw_row_frame() -> pd.DataFrame:
    raw_df = pd.read_csv(RAW, encoding="utf-8-sig")
    df = add_time_fields(raw_df)
    df = add_row_features(df)

    structure = build_county_month_structure(df)
    base = (
        df[["county_label", "month", "date", TARGET]]
        .drop_duplicates(["county_label", "month"])
        .sort_values(["county_label", "date"])
        .reset_index(drop=True)
    )
    history = add_target_history(base)
    history_cols = ["county_label", "month"] + TARGET_HISTORY_FEATURES

    out = df.merge(structure, on=["county_label", "month"], how="left")
    out = out.merge(history[history_cols], on=["county_label", "month"], how="left")
    out["county_name"] = out["county_label"].map(COUNTY_NAMES)
    return out


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= 2022].copy()
    valid = df[df["year"] >= 2023].copy()
    return train, valid


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    return df.dropna(subset=required).copy()


def encode_features(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_cat = pd.get_dummies(
        train[CATEGORICAL_FEATURES].astype(str),
        columns=CATEGORICAL_FEATURES,
        prefix=CATEGORICAL_FEATURES,
    )
    valid_cat = pd.get_dummies(
        valid[CATEGORICAL_FEATURES].astype(str),
        columns=CATEGORICAL_FEATURES,
        prefix=CATEGORICAL_FEATURES,
    )
    valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)

    train_x = pd.concat(
        [train[NUMERIC_FEATURES].reset_index(drop=True), train_cat.reset_index(drop=True)],
        axis=1,
    )
    valid_x = pd.concat(
        [valid[NUMERIC_FEATURES].reset_index(drop=True), valid_cat.reset_index(drop=True)],
        axis=1,
    )
    return train_x, valid_x, train_x.columns.tolist()


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = 1.0 / frame["row_count_per_county_month"].to_numpy(dtype=float)
    return weights / np.mean(weights)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + EPSILON
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / denom))


def epsilon_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), EPSILON)
    return float(np.mean(100.0 * np.abs(y_true - y_pred) / denom))


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "epsilon_mape": epsilon_mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tune_params(train_raw: pd.DataFrame, weighting: str) -> dict:
    inner_train = clean_frame(train_raw[train_raw["year"] < 2022])
    inner_valid = clean_frame(train_raw[train_raw["year"] == 2022])
    inner_train_x, inner_valid_x, model_columns = encode_features(inner_train, inner_valid)
    ytr = inner_train[TARGET].to_numpy()
    yva = inner_valid[TARGET].to_numpy()

    rows = []
    best = None
    best_score = np.inf
    for candidate in model_candidates():
        model = build_model(candidate["params"])
        weights = sample_weights(inner_train) if weighting == "weighted" else None
        model.fit(inner_train_x, ytr, sample_weight=weights)
        preds = model.predict(inner_valid_x[model_columns])
        metric = metrics_for(yva, preds)
        row = {
            "weighting": weighting,
            "candidate": candidate["name"],
            **metric,
            "params": candidate["params"],
        }
        rows.append(row)
        score = metric["smape"] + 0.1 * metric["epsilon_mape"]
        if score < best_score:
            best_score = score
            best = row

    assert best is not None
    return {
        "weighting": weighting,
        "selected_candidate": best["candidate"],
        "selected_params": best["params"],
        "selection_score": best_score,
        "inner_validation": rows,
    }


def write_prediction_outputs(
    outdir: Path,
    valid: pd.DataFrame,
    preds: np.ndarray,
    model,
    model_columns: list[str],
    metrics: dict,
    metadata: dict,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    pred_df = valid[CSV_COLS + ["county_name", "date"]].copy()
    pred_df["predicted_donation_count"] = preds
    pred_df["abs_error"] = np.abs(pred_df[TARGET] - pred_df["predicted_donation_count"])
    pred_df["ape_percent"] = pred_df["abs_error"] / np.maximum(np.abs(pred_df[TARGET]), EPSILON) * 100.0
    pred_df.to_csv(outdir / "validation_predictions.csv", index=False)

    importance = (
        pd.DataFrame({"feature": model_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(outdir / "feature_importance.csv", index=False)
    model.save_model(str(outdir / "xgboost_model.json"))

    scored = pred_df.copy()
    scored["count_diff"] = scored["predicted_donation_count"] - scored[TARGET]
    scored["abs_count_diff"] = scored["count_diff"].abs()

    county_month = (
        scored.groupby(["county_label", "county_name", "month"], sort=True)
        .agg(
            actual_donation_count=(TARGET, "first"),
            predicted_donation_count=("predicted_donation_count", "mean"),
            raw_rows=(TARGET, "size"),
            mean_abs_row_error=("abs_count_diff", "mean"),
        )
        .reset_index()
    )
    county_month["year"] = pd.to_datetime(county_month["month"] + "-01").dt.year
    county_month["count_diff"] = county_month["predicted_donation_count"] - county_month["actual_donation_count"]
    county_month["abs_count_diff"] = county_month["count_diff"].abs()
    county_month["pct_diff"] = (
        county_month["count_diff"] / np.maximum(county_month["actual_donation_count"], EPSILON) * 100.0
    )
    county_month["abs_pct_diff"] = county_month["pct_diff"].abs()
    county_month.to_csv(outdir / "county_month_validation_comparison.csv", index=False)

    county = (
        county_month.groupby(["county_label", "county_name"], sort=True)
        .agg(
            actual_donation_count=("actual_donation_count", "sum"),
            predicted_donation_count=("predicted_donation_count", "sum"),
            mean_abs_month_error=("abs_count_diff", "mean"),
            validation_months=("month", "size"),
            raw_rows=("raw_rows", "sum"),
        )
        .reset_index()
    )
    county["count_diff"] = county["predicted_donation_count"] - county["actual_donation_count"]
    county["abs_count_diff"] = county["count_diff"].abs()
    county["pct_diff"] = county["count_diff"] / np.maximum(county["actual_donation_count"], EPSILON) * 100.0
    county["abs_pct_diff"] = county["pct_diff"].abs()
    county = county.sort_values("abs_count_diff", ascending=False).reset_index(drop=True)
    county.to_csv(outdir / "county_validation_comparison.csv", index=False)
    county_plot_labels = county["county_label"].map(lambda value: f"C{int(value):02d}")

    county_year = (
        county_month.groupby(["county_label", "county_name", "year"], sort=True)
        .agg(
            actual_donation_count=("actual_donation_count", "sum"),
            predicted_donation_count=("predicted_donation_count", "sum"),
            validation_months=("month", "size"),
        )
        .reset_index()
    )
    county_year["count_diff"] = county_year["predicted_donation_count"] - county_year["actual_donation_count"]
    county_year["abs_count_diff"] = county_year["count_diff"].abs()
    county_year["pct_diff"] = county_year["count_diff"] / np.maximum(
        county_year["actual_donation_count"], EPSILON
    ) * 100.0
    county_year["abs_pct_diff"] = county_year["pct_diff"].abs()
    county_year = county_year.sort_values(["year", "abs_count_diff"], ascending=[True, False])
    county_year.to_csv(outdir / "county_year_validation_comparison.csv", index=False)

    plt.figure(figsize=(11, 6))
    x = np.arange(len(county))
    width = 0.4
    plt.bar(x - width / 2, county["actual_donation_count"], width=width, label="Actual")
    plt.bar(x + width / 2, county["predicted_donation_count"], width=width, label="Predicted")
    plt.xticks(x, county_plot_labels, rotation=35, ha="right")
    plt.ylabel("donation_count sum, county-month deduplicated")
    plt.title(f"County actual vs predicted donation_count - {metadata['variant']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "county_validation_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()

    summary = {
        **metadata,
        **metrics,
        "mean_county_abs_pct_diff": float(county["abs_pct_diff"].mean()),
        "median_county_abs_pct_diff": float(county["abs_pct_diff"].median()),
        "largest_abs_count_diff_county": county.iloc[0][
            ["county_name", "abs_count_diff", "pct_diff"]
        ].to_dict(),
        "largest_abs_pct_diff_county": county.sort_values("abs_pct_diff", ascending=False)
        .iloc[0][["county_name", "abs_pct_diff", "count_diff"]]
        .to_dict(),
    }
    (outdir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def train_and_score_variant(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    weighting: str,
    params: dict,
    candidate_name: str,
    variant_outdir: Path,
) -> tuple[dict, object, pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    train_x, valid_x, model_columns = encode_features(train, valid)
    ytr = train[TARGET].to_numpy()
    yva = valid[TARGET].to_numpy()

    model = build_model(params)
    weights = sample_weights(train) if weighting == "weighted" else None
    model.fit(train_x, ytr, sample_weight=weights)
    preds = model.predict(valid_x[model_columns])
    metric = metrics_for(yva, preds)
    metadata = {
        "pipeline": "raw_row_enhanced",
        "variant": f"{weighting}_{candidate_name}",
        "target": TARGET,
        "weighting": weighting,
        "selected_candidate": candidate_name,
        "train_policy": "year <= 2022",
        "validation_policy": "year >= 2023",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_columns_after_encoding": model_columns,
        "params": params,
    }
    summary = write_prediction_outputs(
        variant_outdir, valid, preds, model, model_columns, metric, metadata
    )
    return summary, model, train_x, valid_x, model_columns, preds


def write_shap_outputs(
    outdir: Path,
    valid_x: pd.DataFrame,
    model_columns: list[str],
    model,
) -> dict:
    sample_n = min(5000, len(valid_x))
    shap_x = valid_x.sample(n=sample_n, random_state=42).reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_x)
    imp = (
        pd.DataFrame({"feature": model_columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(outdir / "shap_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    top = imp.head(20).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"], color="tab:green")
    plt.title("Enhanced donation_count SHAP importance")
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(outdir / "shap_importance.png", dpi=130)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, shap_x, feature_names=model_columns, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    summary = {
        "target": TARGET,
        "shap_sample_rows": sample_n,
        "top_features": imp.head(15).to_dict(orient="records"),
    }
    (outdir / "shap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load_baseline_metrics() -> dict | None:
    path = BASELINE_OUTDIR / "metrics.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    county_path = BASELINE_OUTDIR / "county_validation_comparison.csv"
    if county_path.exists():
        county = pd.read_csv(county_path)
        metrics["mean_county_abs_pct_diff"] = float(county["abs_pct_diff"].mean())
    return metrics


def add_selection_scores(comparison: pd.DataFrame) -> pd.DataFrame:
    out = comparison.copy()
    rank_cols = ["rmse", "mae", "smape", "epsilon_mape", "mean_county_abs_pct_diff"]
    score_parts = []
    for col in rank_cols:
        rank_col = f"{col}_rank"
        out[rank_col] = out[col].rank(method="min", ascending=True)
        score_parts.append(rank_col)
    out["r2_rank"] = out["r2"].rank(method="min", ascending=False)
    score_parts.append("r2_rank")
    out["selection_score"] = out[score_parts].mean(axis=1)
    return out.sort_values(["selection_score", "smape"]).reset_index(drop=True)


def plot_model_comparison(comparison: pd.DataFrame) -> None:
    plot_df = comparison.copy()
    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["variant"], plot_df["smape"], color="tab:blue")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("sMAPE")
    plt.title("Enhanced model comparison by validation sMAPE")
    plt.tight_layout()
    plt.savefig(OUTDIR / "model_comparison_smape.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["variant"], plot_df["mean_county_abs_pct_diff"], color="tab:orange")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean county abs pct diff")
    plt.title("Enhanced model comparison by county-level error")
    plt.tight_layout()
    plt.savefig(OUTDIR / "model_comparison_county_error.png", dpi=130, bbox_inches="tight")
    plt.close()


def fmt_num(value: float) -> str:
    return f"{value:,.3f}"


def write_report(
    comparison: pd.DataFrame,
    best_summary: dict,
    shap_summary: dict,
    baseline: dict | None,
    selection_rows: list[dict],
) -> None:
    best_dir = OUTDIR / "best_model"
    county = pd.read_csv(best_dir / "county_validation_comparison.csv")
    top_county = county.head(8)
    selection_df = pd.DataFrame(selection_rows)

    lines = [
        "# Enhanced donation_count XGBoost Report",
        "",
        "## Scope",
        "",
        "- Source data: `data/raw/encoded_ml_dataset.csv`",
        "- Target: `donation_count`",
        "- Grain: raw encoded rows; no county-month aggregation for training rows",
        "- Train split: `year <= 2022`",
        "- Validation split: `year >= 2023`",
        "- The unrelated Trimmomatic/HISAT2/featureCounts table was not used.",
        "",
        "## Best Enhanced Model",
        "",
        f"- Variant: `{best_summary['variant']}`",
        f"- RMSE: `{fmt_num(best_summary['rmse'])}`",
        f"- MAE: `{fmt_num(best_summary['mae'])}`",
        f"- sMAPE: `{fmt_num(best_summary['smape'])}%`",
        f"- epsilon-MAPE: `{fmt_num(best_summary['epsilon_mape'])}%`",
        f"- R2: `{fmt_num(best_summary['r2'])}`",
        f"- Mean county abs pct diff: `{fmt_num(best_summary['mean_county_abs_pct_diff'])}%`",
        "",
    ]

    if baseline:
        lines += [
            "## Baseline Comparison",
            "",
            "| Metric | Baseline | Enhanced | Delta |",
            "|---|---:|---:|---:|",
        ]
        metrics_to_compare = ["rmse", "mae", "smape", "epsilon_mape", "r2"]
        if "mean_county_abs_pct_diff" in baseline:
            metrics_to_compare.append("mean_county_abs_pct_diff")
        for metric in metrics_to_compare:
            base_value = float(baseline[metric])
            enh_value = float(best_summary[metric])
            delta = enh_value - base_value
            lines.append(
                f"| {metric} | {fmt_num(base_value)} | {fmt_num(enh_value)} | {fmt_num(delta)} |"
            )
        lines.append("")

    lines += [
        "## Model Variants",
        "",
        "| Variant | Selection score | RMSE | MAE | sMAPE | epsilon-MAPE | R2 | Mean county abs pct diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            "| "
            + f"{row['variant']} | {fmt_num(row['selection_score'])} | "
            + f"{fmt_num(row['rmse'])} | {fmt_num(row['mae'])} | "
            + f"{fmt_num(row['smape'])}% | {fmt_num(row['epsilon_mape'])}% | "
            + f"{fmt_num(row['r2'])} | {fmt_num(row['mean_county_abs_pct_diff'])}% |"
        )
    lines.append("")

    lines += [
        "## Inner 2022 Parameter Selection",
        "",
        "| Weighting | Candidate | Inner sMAPE | Inner MAE | Inner R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in selection_df.iterrows():
        lines.append(
            f"| {row['weighting']} | {row['candidate']} | {fmt_num(row['smape'])}% | "
            f"{fmt_num(row['mae'])} | {fmt_num(row['r2'])} |"
        )
    lines.append("")

    lines += [
        "## Top SHAP Features",
        "",
        "| Feature | mean_abs_SHAP |",
        "|---|---:|",
    ]
    for row in shap_summary["top_features"][:10]:
        lines.append(f"| {row['feature']} | {fmt_num(float(row['mean_abs_shap']))} |")
    lines.append("")

    lines += [
        "## Largest County-Level Validation Errors",
        "",
        "| County | Actual | Predicted | Diff | Pct diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in top_county.iterrows():
        lines.append(
            f"| {row['county_name']} | {fmt_num(row['actual_donation_count'])} | "
            f"{fmt_num(row['predicted_donation_count'])} | {fmt_num(row['count_diff'])} | "
            f"{fmt_num(row['pct_diff'])}% |"
        )
    lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- County identity is now explicit, so stable county-specific bias can be learned instead of forced through common economic variables only.",
        "- County-month structure features preserve raw-row detail while giving the model month-level composition information.",
        "- Weighted raw-row training tests whether repeated county-month targets were dominating the fit through row count alone.",
        "- Use `county_year_validation_comparison.csv` to identify whether remaining error is concentrated in 2023 or 2024.",
    ]
    (OUTDIR / "enhancement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "variants").mkdir(parents=True, exist_ok=True)

    frame = build_raw_row_frame()
    train_raw, valid_raw = split_frames(frame)
    train_raw[CSV_COLS].to_csv(DATA_DIR / "trainning.csv", index=False)
    valid_raw[CSV_COLS].to_csv(DATA_DIR / "vaildation.csv", index=False)

    train = clean_frame(train_raw)
    valid = clean_frame(valid_raw)

    selection_payload = {}
    selection_rows = []
    variant_specs = []
    for weighting in ("unweighted", "weighted"):
        tuned = tune_params(train_raw, weighting)
        selection_payload[weighting] = tuned
        for row in tuned["inner_validation"]:
            flat = {k: v for k, v in row.items() if k != "params"}
            selection_rows.append(flat)
        variant_specs.append(
            {
                "weighting": weighting,
                "candidate": tuned["selected_candidate"],
                "params": tuned["selected_params"],
            }
        )

    (OUTDIR / "tuning_results.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(selection_rows).to_csv(OUTDIR / "tuning_results.csv", index=False)

    summaries = []
    bundles = []
    for spec in variant_specs:
        variant = f"{spec['weighting']}_{spec['candidate']}"
        print(f"Training final variant: {variant}")
        summary, model, train_x, valid_x, model_columns, preds = train_and_score_variant(
            train,
            valid,
            spec["weighting"],
            spec["params"],
            spec["candidate"],
            OUTDIR / "variants" / variant,
        )
        summaries.append(summary)
        bundles.append(
            {
                "summary": summary,
                "model": model,
                "train_x": train_x,
                "valid_x": valid_x,
                "model_columns": model_columns,
                "preds": preds,
            }
        )

    comparison = add_selection_scores(pd.DataFrame(summaries))
    comparison.to_csv(OUTDIR / "model_comparison.csv", index=False)
    plot_model_comparison(comparison)

    best_variant = comparison.iloc[0]["variant"]
    best_bundle = next(bundle for bundle in bundles if bundle["summary"]["variant"] == best_variant)

    best_summary = best_bundle["summary"]
    best_dir = OUTDIR / "best_model"
    write_prediction_outputs(
        best_dir,
        valid,
        best_bundle["preds"],
        best_bundle["model"],
        best_bundle["model_columns"],
        {k: best_summary[k] for k in ["rmse", "mae", "smape", "epsilon_mape", "r2"]},
        {
            key: best_summary[key]
            for key in [
                "pipeline",
                "variant",
                "target",
                "weighting",
                "selected_candidate",
                "train_policy",
                "validation_policy",
                "numeric_features",
                "categorical_features",
                "model_columns_after_encoding",
                "params",
            ]
        },
    )

    shap_summary = write_shap_outputs(
        best_dir,
        best_bundle["valid_x"],
        best_bundle["model_columns"],
        best_bundle["model"],
    )
    baseline = load_baseline_metrics()
    write_report(comparison, best_summary, shap_summary, baseline, selection_rows)

    print(f"Training CSV: {DATA_DIR / 'trainning.csv'}")
    print(f"Validation CSV: {DATA_DIR / 'vaildation.csv'}")
    print(f"Output dir: {OUTDIR}")
    print(f"Best variant: {best_summary['variant']}")
    print(f"RMSE: {best_summary['rmse']:.6f}")
    print(f"MAE: {best_summary['mae']:.6f}")
    print(f"sMAPE: {best_summary['smape']:.6f}")
    print(f"epsilon-MAPE: {best_summary['epsilon_mape']:.6f}")
    print(f"R2: {best_summary['r2']:.6f}")
    print(f"Mean county abs pct diff: {best_summary['mean_county_abs_pct_diff']:.6f}")


if __name__ == "__main__":
    main()
