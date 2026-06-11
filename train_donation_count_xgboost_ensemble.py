"""
Train the MecDonate donation_count XGBoost ensemble.

Pipeline:
  - build one county-month row from raw county x industry x carrier rows
  - run monthly expanding walk-forward validation from 2023-01 to 2024-12
  - train global XGBoost on log1p(donation_count)
  - build county historical baseline from lag/rolling features
  - fit a train-only county calibration layer
  - emit component and final ensemble predictions for audit
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
OUTDIR = ROOT / "model_outputs" / "donation_count_xgboost_ensemble"
FOLD_DIR = OUTDIR / "monthly_walk_forward"
TARGET = "donation_count"
EPSILON = 1e-6
ENSEMBLE_XGB_WEIGHT = 0.6
ENSEMBLE_BASELINE_WEIGHT = 0.4

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

NUMERIC_FEATURES = [
    "housing_burden",
    "price_income_ratio",
    "total_county_invoice_count",
    "total_county_invoice_amount",
    "avg_invoice_value",
    "log_total_county_invoice_count",
    "log_total_county_invoice_amount",
    "carrier_usage_ratio",
    "n_active_industries",
    "industry_hhi",
    "industry_entropy",
    "top1_industry_share",
    "top3_industry_share",
    "row_count_per_county_month",
    "month_num",
    "month_sin",
    "month_cos",
    "time_index",
    "donation_count_lag1",
    "donation_count_lag2",
    "donation_count_lag3",
    "donation_count_roll3_mean",
    "donation_count_roll3_std",
    "donation_count_roll3_min",
    "donation_count_roll3_max",
]
CATEGORICAL_FEATURES = ["county_label"]
PREDICTION_COLUMNS = [
    "county_label",
    "county_name",
    "month",
    "actual_donation_count",
    "xgb_log_prediction",
    "calibrated_xgb_prediction",
    "historical_baseline_prediction",
    "ensemble_prediction",
    "ensemble_abs_error",
    "ensemble_ape_percent",
]


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def build_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=4.0,
        random_state=42,
        n_jobs=1,
    )


def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["month"] + "-01")
    out["year"] = out["date"].dt.year
    out["month_num"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
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


def add_target_history(cm: pd.DataFrame) -> pd.DataFrame:
    out = cm.copy()
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


def build_county_month_frame(raw_path: Path = RAW) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    raw = add_time_fields(raw)
    grp = raw.groupby(["county_label", "month"], sort=True)
    base = grp[
        [
            "housing_burden",
            "price_income_ratio",
            "total_county_invoice_count",
            "total_county_invoice_amount",
            TARGET,
        ]
    ].first()
    structure = grp.apply(county_month_aggregate)
    cm = base.join(structure).reset_index()
    cm["date"] = pd.to_datetime(cm["month"] + "-01")
    cm["year"] = cm["date"].dt.year
    cm["month_num"] = cm["date"].dt.month
    cm["month_sin"] = np.sin(2 * np.pi * cm["month_num"] / 12)
    cm["month_cos"] = np.cos(2 * np.pi * cm["month_num"] / 12)
    cm["time_index"] = (cm["date"].dt.year - cm["date"].dt.year.min()) * 12 + cm["date"].dt.month
    cm["avg_invoice_value"] = cm["total_county_invoice_amount"] / cm["total_county_invoice_count"]
    cm["log_total_county_invoice_count"] = np.log1p(cm["total_county_invoice_count"])
    cm["log_total_county_invoice_amount"] = np.log1p(cm["total_county_invoice_amount"])
    cm["county_name"] = cm["county_label"].map(COUNTY_NAMES)
    cm = cm.sort_values(["county_label", "date"]).reset_index(drop=True)
    return add_target_history(cm)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    return df.dropna(subset=required).copy()


def monthly_walk_forward_splits(cm: pd.DataFrame) -> list[dict]:
    folds = []
    for valid_date in sorted(cm.loc[cm["date"] >= pd.Timestamp("2023-01-01"), "date"].unique()):
        train = cm[cm["date"] < valid_date].copy()
        valid = cm[cm["date"] == valid_date].copy()
        if train.empty or valid.empty:
            continue
        valid_month = pd.Timestamp(valid_date).strftime("%Y-%m")
        folds.append(
            {
                "fold": f"fold_{valid_month}",
                "train_policy": f"date < {valid_month}",
                "validation_policy": f"month == {valid_month}",
                "validation_month": valid_month,
                "train": train,
                "valid": valid,
            }
        )
    return folds


def fit_county_scalers(train: pd.DataFrame) -> dict[int, dict[str, dict[str, float]]]:
    scalers: dict[int, dict[str, dict[str, float]]] = {}
    for county, sub in train.groupby("county_label", sort=True):
        means = sub[NUMERIC_FEATURES].mean()
        stds = sub[NUMERIC_FEATURES].std(ddof=0).replace(0, 1.0).fillna(1.0)
        scalers[int(county)] = {
            "mean": {col: float(means[col]) for col in NUMERIC_FEATURES},
            "std": {col: float(stds[col]) for col in NUMERIC_FEATURES},
        }
    return scalers


def apply_county_scalers(frame: pd.DataFrame, scalers: dict[int, dict[str, dict[str, float]]]) -> pd.DataFrame:
    out = frame.copy()
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype(float)
    missing = sorted(set(int(c) for c in out["county_label"].unique()) - set(scalers))
    if missing:
        raise ValueError(f"Validation contains counties absent from training scaler map: {missing}")
    for county, idx in out.groupby("county_label", sort=True).groups.items():
        scaler = scalers[int(county)]
        means = pd.Series(scaler["mean"])
        stds = pd.Series(scaler["std"])
        out.loc[idx, NUMERIC_FEATURES] = (out.loc[idx, NUMERIC_FEATURES] - means) / stds
    return out


def encode_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    scalers = fit_county_scalers(train)
    train_scaled = apply_county_scalers(train, scalers)
    valid_scaled = apply_county_scalers(valid, scalers)

    train_cat = pd.get_dummies(
        train_scaled[CATEGORICAL_FEATURES].astype(str),
        columns=CATEGORICAL_FEATURES,
        prefix=CATEGORICAL_FEATURES,
    )
    valid_cat = pd.get_dummies(
        valid_scaled[CATEGORICAL_FEATURES].astype(str),
        columns=CATEGORICAL_FEATURES,
        prefix=CATEGORICAL_FEATURES,
    )
    valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)

    train_x = pd.concat(
        [train_scaled[NUMERIC_FEATURES].reset_index(drop=True), train_cat.reset_index(drop=True)],
        axis=1,
    )
    valid_x = pd.concat(
        [valid_scaled[NUMERIC_FEATURES].reset_index(drop=True), valid_cat.reset_index(drop=True)],
        axis=1,
    )
    return train_x, valid_x, train_x.columns.tolist(), scalers


def historical_baseline(valid: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    county_median = train.groupby("county_label")[TARGET].median()
    global_median = float(train[TARGET].median())
    preds = []
    for _, row in valid.iterrows():
        value = row.get("donation_count_roll3_mean")
        if pd.isna(value):
            value = row.get("donation_count_lag1")
        if pd.isna(value):
            value = county_median.get(row["county_label"], global_median)
        if pd.isna(value):
            value = global_median
        preds.append(max(float(value), 0.0))
    return np.asarray(preds, dtype=float)


def fit_linear_calibration(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(actual) < 2 or np.nanstd(pred) < EPSILON:
        ratio = float(np.sum(actual) / max(np.sum(pred), EPSILON))
        return {"a": ratio, "b": 0.0, "method": "ratio"}
    design = np.column_stack([pred, np.ones(len(pred))])
    a, b = np.linalg.lstsq(design, actual, rcond=None)[0]
    return {"a": float(a), "b": float(b), "method": "linear"}


def fit_calibration_params(train: pd.DataFrame) -> dict:
    dates = sorted(train["date"].unique())
    calib_count = min(6, max(1, len(dates) // 4))
    calib_dates = set(dates[-calib_count:])
    main = train[~train["date"].isin(calib_dates)].copy()
    calib = train[train["date"].isin(calib_dates)].copy()
    if main.empty or calib.empty:
        global_median = float(train[TARGET].median())
        return {"global": {"a": 1.0, "b": 0.0, "method": "identity"}, "county": {}, "global_median": global_median}

    main_x, calib_x, cols, _ = encode_features(main, calib)
    model = build_model()
    model.fit(main_x[cols], np.log1p(main[TARGET].to_numpy()))
    calib_pred = np.expm1(model.predict(calib_x[cols]))
    calib_pred = np.clip(calib_pred, 0.0, None)
    actual = calib[TARGET].to_numpy(dtype=float)

    global_params = fit_linear_calibration(actual, calib_pred)
    county_params = {}
    for county, sub_idx in calib.groupby("county_label", sort=True).groups.items():
        pos = calib.index.get_indexer(sub_idx)
        if len(pos) >= 3:
            county_params[str(int(county))] = fit_linear_calibration(actual[pos], calib_pred[pos])

    return {
        "global": global_params,
        "county": county_params,
        "global_median": float(train[TARGET].median()),
        "calibration_months": [pd.Timestamp(d).strftime("%Y-%m") for d in sorted(calib_dates)],
    }


def apply_calibration(pred: np.ndarray, counties: pd.Series, params: dict) -> np.ndarray:
    out = []
    global_params = params["global"]
    for value, county in zip(pred, counties):
        item = params.get("county", {}).get(str(int(county)), global_params)
        calibrated = item["a"] * float(value) + item["b"]
        out.append(max(calibrated, 0.0))
    return np.asarray(out, dtype=float)


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
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def build_prediction_frame(
    valid: pd.DataFrame,
    xgb_pred: np.ndarray,
    calibrated_pred: np.ndarray,
    baseline_pred: np.ndarray,
    ensemble_pred: np.ndarray,
) -> pd.DataFrame:
    out = valid[["county_label", "county_name", "month", TARGET]].copy()
    out = out.rename(columns={TARGET: "actual_donation_count"})
    out["xgb_log_prediction"] = xgb_pred
    out["calibrated_xgb_prediction"] = calibrated_pred
    out["historical_baseline_prediction"] = baseline_pred
    out["ensemble_prediction"] = ensemble_pred
    out["ensemble_abs_error"] = (out["ensemble_prediction"] - out["actual_donation_count"]).abs()
    out["ensemble_ape_percent"] = (
        out["ensemble_abs_error"] / np.maximum(out["actual_donation_count"].abs(), EPSILON) * 100.0
    )
    return out


def train_fold(fold_spec: dict) -> tuple[dict, pd.DataFrame]:
    fold_dir = FOLD_DIR / fold_spec["fold"]
    fold_dir.mkdir(parents=True, exist_ok=True)

    train = clean_frame(fold_spec["train"])
    valid = clean_frame(fold_spec["valid"])
    if train.empty or valid.empty:
        raise ValueError(f"{fold_spec['fold']} produced an empty train or validation frame.")

    train_x, valid_x, model_columns, scalers = encode_features(train, valid)
    calibration = fit_calibration_params(train)

    model = build_model()
    model.fit(train_x[model_columns], np.log1p(train[TARGET].to_numpy()))
    xgb_pred = np.expm1(model.predict(valid_x[model_columns]))
    xgb_pred = np.clip(xgb_pred, 0.0, None)
    calibrated_pred = apply_calibration(xgb_pred, valid["county_label"], calibration)
    baseline_pred = historical_baseline(valid, train)
    ensemble_pred = (
        ENSEMBLE_XGB_WEIGHT * calibrated_pred + ENSEMBLE_BASELINE_WEIGHT * baseline_pred
    )
    ensemble_pred = np.clip(ensemble_pred, 0.0, None)

    pred_df = build_prediction_frame(valid, xgb_pred, calibrated_pred, baseline_pred, ensemble_pred)
    pred_df.to_csv(fold_dir / "ensemble_predictions.csv", index=False)
    pred_df.to_csv(fold_dir / "component_predictions.csv", index=False)

    importance = (
        pd.DataFrame({"feature": model_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(fold_dir / "feature_importance.csv", index=False)
    model.save_model(str(fold_dir / "xgboost_model.json"))

    write_json(
        fold_dir / "county_numeric_scalers.json",
        {
            "standardization": "numeric features standardized per county_label",
            "fit_scope": "train fold only",
            "numeric_features": NUMERIC_FEATURES,
            "county_scalers": scalers,
        },
    )
    write_json(fold_dir / "calibration_params.json", calibration)

    y_true = valid[TARGET].to_numpy(dtype=float)
    metrics = {
        "pipeline": "donation_count_xgboost_ensemble",
        "fold": fold_spec["fold"],
        "validation_month": fold_spec["validation_month"],
        "train_policy": fold_spec["train_policy"],
        "validation_policy": fold_spec["validation_policy"],
        "target": TARGET,
        "modeling_grain": "county_month",
        "ensemble_formula": f"{ENSEMBLE_XGB_WEIGHT} * calibrated_xgb + {ENSEMBLE_BASELINE_WEIGHT} * historical_baseline",
        "train_rows_used": len(train),
        "validation_rows_used": len(valid),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_columns_after_encoding": model_columns,
        "xgb_log_metrics": metrics_for(y_true, xgb_pred),
        "calibrated_xgb_metrics": metrics_for(y_true, calibrated_pred),
        "historical_baseline_metrics": metrics_for(y_true, baseline_pred),
        "ensemble_metrics": metrics_for(y_true, ensemble_pred),
    }
    write_json(fold_dir / "metrics.json", metrics)
    return metrics, pred_df


def write_aggregate_outputs(fold_metrics: list[dict], predictions: list[pd.DataFrame]) -> dict:
    all_preds = pd.concat(predictions, ignore_index=True)
    all_preds.to_csv(OUTDIR / "component_predictions.csv", index=False)
    all_preds[PREDICTION_COLUMNS].to_csv(OUTDIR / "ensemble_predictions.csv", index=False)

    monthly_rows = []
    for month, sub in all_preds.groupby("month", sort=True):
        y_true = sub["actual_donation_count"].to_numpy(dtype=float)
        row = {"month": month, "rows": len(sub)}
        for name, col in [
            ("xgb_log", "xgb_log_prediction"),
            ("calibrated_xgb", "calibrated_xgb_prediction"),
            ("historical_baseline", "historical_baseline_prediction"),
            ("ensemble", "ensemble_prediction"),
        ]:
            metric = metrics_for(y_true, sub[col].to_numpy(dtype=float))
            row.update({f"{name}_{k}": v for k, v in metric.items()})
        monthly_rows.append(row)
    monthly_summary = pd.DataFrame(monthly_rows)
    monthly_summary.to_csv(OUTDIR / "monthly_walk_forward_summary.csv", index=False)

    county_month = all_preds.copy()
    county_month["count_diff"] = county_month["ensemble_prediction"] - county_month["actual_donation_count"]
    county_month["abs_count_diff"] = county_month["count_diff"].abs()
    county_month["pct_diff"] = (
        county_month["count_diff"] / np.maximum(county_month["actual_donation_count"], EPSILON) * 100.0
    )
    county_month["abs_pct_diff"] = county_month["pct_diff"].abs()
    county_month.to_csv(OUTDIR / "county_month_validation_comparison.csv", index=False)

    county = (
        county_month.groupby(["county_label", "county_name"], sort=True)
        .agg(
            actual_donation_count=("actual_donation_count", "sum"),
            predicted_donation_count=("ensemble_prediction", "sum"),
            mean_abs_month_error=("abs_count_diff", "mean"),
            validation_months=("month", "size"),
        )
        .reset_index()
    )
    county["count_diff"] = county["predicted_donation_count"] - county["actual_donation_count"]
    county["abs_count_diff"] = county["count_diff"].abs()
    county["pct_diff"] = county["count_diff"] / np.maximum(county["actual_donation_count"], EPSILON) * 100.0
    county["abs_pct_diff"] = county["pct_diff"].abs()
    county = county.sort_values("abs_count_diff", ascending=False).reset_index(drop=True)
    county.to_csv(OUTDIR / "county_validation_comparison.csv", index=False)

    summary_rows = []
    for item in fold_metrics:
        row = {
            "fold": item["fold"],
            "validation_month": item["validation_month"],
            "train_rows_used": item["train_rows_used"],
            "validation_rows_used": item["validation_rows_used"],
        }
        row.update({f"ensemble_{k}": v for k, v in item["ensemble_metrics"].items()})
        summary_rows.append(row)

    latest_fold = fold_metrics[-1]["fold"]
    summary = {
        "pipeline": "donation_count_xgboost_ensemble",
        "target": TARGET,
        "validation": "monthly expanding walk-forward",
        "fold_count": len(fold_metrics),
        "selected_root_fold": latest_fold,
        "mean_monthly_ensemble_rmse": float(monthly_summary["ensemble_rmse"].mean()),
        "mean_monthly_ensemble_mae": float(monthly_summary["ensemble_mae"].mean()),
        "mean_monthly_ensemble_smape": float(monthly_summary["ensemble_smape"].mean()),
        "mean_monthly_ensemble_r2": float(monthly_summary["ensemble_r2"].mean()),
        "mean_county_abs_pct_diff": float(county["abs_pct_diff"].mean()),
        "median_county_abs_pct_diff": float(county["abs_pct_diff"].median()),
        "largest_abs_count_diff_county": county.iloc[0][
            ["county_name", "abs_count_diff", "pct_diff"]
        ].to_dict(),
        "folds": summary_rows,
    }
    write_json(OUTDIR / "metrics.json", summary)
    write_json(OUTDIR / "monthly_walk_forward_summary.json", summary)

    latest_dir = FOLD_DIR / latest_fold
    for name in ["xgboost_model.json", "feature_importance.csv", "county_numeric_scalers.json", "calibration_params.json"]:
        src = latest_dir / name
        if src.exists():
            (OUTDIR / name).write_bytes(src.read_bytes())
    return summary


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    cm = build_county_month_frame()
    cm.to_csv(OUTDIR / "county_month_matrix.csv", index=False)
    folds = monthly_walk_forward_splits(cm)
    if not folds:
        raise ValueError("No monthly walk-forward folds were produced.")

    fold_metrics = []
    fold_predictions = []
    for fold_spec in folds:
        metrics, pred_df = train_fold(fold_spec)
        fold_metrics.append(metrics)
        fold_predictions.append(pred_df)
        em = metrics["ensemble_metrics"]
        print(
            f"{metrics['fold']}: rows={metrics['validation_rows_used']} "
            f"RMSE={em['rmse']:.6f} MAE={em['mae']:.6f} "
            f"sMAPE={em['smape']:.6f} R2={em['r2']:.6f}"
        )

    summary = write_aggregate_outputs(fold_metrics, fold_predictions)
    print(f"Monthly folds: {summary['fold_count']}")
    print(f"Mean monthly ensemble sMAPE: {summary['mean_monthly_ensemble_smape']:.6f}")
    print(f"Mean county abs pct diff: {summary['mean_county_abs_pct_diff']:.6f}")
    print(f"Output dir: {OUTDIR}")


if __name__ == "__main__":
    main()
