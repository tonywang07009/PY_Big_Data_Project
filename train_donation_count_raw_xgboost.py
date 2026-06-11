"""
Legacy donation_count training flow for the raw-row XGBoost diagnostic path.

The formal project mainline is now `train_donation_count_xgboost_ensemble.py`.
This script is retained to preserve the evidence behind the raw-row repeated
target diagnosis:
  - use all county_label values available in the raw encoded dataset
  - keep raw county x industry x carrier-type rows
  - build county-month donation_count lag features without crossing date gaps
  - run yearly expanding walk-forward validation
  - standardize numeric features per county using train-fold statistics only
  - one-hot encode categorical features and align validation columns to train
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
DATA_DIR = ROOT / "data" / "donation_count_raw_xgboost"
OUTDIR = ROOT / "model_outputs" / "donation_count_raw_xgboost"
WALK_DIR = OUTDIR / "walk_forward"
TARGET = "donation_count"
EPSILON = 1e-6

NUMERIC_FEATURES = [
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
    "donation_count_lag1",
    "donation_count_lag2",
    "donation_count_lag3",
]
CATEGORICAL_FEATURES = ["county_label", "industry_label", "carrier_type_label"]
CSV_COLS = ["county_label", "month", TARGET] + NUMERIC_FEATURES + [
    "industry_label",
    "carrier_type_label",
]

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
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
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


def add_target_lags(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame.copy()
    key = out[["county_label", "date", target]].copy()
    for lag in (1, 2, 3):
        shifted = key.copy()
        shifted["date"] = shifted["date"] + pd.DateOffset(months=lag)
        shifted = shifted.rename(columns={target: f"{target}_lag{lag}"})
        out = out.merge(
            shifted[["county_label", "date", f"{target}_lag{lag}"]],
            on=["county_label", "date"],
            how="left",
        )
    return out


def build_raw_row_frame() -> pd.DataFrame:
    raw_df = pd.read_csv(RAW, encoding="utf-8-sig")
    df = add_time_fields(raw_df)
    base = (
        df[["county_label", "month", "date", TARGET]]
        .drop_duplicates(["county_label", "month"])
        .sort_values(["county_label", "date"])
        .reset_index(drop=True)
    )
    lagged = add_target_lags(base, TARGET)
    out = df.merge(
        lagged[
            [
                "county_label",
                "month",
                "donation_count_lag1",
                "donation_count_lag2",
                "donation_count_lag3",
            ]
        ],
        on=["county_label", "month"],
        how="left",
    )
    out["county_name"] = out["county_label"].map(COUNTY_NAMES)
    return out


def walk_forward_splits(df: pd.DataFrame) -> list[dict]:
    years = sorted(int(y) for y in df["year"].unique())
    folds = []
    for valid_year in years:
        if valid_year < 2023:
            continue
        train = df[df["year"] < valid_year].copy()
        valid = df[df["year"] == valid_year].copy()
        if train.empty or valid.empty:
            continue
        folds.append(
            {
                "fold": f"fold_{valid_year}",
                "train_policy": f"year < {valid_year}",
                "validation_policy": f"year == {valid_year}",
                "validation_year": valid_year,
                "train": train,
                "valid": valid,
            }
        )
    return folds


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    return df.dropna(subset=required).copy()


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


def apply_county_scalers(
    frame: pd.DataFrame,
    scalers: dict[int, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    out = frame.copy()
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype(float)
    missing_counties = sorted(set(int(c) for c in out["county_label"].unique()) - set(scalers))
    if missing_counties:
        raise ValueError(f"Validation contains counties absent from training scaler map: {missing_counties}")
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


def write_prediction_file(path: Path, valid: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    pred_df = valid[CSV_COLS + ["county_name"]].copy()
    pred_df["predicted_donation_count"] = preds
    pred_df["abs_error"] = np.abs(pred_df[TARGET] - pred_df["predicted_donation_count"])
    pred_df["ape_percent"] = pred_df["abs_error"] / np.maximum(np.abs(pred_df[TARGET]), EPSILON) * 100.0
    pred_df.to_csv(path, index=False)
    return pred_df


def train_fold(fold_spec: dict) -> dict:
    fold_dir = WALK_DIR / fold_spec["fold"]
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_raw = fold_spec["train"]
    valid_raw = fold_spec["valid"]
    train_before = len(train_raw)
    valid_before = len(valid_raw)
    train = clean_frame(train_raw)
    valid = clean_frame(valid_raw)
    if train.empty or valid.empty:
        raise ValueError(f"{fold_spec['fold']} produced an empty train or validation frame after cleaning.")

    train_x, valid_x, model_columns, scalers = encode_features(train, valid)
    ytr = train[TARGET].to_numpy()
    yva = valid[TARGET].to_numpy()

    model = build_model()
    model.fit(train_x[model_columns], ytr)
    preds = model.predict(valid_x[model_columns])
    metric = metrics_for(yva, preds)

    write_prediction_file(fold_dir / "validation_predictions.csv", valid, preds)
    importance = (
        pd.DataFrame({"feature": model_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(fold_dir / "feature_importance.csv", index=False)
    model.save_model(str(fold_dir / "xgboost_model.json"))

    scaler_payload = {
        "standardization": "numeric features standardized per county_label",
        "fit_scope": "train fold only",
        "numeric_features": NUMERIC_FEATURES,
        "county_scalers": scalers,
    }
    write_json(fold_dir / "county_numeric_scalers.json", scaler_payload)

    metrics = {
        "pipeline": "raw_row_xgboost_walk_forward",
        "fold": fold_spec["fold"],
        "target": TARGET,
        "train_policy": fold_spec["train_policy"],
        "validation_policy": fold_spec["validation_policy"],
        "validation_year": fold_spec["validation_year"],
        "standardization_policy": "fit numeric county-wise scalers on train fold, transform validation fold",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_columns_after_encoding": model_columns,
        "train_rows_before_dropna": train_before,
        "train_rows_used": len(train),
        "train_rows_dropped_missing_features": train_before - len(train),
        "validation_rows_before_dropna": valid_before,
        "validation_rows_used": len(valid),
        "validation_rows_dropped_missing_features": valid_before - len(valid),
        "train_years": sorted(int(y) for y in train["year"].unique()),
        "validation_years": sorted(int(y) for y in valid["year"].unique()),
        "county_labels": sorted(int(c) for c in valid["county_label"].unique()),
        **metric,
    }
    write_json(fold_dir / "metrics.json", metrics)
    return metrics


def write_summary(fold_metrics: list[dict]) -> dict:
    summary_rows = [
        {
            "fold": item["fold"],
            "train_policy": item["train_policy"],
            "validation_policy": item["validation_policy"],
            "train_rows_used": item["train_rows_used"],
            "validation_rows_used": item["validation_rows_used"],
            "rmse": item["rmse"],
            "mae": item["mae"],
            "smape": item["smape"],
            "epsilon_mape": item["epsilon_mape"],
            "r2": item["r2"],
        }
        for item in fold_metrics
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "walk_forward_summary.csv", index=False)
    aggregate = {
        "pipeline": "raw_row_xgboost_walk_forward",
        "target": TARGET,
        "walk_forward_design": "yearly expanding",
        "standardization_policy": "numeric features per county_label, train fold only",
        "folds": summary_rows,
        "mean_rmse": float(summary_df["rmse"].mean()),
        "mean_mae": float(summary_df["mae"].mean()),
        "mean_smape": float(summary_df["smape"].mean()),
        "mean_epsilon_mape": float(summary_df["epsilon_mape"].mean()),
        "mean_r2": float(summary_df["r2"].mean()),
        "selected_root_fold": fold_metrics[-1]["fold"],
    }
    write_json(OUTDIR / "walk_forward_summary.json", aggregate)
    write_json(OUTDIR / "metrics.json", fold_metrics[-1])
    return aggregate


def copy_root_compatibility_outputs(selected_fold: str) -> None:
    fold_dir = WALK_DIR / selected_fold
    for name in [
        "validation_predictions.csv",
        "feature_importance.csv",
        "xgboost_model.json",
        "county_numeric_scalers.json",
    ]:
        src = fold_dir / name
        if src.exists():
            (OUTDIR / name).write_bytes(src.read_bytes())


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    WALK_DIR.mkdir(parents=True, exist_ok=True)

    frame = build_raw_row_frame()
    legacy_train = frame[frame["year"] <= 2022].copy()
    legacy_valid = frame[frame["year"] >= 2023].copy()
    legacy_train[CSV_COLS].to_csv(DATA_DIR / "trainning.csv", index=False)
    legacy_valid[CSV_COLS].to_csv(DATA_DIR / "vaildation.csv", index=False)

    folds = walk_forward_splits(frame)
    if not folds:
        raise ValueError("No walk-forward folds were produced.")

    fold_metrics = []
    for fold_spec in folds:
        metrics = train_fold(fold_spec)
        fold_metrics.append(metrics)
        print(
            f"{metrics['fold']}: valid={metrics['validation_year']} "
            f"rows={metrics['validation_rows_used']} "
            f"RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} "
            f"sMAPE={metrics['smape']:.6f} R2={metrics['r2']:.6f}"
        )

    summary = write_summary(fold_metrics)
    copy_root_compatibility_outputs(summary["selected_root_fold"])

    print(f"Training CSV: {DATA_DIR / 'trainning.csv'}")
    print(f"Validation CSV: {DATA_DIR / 'vaildation.csv'}")
    print(f"Walk-forward summary: {OUTDIR / 'walk_forward_summary.csv'}")
    print(f"Selected root fold: {summary['selected_root_fold']}")


if __name__ == "__main__":
    main()
