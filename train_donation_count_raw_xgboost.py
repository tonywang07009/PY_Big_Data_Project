"""
Formal donation_count training flow using the best raw-row XGBoost pipeline.

Pipeline:
  - keep raw encoded rows
  - add time fields
  - compute county-month donation_count lag1/2/3 and merge back to raw rows
  - one-hot encode industry_label and carrier_type_label
  - train XGBoost on year <= 2022
  - validate on year >= 2023
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
DATA_DIR = ROOT / "data" / "donation_count_raw_xgboost"
OUTDIR = ROOT / "model_outputs" / "donation_count_raw_xgboost"
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
CATEGORICAL_FEATURES = ["industry_label", "carrier_type_label"]
CSV_COLS = ["county_label", "month", TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_model() -> XGBRegressor:
    return XGBRegressor(
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
    raw_df = pd.read_csv(RAW)
    df = add_time_fields(raw_df)
    base = (
        df[["county_label", "month", "date", TARGET]]
        .drop_duplicates(["county_label", "month"])
        .sort_values(["county_label", "date"])
        .reset_index(drop=True)
    )
    lagged = add_target_lags(base, TARGET)
    return df.merge(
        lagged[["county_label", "month", "donation_count_lag1", "donation_count_lag2", "donation_count_lag3"]],
        on=["county_label", "month"],
        how="left",
    )


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= 2022].copy()
    valid = df[df["year"] >= 2023].copy()
    return train, valid


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

    train_x = pd.concat([train[NUMERIC_FEATURES].reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    valid_x = pd.concat([valid[NUMERIC_FEATURES].reset_index(drop=True), valid_cat.reset_index(drop=True)], axis=1)
    return train_x, valid_x, train_x.columns.tolist()


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + EPSILON
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / denom))


def epsilon_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), EPSILON)
    return float(np.mean(100.0 * np.abs(y_true - y_pred) / denom))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    frame = build_raw_row_frame()
    train_raw, valid_raw = split_frames(frame)
    train_raw[CSV_COLS].to_csv(DATA_DIR / "trainning.csv", index=False)
    valid_raw[CSV_COLS].to_csv(DATA_DIR / "vaildation.csv", index=False)

    train_before = len(train_raw)
    valid_before = len(valid_raw)
    train = train_raw.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]).copy()
    valid = valid_raw.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]).copy()

    train_x, valid_x, model_columns = encode_features(train, valid)
    ytr = train[TARGET].to_numpy()
    yva = valid[TARGET].to_numpy()

    model = build_model()
    model.fit(train_x, ytr)
    preds = model.predict(valid_x)

    pred_df = valid[CSV_COLS].copy()
    pred_df["predicted_donation_count"] = preds
    pred_df["abs_error"] = np.abs(pred_df[TARGET] - pred_df["predicted_donation_count"])
    pred_df["ape_percent"] = pred_df["abs_error"] / np.maximum(np.abs(pred_df[TARGET]), EPSILON) * 100.0
    pred_df.to_csv(OUTDIR / "validation_predictions.csv", index=False)

    importance = pd.DataFrame(
        {"feature": model_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUTDIR / "feature_importance.csv", index=False)
    model.save_model(str(OUTDIR / "xgboost_model.json"))

    metrics = {
        "pipeline": "raw_row_baseline",
        "target": TARGET,
        "train_policy": "year <= 2022",
        "validation_policy": "year >= 2023",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_columns_after_encoding": model_columns,
        "train_rows_before_dropna": train_before,
        "train_rows_used": len(train),
        "train_rows_dropped_missing_features": train_before - len(train),
        "validation_rows_before_dropna": valid_before,
        "validation_rows_used": len(valid),
        "validation_rows_dropped_missing_features": valid_before - len(valid),
        "rmse": float(np.sqrt(mean_squared_error(yva, preds))),
        "mae": float(mean_absolute_error(yva, preds)),
        "smape": smape(yva, preds),
        "epsilon_mape": epsilon_mape(yva, preds),
        "r2": float(r2_score(yva, preds)),
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Training CSV: {DATA_DIR / 'trainning.csv'}")
    print(f"Validation CSV: {DATA_DIR / 'vaildation.csv'}")
    print(f"Train rows used: {len(train)}")
    print(f"Validation rows used: {len(valid)}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"sMAPE: {metrics['smape']:.6f}")
    print(f"epsilon-MAPE: {metrics['epsilon_mape']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")


if __name__ == "__main__":
    main()
