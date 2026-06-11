"""
Experiment: train one XGBoost model per predefined county group.

This script is not the formal project mainline. The formal path is
`train_donation_count_xgboost_ensemble.py`, which uses all available counties,
county-month `donation_count`, monthly expanding walk-forward validation,
county-wise numeric scaling, train-only calibration, and historical baseline
blending.

For each group:
  - preserve raw encoded rows (no county-month aggregation for modeling rows)
  - compute county-month lag features and merge them back onto each raw row
  - write group-specific trainning.csv and vaildation.csv
  - train on year <= 2022
  - validate on year >= 2023
  - save metrics, predictions, and feature importance
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
GROUP_ROOT = ROOT / "data" / "group_models"
TARGET = "donation_ratio"
EPSILON = 1e-6

GROUPS = {
    "group1": [15, 8, 16, 4, 7],
    "group2": [9, 0, 13, 12, 11, 10, 5, 17, 6],
    "group3": [1, 18, 3, 14],
    "group4": [2],
}

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

NUMERIC_MODEL_COLS = [
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
    "donation_ratio_lag1",
    "donation_ratio_lag2",
    "donation_ratio_lag3",
]
CATEGORICAL_MODEL_COLS = ["industry_label", "carrier_type_label"]
RAW_FEATURE_COLS = NUMERIC_MODEL_COLS + CATEGORICAL_MODEL_COLS
CSV_COLS = [
    "county_label",
    "county_name",
    "month",
    "year",
    TARGET,
] + RAW_FEATURE_COLS


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


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df[["county_label", "month", "date", TARGET]]
        .drop_duplicates(subset=["county_label", "month"])
        .sort_values(["county_label", "date"])
        .reset_index(drop=True)
    )
    key = base[["county_label", "date", TARGET]].copy()
    out = base.copy()
    for lag in (1, 2, 3):
        shifted = key.copy()
        shifted["date"] = shifted["date"] + pd.DateOffset(months=lag)
        shifted = shifted.rename(columns={TARGET: f"donation_ratio_lag{lag}"})
        out = out.merge(
            shifted[["county_label", "date", f"donation_ratio_lag{lag}"]],
            on=["county_label", "date"],
            how="left",
        )
    merged = df.merge(
        out[["county_label", "month", "donation_ratio_lag1", "donation_ratio_lag2", "donation_ratio_lag3"]],
        on=["county_label", "month"],
        how="left",
    )
    return merged


def prepare_raw_matrix() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df["county_name"] = df["county_label"].map(COUNTY_NAMES).fillna(df["county_label"].astype(str))
    df = add_time_fields(df)
    df = add_lag_features(df)
    return df


def split_group_frames(df: pd.DataFrame, county_codes: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df[df["county_label"].isin(county_codes)].copy()
    train = grouped[grouped["year"] <= 2022].copy()
    valid = grouped[grouped["year"] >= 2023].copy()
    return train, valid


def write_group_csvs(group_dir: Path, train: pd.DataFrame, valid: pd.DataFrame) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    train[CSV_COLS].to_csv(group_dir / "trainning.csv", index=False)
    valid[CSV_COLS].to_csv(group_dir / "vaildation.csv", index=False)


def drop_missing(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_before = len(train)
    valid_before = len(valid)
    train_clean = train.dropna(subset=RAW_FEATURE_COLS + [TARGET]).copy()
    valid_clean = valid.dropna(subset=RAW_FEATURE_COLS + [TARGET]).copy()
    info = {
        "train_rows_before_dropna": train_before,
        "train_rows_used": len(train_clean),
        "train_rows_dropped_missing_features": train_before - len(train_clean),
        "validation_rows_before_dropna": valid_before,
        "validation_rows_used": len(valid_clean),
        "validation_rows_dropped_missing_features": valid_before - len(valid_clean),
    }
    return train_clean, valid_clean, info


def encode_categoricals(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_num = train[NUMERIC_MODEL_COLS].copy()
    valid_num = valid[NUMERIC_MODEL_COLS].copy()

    train_cat = pd.get_dummies(
        train[CATEGORICAL_MODEL_COLS].astype(str),
        columns=CATEGORICAL_MODEL_COLS,
        prefix=CATEGORICAL_MODEL_COLS,
    )
    valid_cat = pd.get_dummies(
        valid[CATEGORICAL_MODEL_COLS].astype(str),
        columns=CATEGORICAL_MODEL_COLS,
        prefix=CATEGORICAL_MODEL_COLS,
    )
    valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)

    train_x = pd.concat([train_num.reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    valid_x = pd.concat([valid_num.reset_index(drop=True), valid_cat.reset_index(drop=True)], axis=1)
    model_cols = train_x.columns.tolist()
    return train_x, valid_x, model_cols


def normalize_group(train_x: pd.DataFrame, valid_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scaler = StandardScaler()
    train_scaled = train_x.copy()
    valid_scaled = valid_x.copy()
    scaler.fit(train_x[NUMERIC_MODEL_COLS])
    train_scaled[NUMERIC_MODEL_COLS] = scaler.transform(train_x[NUMERIC_MODEL_COLS])
    valid_scaled[NUMERIC_MODEL_COLS] = scaler.transform(valid_x[NUMERIC_MODEL_COLS])
    stats = {
        "normalization": "StandardScaler per group, fit on training rows only",
        "normalized_numeric_columns": NUMERIC_MODEL_COLS,
        "one_hot_columns": [c for c in train_x.columns if c not in NUMERIC_MODEL_COLS],
        "scaler_mean": {col: float(v) for col, v in zip(NUMERIC_MODEL_COLS, scaler.mean_)},
        "scaler_scale": {col: float(v) for col, v in zip(NUMERIC_MODEL_COLS, scaler.scale_)},
    }
    return train_scaled, valid_scaled, stats


def score_predictions(valid: pd.DataFrame, preds: np.ndarray) -> tuple[pd.DataFrame, dict]:
    scored = valid[CSV_COLS].copy()
    scored["predicted_donation_ratio"] = preds
    scored["abs_error"] = np.abs(scored[TARGET] - scored["predicted_donation_ratio"])
    scored["ape_percent"] = (
        scored["abs_error"] / np.maximum(np.abs(scored[TARGET]), EPSILON) * 100.0
    )
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(scored[TARGET], scored["predicted_donation_ratio"]))),
        "r2": float(r2_score(scored[TARGET], scored["predicted_donation_ratio"])),
        "mape_percent": float(scored["ape_percent"].mean()),
    }
    return scored, metrics


def train_group(group_name: str, county_codes: list[int], raw_df: pd.DataFrame) -> dict:
    group_dir = GROUP_ROOT / group_name
    train_raw, valid_raw = split_group_frames(raw_df, county_codes)
    write_group_csvs(group_dir, train_raw, valid_raw)
    train, valid, drop_info = drop_missing(train_raw, valid_raw)

    summary = {
        "group_name": group_name,
        "county_codes": county_codes,
        "county_names": [COUNTY_NAMES[c] for c in county_codes],
        "train_years": sorted(int(y) for y in train_raw["year"].unique().tolist()),
        "validation_years": sorted(int(y) for y in valid_raw["year"].unique().tolist()),
        "raw_row_workflow": True,
        **drop_info,
    }

    if train.empty or valid.empty:
        summary.update(
            {
                "status": "FAIL",
                "failure_reason": "Empty training or validation set after dropping missing features.",
            }
        )
        (group_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    train_x, valid_x, model_columns = encode_categoricals(train, valid)
    train_scaled, valid_scaled, norm_stats = normalize_group(train_x, valid_x)

    model = build_model()
    model.fit(train_scaled[model_columns], train[TARGET].to_numpy())
    preds = model.predict(valid_scaled[model_columns])
    scored, metrics = score_predictions(valid, preds)

    scored.to_csv(group_dir / "xgboost_predictions.csv", index=False)
    pd.DataFrame(
        {
            "feature": model_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        group_dir / "feature_importance.csv",
        index=False,
    )

    summary.update(
        {
            "status": "SUCCESS",
            "best_model": "XGBoost",
            "metric": "time_split_validation",
            "threshold_mape_percent": 5.0,
            "model_columns_after_encoding": model_columns,
            **norm_stats,
            **metrics,
        }
    )
    (group_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run() -> dict:
    GROUP_ROOT.mkdir(parents=True, exist_ok=True)
    raw_df = prepare_raw_matrix()
    group_summaries = [
        train_group(group_name, county_codes, raw_df)
        for group_name, county_codes in GROUPS.items()
    ]
    summary = {
        "target": TARGET,
        "raw_row_workflow": True,
        "numeric_model_columns": NUMERIC_MODEL_COLS,
        "categorical_model_columns": CATEGORICAL_MODEL_COLS,
        "train_policy": "year <= 2022",
        "validation_policy": "year >= 2023",
        "groups": group_summaries,
    }
    (GROUP_ROOT / "group_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for item in group_summaries:
        status = item["status"]
        if status == "SUCCESS":
            print(
                f"{item['group_name']}: rows train={item['train_rows_used']}, valid={item['validation_rows_used']}, "
                f"RMSE={item['rmse']:.6f}, R2={item['r2']:.6f}, MAPE={item['mape_percent']:.4f}%"
            )
        else:
            print(f"{item['group_name']}: FAIL - {item['failure_reason']}")
    return summary


if __name__ == "__main__":
    run()
