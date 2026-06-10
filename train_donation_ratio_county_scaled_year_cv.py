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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RAW_PATH = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
OUTDIR = ROOT / "model_outputs" / "donation_ratio_county_scaled_year_cv"
TARGET = "donation_ratio"
EPSILON = 1e-6

CATEGORICAL_FEATURES = ["county_label", "industry_label", "carrier_type_label"]
NON_SCALE_COLUMNS = {
    "month",
    "date",
    "year",
    "county_label",
    "industry_label",
    "carrier_type_label",
    "donation_ratio",
}


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


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["month"].astype(str) + "-01")
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    return df


def get_scale_columns(train_df: pd.DataFrame) -> list[str]:
    numeric_columns = train_df.select_dtypes(include="number").columns.tolist()
    return [col for col in numeric_columns if col not in NON_SCALE_COLUMNS]


def split_by_county(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(county_label): group_df.copy()
        for county_label, group_df in df.groupby("county_label", sort=True)
    }


def scale_each_county_group(
    train_groups: dict[int, pd.DataFrame],
    valid_groups: dict[int, pd.DataFrame],
    scale_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled_train_groups = []
    scaled_valid_groups = []

    for county_label, train_group in train_groups.items():
        valid_group = valid_groups.get(county_label)
        if valid_group is None or valid_group.empty:
            continue

        scaler = StandardScaler()
        train_scaled = train_group.copy()
        valid_scaled = valid_group.copy()

        train_scaled[scale_columns] = scaler.fit_transform(train_group[scale_columns])
        valid_scaled[scale_columns] = scaler.transform(valid_group[scale_columns])

        scaled_train_groups.append(train_scaled)
        scaled_valid_groups.append(valid_scaled)

    if not scaled_train_groups or not scaled_valid_groups:
        raise ValueError("No overlapping county groups were available for scaling.")

    train_scaled_df = pd.concat(scaled_train_groups, axis=0).sort_index()
    valid_scaled_df = pd.concat(scaled_valid_groups, axis=0).sort_index()
    return train_scaled_df, valid_scaled_df


def clean_frame(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    required = numeric_features + CATEGORICAL_FEATURES + [TARGET]
    return df.dropna(subset=required).copy()


def encode_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
        [train[numeric_features].reset_index(drop=True), train_cat.reset_index(drop=True)],
        axis=1,
    )
    valid_x = pd.concat(
        [valid[numeric_features].reset_index(drop=True), valid_cat.reset_index(drop=True)],
        axis=1,
    )
    return train_x, valid_x, train_x.columns.tolist()


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + EPSILON
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / denom))


def epsilon_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), EPSILON)
    return float(np.mean(100.0 * np.abs(y_true - y_pred) / denom))


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "epsilon_mape": epsilon_mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_fold(df: pd.DataFrame, valid_year: int) -> tuple[dict[str, float], pd.DataFrame]:
    train_raw = df[df["year"] < valid_year].copy()
    valid_raw = df[df["year"] == valid_year].copy()

    if train_raw.empty or valid_raw.empty:
        raise ValueError(f"Fold year {valid_year} does not have both train and validation rows.")

    scale_columns = get_scale_columns(train_raw)
    train_scaled, valid_scaled = scale_each_county_group(
        split_by_county(train_raw),
        split_by_county(valid_raw),
        scale_columns,
    )

    numeric_features = [
        col
        for col in train_scaled.select_dtypes(include="number").columns.tolist()
        if col not in {TARGET, *CATEGORICAL_FEATURES}
    ]

    train = clean_frame(train_scaled, numeric_features)
    valid = clean_frame(valid_scaled, numeric_features)

    train_x, valid_x, model_columns = encode_features(train, valid, numeric_features)
    ytr = train[TARGET].to_numpy()
    yva = valid[TARGET].to_numpy()

    model = build_model()
    model.fit(train_x, ytr)
    preds = model.predict(valid_x[model_columns])
    metrics = metrics_for(yva, preds)

    pred_df = valid[
        [
            "month",
            "date",
            "year",
            "month_num",
            "county_label",
            "industry_label",
            "carrier_type_label",
            TARGET,
        ]
    ].copy()
    pred_df["predicted_donation_ratio"] = preds
    pred_df["abs_error"] = np.abs(pred_df[TARGET] - pred_df["predicted_donation_ratio"])
    pred_df["fold_year"] = valid_year

    fold_summary = {
        "validation_year": valid_year,
        "train_years": sorted(train_raw["year"].unique().tolist()),
        "train_rows_used": len(train),
        "validation_rows_used": len(valid),
        "scale_columns": scale_columns,
        "numeric_features": numeric_features,
        "model_columns_after_encoding": model_columns,
        **metrics,
    }
    return fold_summary, pred_df


def plot_cv_performance(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].plot(metrics_df["validation_year"], metrics_df["rmse"], marker="o", linewidth=2, color="#D95F02")
    axes[0].set_title("RMSE by Validation Year")
    axes[0].set_xlabel("Validation year")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.2)

    axes[1].plot(metrics_df["validation_year"], metrics_df["r2"], marker="o", linewidth=2, color="#1B9E77")
    axes[1].set_title("R2 by Validation Year")
    axes[1].set_xlabel("Validation year")
    axes[1].set_ylabel("R2")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Donation Ratio Rolling Year Validation", fontsize=15)
    fig.savefig(OUTDIR / "performance_result.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    years = sorted(df["year"].unique().tolist())
    valid_years = years[1:]

    fold_summaries = []
    fold_predictions = []

    for valid_year in valid_years:
        summary, pred_df = run_fold(df, valid_year)
        fold_summaries.append(summary)
        fold_predictions.append(pred_df)
        print(
            f"Fold {valid_year}: "
            f"train_years={summary['train_years']} "
            f"rows(train/valid)=({summary['train_rows_used']}, {summary['validation_rows_used']}) "
            f"RMSE={summary['rmse']:.6f} R2={summary['r2']:.6f}"
        )

    metrics_df = pd.DataFrame(fold_summaries)
    pred_df = pd.concat(fold_predictions, ignore_index=True)

    metrics_df.to_csv(OUTDIR / "metrics_by_fold.csv", index=False)
    pred_df.to_csv(OUTDIR / "validation_predictions_by_fold.csv", index=False)

    aggregate = {
        "pipeline": "county_scaled_year_cv",
        "target": TARGET,
        "source_dataset": str(RAW_PATH),
        "validation_years": valid_years,
        "mean_rmse": float(metrics_df["rmse"].mean()),
        "mean_mae": float(metrics_df["mae"].mean()),
        "mean_smape": float(metrics_df["smape"].mean()),
        "mean_epsilon_mape": float(metrics_df["epsilon_mape"].mean()),
        "mean_r2": float(metrics_df["r2"].mean()),
        "best_r2_year": int(metrics_df.loc[metrics_df["r2"].idxmax(), "validation_year"]),
        "worst_r2_year": int(metrics_df.loc[metrics_df["r2"].idxmin(), "validation_year"]),
    }
    (OUTDIR / "metrics_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_cv_performance(metrics_df)

    print(f"Output directory: {OUTDIR}")
    print(f"Validation years: {valid_years}")
    print(f"Mean RMSE: {aggregate['mean_rmse']:.6f}")
    print(f"Mean MAE: {aggregate['mean_mae']:.6f}")
    print(f"Mean sMAPE: {aggregate['mean_smape']:.6f}")
    print(f"Mean epsilon-MAPE: {aggregate['mean_epsilon_mape']:.6f}")
    print(f"Mean R2: {aggregate['mean_r2']:.6f}")


if __name__ == "__main__":
    main()
