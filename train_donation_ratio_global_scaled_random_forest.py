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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_PATH = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
OUTDIR = ROOT / "model_outputs" / "donation_ratio_global_scaled_random_forest"
TARGET = "donation_ratio"
EPSILON = 1e-6

CATEGORICAL_FEATURES = ["county_label", "industry_label", "carrier_type_label"]
EXCLUDED_NUMERIC_FEATURES = {TARGET, *CATEGORICAL_FEATURES, "donation_count", "donation_amount"}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["month"].astype(str) + "-01")
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] < 2024].copy()
    valid = df[df["year"] >= 2024].copy()
    return train, valid


def get_numeric_features(train: pd.DataFrame) -> list[str]:
    numeric_columns = train.select_dtypes(include="number").columns.tolist()
    return [col for col in numeric_columns if col not in EXCLUDED_NUMERIC_FEATURES]


def clean_frame(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    required = numeric_features + CATEGORICAL_FEATURES + [TARGET]
    return df.dropna(subset=required).copy()


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model(numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=1)
    return Pipeline([("pre", pre), ("model", model)])


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


def plot_performance(pred_df: pd.DataFrame, metrics: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    scatter_ax = axes[0]
    actual = pred_df[TARGET].to_numpy()
    predicted = pred_df["predicted_donation_ratio"].to_numpy()
    scatter_ax.scatter(actual, predicted, s=10, alpha=0.25, color="#2C7FB8", edgecolors="none")
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    scatter_ax.plot([low, high], [low, high], color="#D95F02", linewidth=1.4)
    scatter_ax.set_title("Actual vs Predicted")
    scatter_ax.set_xlabel("Actual donation_ratio")
    scatter_ax.set_ylabel("Predicted donation_ratio")
    scatter_ax.grid(alpha=0.2)
    scatter_ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                f"RMSE: {metrics['rmse']:.6f}",
                f"MAE: {metrics['mae']:.6f}",
                f"sMAPE: {metrics['smape']:.3f}",
                f"R2: {metrics['r2']:.6f}",
            ]
        ),
        transform=scatter_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )

    monthly_ax = axes[1]
    monthly = (
        pred_df.groupby("month_num", sort=True)
        .agg(
            actual_donation_ratio=(TARGET, "mean"),
            predicted_donation_ratio=("predicted_donation_ratio", "mean"),
        )
        .reset_index()
    )
    monthly_ax.plot(
        monthly["month_num"],
        monthly["actual_donation_ratio"],
        marker="o",
        linewidth=2,
        color="#1B9E77",
        label="Actual mean",
    )
    monthly_ax.plot(
        monthly["month_num"],
        monthly["predicted_donation_ratio"],
        marker="o",
        linewidth=2,
        color="#D95F02",
        label="Predicted mean",
    )
    monthly_ax.set_title("2024 Monthly Mean Comparison")
    monthly_ax.set_xlabel("Month")
    monthly_ax.set_ylabel("Mean donation_ratio")
    monthly_ax.set_xticks(monthly["month_num"])
    monthly_ax.grid(alpha=0.2)
    monthly_ax.legend()

    fig.suptitle("Donation Ratio Global-Scaled Random Forest Performance", fontsize=16)
    fig.savefig(OUTDIR / "performance_result.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_raw, valid_raw = split_train_test(df)
    numeric_features = get_numeric_features(train_raw)

    train = clean_frame(train_raw, numeric_features)
    valid = clean_frame(valid_raw, numeric_features)

    xtr = train[numeric_features + CATEGORICAL_FEATURES].copy()
    xva = valid[numeric_features + CATEGORICAL_FEATURES].copy()
    ytr = train[TARGET].to_numpy()
    yva = valid[TARGET].to_numpy()

    model = build_model(numeric_features)
    model.fit(xtr, ytr)
    preds = model.predict(xva)
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
    pred_df["ape_percent"] = pred_df["abs_error"] / np.maximum(np.abs(pred_df[TARGET]), EPSILON) * 100.0
    pred_df.to_csv(OUTDIR / "validation_predictions.csv", index=False)

    summary = {
        "pipeline": "global_scaled_random_forest",
        "target": TARGET,
        "train_source": str(RAW_PATH),
        "validation_source": str(RAW_PATH),
        "split_policy": "train uses year < 2024; test uses year >= 2024",
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_rows_used": len(train),
        "validation_rows_used": len(valid),
        **metrics,
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_performance(pred_df, metrics)

    print(f"Output directory: {OUTDIR}")
    print(f"Train rows used: {len(train)}")
    print(f"Validation rows used: {len(valid)}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"sMAPE: {metrics['smape']:.6f}")
    print(f"epsilon-MAPE: {metrics['epsilon_mape']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")


if __name__ == "__main__":
    main()
