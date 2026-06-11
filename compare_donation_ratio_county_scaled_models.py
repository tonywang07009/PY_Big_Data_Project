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
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

TRAIN_PATH = ROOT / "data" / "county_zscore_split" / "train_scaled_all_counties.csv"
TEST_PATH = ROOT / "data" / "county_zscore_split" / "test_scaled_all_counties.csv"
OUTDIR = ROOT / "model_outputs" / "donation_ratio_county_scaled_model_comparison"
TARGET = "donation_ratio"
EPSILON = 1e-6

CATEGORICAL_FEATURES = ["county_label", "industry_label", "carrier_type_label"]
EXCLUDED_NUMERIC_FEATURES = {TARGET, *CATEGORICAL_FEATURES,"donation_count", "donation_amount"}
# 去除掉目標變數和類別特徵，剩下的就是數值特徵了

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month_num"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
    return out


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = add_time_features(pd.read_csv(TRAIN_PATH))
    valid = add_time_features(pd.read_csv(TEST_PATH))
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


def linear_pipe(numeric_features: list[str], model) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def tree_pipe(numeric_features: list[str], model) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
            ("cat", one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def build_models(numeric_features: list[str]) -> dict[str, Pipeline]:
    return {
        "Linear Regression": linear_pipe(numeric_features, LinearRegression()),
        "Ridge": linear_pipe(numeric_features, Ridge(alpha=1.0)),
        "Lasso": linear_pipe(numeric_features, Lasso(alpha=1e-4, max_iter=10000)),
        "Random Forest": tree_pipe(
            numeric_features,
            RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=1),
            # 找論文說明最佳參數
        ),
        "XGBoost": tree_pipe(
            numeric_features,
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=1,
            ),
        ),
    }

# 值越小預測越精準 對稱百分比誤差
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + EPSILON
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / denom))
    
# smape 的保護機制
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


def plot_comparison(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ranked = results.sort_values("r2", ascending=True)
    axes[0].barh(ranked["model"], ranked["r2"], color="#2C7FB8")
    axes[0].set_title("Validation R2 by model")
    axes[0].set_xlabel("R2")
    axes[0].grid(axis="x", alpha=0.2)

    ranked_rmse = results.sort_values("rmse", ascending=False)
    axes[1].barh(ranked_rmse["model"], ranked_rmse["rmse"], color="#D95F02")
    axes[1].set_title("Validation RMSE by model")
    axes[1].set_xlabel("RMSE")
    axes[1].grid(axis="x", alpha=0.2)

    fig.savefig(OUTDIR / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_scatter(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    ax.scatter(actual, predicted, s=10, alpha=0.25, color="#2C7FB8", edgecolors="none")
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], color="#D95F02", linewidth=1.4)
    ax.set_title(f"{model_name}: actual vs predicted")
    ax.set_xlabel("Actual donation_ratio")
    ax.set_ylabel("Predicted donation_ratio")
    ax.grid(alpha=0.2)
    fig.savefig(OUTDIR / "prediction_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    train_raw, valid_raw = load_frames() # 切分训练集和验证集
    numeric_features = get_numeric_features(train_raw) # 取特徵 (查是否有數值轉換)
    print(f"Numeric features: {numeric_features}")
    train = clean_frame(train_raw, numeric_features) # 清洗数据，去掉缺失值
    valid = clean_frame(valid_raw, numeric_features) # 驗證數據 去掉缺失值
    
    xtr = train[numeric_features + CATEGORICAL_FEATURES].copy() # 取出特征列，准备训练数据
    xva = valid[numeric_features + CATEGORICAL_FEATURES].copy()
    print(f"train rows: {train}, valid rows: {valid} xtr shape: {xtr.columns.tolist()}, xva shape: {xva.columns.tolist()}")
    ytr = train[TARGET].to_numpy() # 取出目标变量，转换为numpy数组
    yva = valid[TARGET].to_numpy()

    no_in_set = set(xtr.columns) - set(xva.columns)
    print(f"Features in training but not in validation: {no_in_set}")

    rows = []
    predictions = {}

    for name, pipe in build_models(numeric_features).items():
        print(name)
        pipe.fit(xtr, ytr)
        pred = pipe.predict(xva)
        predictions[name] = pred
        rows.append(
            {
                "model": name,
                "train_r2": float(r2_score(ytr, pipe.predict(xtr))),
                **metrics_for(yva, pred),
                # 把字典裡面的值展開放進去
            }
        )

    results = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    # index rewirte to 0,1,2... based on r2 ranking
    results.to_csv(OUTDIR / "model_comparison.csv", index=False)

    best_model = results.iloc[0]["model"]
    best_pred = predictions[best_model]
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
    pred_df["best_model"] = best_model
    pred_df["predicted_donation_ratio"] = best_pred
    pred_df["abs_error"] = np.abs(pred_df[TARGET] - pred_df["predicted_donation_ratio"])
    pred_df.to_csv(OUTDIR / "validation_predictions.csv", index=False)

    summary = {
        "pipeline": "county_scaled_model_comparison",
        "target": TARGET,
        "train_source": str(TRAIN_PATH),
        "validation_source": str(TEST_PATH),
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_rows_used": len(train),
        "validation_rows_used": len(valid),
        "best_model": best_model,
        "results": results.to_dict(orient="records"),
    }

    (OUTDIR / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_comparison(results)
    plot_best_scatter(yva, best_pred, best_model)

    print(f"Output directory: {OUTDIR}")
    print(results.to_string(index=False))
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
