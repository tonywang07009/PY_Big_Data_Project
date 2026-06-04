"""
MecDonate - Step 3: regression model comparison.

Predicts the monthly county donation_ratio. A time-based split prevents
leakage: 2019-2022 for training, 2023-2024 for testing. Five models are
compared (Linear, Ridge, Lasso, Random Forest, XGBoost) on RMSE and R2.
Linear models receive z-score-standardised + one-hot-encoded inputs;
tree models receive the raw numeric matrix.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from features import FEATURE_COLS, TARGET

CAT_COLS = ["county_type"]


def split(cm: pd.DataFrame):
    cm = cm.dropna(subset=FEATURE_COLS).copy()
    train = cm[cm["year"] <= 2022]
    test = cm[cm["year"] >= 2023]
    return train, test


def _linear_pipe(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), FEATURE_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def _tree_pipe(model):
    pre = ColumnTransformer([
        ("num", "passthrough", FEATURE_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def build_models():
    return {
        "Linear Regression": _linear_pipe(LinearRegression()),
        "Ridge": _linear_pipe(Ridge(alpha=1.0)),
        "Lasso": _linear_pipe(Lasso(alpha=1e-4, max_iter=10000)),
        "Random Forest": _tree_pipe(
            RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
        "XGBoost": _tree_pipe(
            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                         subsample=0.9, colsample_bytree=0.9,
                         random_state=42, n_jobs=-1)),
    }


def run(cm: pd.DataFrame, outdir: str = "outputs"):
    train, test = split(cm)
    Xtr, ytr = train, train[TARGET].values
    Xte, yte = test, test[TARGET].values

    rows, preds = [], {}
    for name, pipe in build_models().items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        preds[name] = pred
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        rows.append({
            "model": name,
            "test_RMSE": rmse,
            "test_R2": float(r2_score(yte, pred)),
            "train_R2": float(r2_score(ytr, pipe.predict(Xtr))),
        })

    res = pd.DataFrame(rows).sort_values("test_R2", ascending=False)
    res.to_csv(f"{outdir}/model_comparison.csv", index=False)

    # --- bar chart of test R2 / RMSE ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].barh(res["model"], res["test_R2"], color="tab:blue")
    ax[0].set(title="Test R2 by model", xlabel="R2")
    ax[0].invert_yaxis()
    ax[1].barh(res["model"], res["test_RMSE"], color="tab:orange")
    ax[1].set(title="Test RMSE by model", xlabel="RMSE")
    ax[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{outdir}/model_comparison.png", dpi=130)
    plt.close(fig)

    # --- predicted vs actual for the best model ---
    best = res.iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, preds[best], alpha=0.4, s=18)
    lims = [min(yte.min(), preds[best].min()), max(yte.max(), preds[best].max())]
    ax.plot(lims, lims, "r--")
    ax.set(title=f"{best}: predicted vs actual (2023-2024)",
           xlabel="actual donation_ratio", ylabel="predicted")
    fig.tight_layout()
    fig.savefig(f"{outdir}/prediction_scatter.png", dpi=130)
    plt.close(fig)

    with open(f"{outdir}/model_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": rows, "best_model": best,
                   "n_train": len(train), "n_test": len(test)}, f,
                  ensure_ascii=False, indent=2)

    print(res.to_string(index=False))
    print(f"[model] best = {best}")
    return res, best


if __name__ == "__main__":
    cm = pd.read_csv("outputs/county_month_matrix.csv")
    run(cm)
