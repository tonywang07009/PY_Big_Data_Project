"""
MecDonate - Step 4: SHAP feature-importance analysis.

SHAP is computed on the XGBoost model (the best-performing tree model,
which SHAP's TreeExplainer supports exactly) to interpret which features
drive county donation-ratio predictions.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from features import FEATURE_COLS, TARGET
from model import split, CAT_COLS


def run(cm: pd.DataFrame, outdir: str = "outputs"):
    train, test = split(cm)

    # design matrix (numeric features + one-hot county_type)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[CAT_COLS])
    cat_names = list(ohe.get_feature_names_out(CAT_COLS))

    def design(df):
        return np.hstack([df[FEATURE_COLS].values, ohe.transform(df[CAT_COLS])])

    feat_names = FEATURE_COLS + cat_names
    Xtr = design(train)
    Xte = design(test)

    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                         subsample=0.9, colsample_bytree=0.9,
                         random_state=42, n_jobs=-1)
    model.fit(Xtr, train[TARGET].values)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)

    # mean |SHAP| importance table
    imp = (pd.DataFrame({"feature": feat_names,
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    imp.to_csv(f"{outdir}/shap_importance.csv", index=False)

    # beeswarm summary
    plt.figure()
    shap.summary_plot(sv, Xte, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    # bar importance
    plt.figure(figsize=(8, 5))
    plt.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="tab:purple")
    plt.title("SHAP feature importance (mean |SHAP|)")
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(f"{outdir}/shap_importance.png", dpi=130)
    plt.close()

    print("[shap] top features:")
    print(imp.head(8).to_string(index=False))
    return imp


if __name__ == "__main__":
    cm = pd.read_csv("outputs/county_month_matrix.csv")
    run(cm)
