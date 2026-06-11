"""
Legacy filtered validation experiment.

This script is not the formal project mainline. The formal path is
`train_donation_count_xgboost_ensemble.py`, which uses all available counties,
county-month `donation_count`, monthly expanding walk-forward validation,
county-wise numeric scaling, calibration, and historical baseline blending.

Training counties: 1, 2, 15, 17, 18
Validation counties: every other county in the engineered county-month matrix

Requested output files:
  - data/raw/filtter_trainning.csv
  - data/raw/filtter_vaildation.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import features  # noqa: E402
from project_config import COUNTY_NAMES  # noqa: E402

RAW = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
DATA_RAW = ROOT / "data" / "raw"
OUTDIR = ROOT / "model_outputs" / "filtered_validation"

TRAIN_COUNTIES = {1, 2, 15, 17, 18}
FEATURE_COLS = [
    "housing_burden",
    "donation_ratio_lag1",
    "industry_hhi",
    "donation_ratio_lag3",
    "donation_ratio_lag2",
]
TARGET = "donation_ratio"
META_COLS = ["county_label", "month", TARGET]
TRAIN_FILE = DATA_RAW / "filtter_trainning.csv"
VALID_FILE = DATA_RAW / "filtter_vaildation.csv"
EPSILON = 1e-6


def build_filtered_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cm = features.build_county_month(str(RAW))
    selected = cm[META_COLS + FEATURE_COLS].copy()

    train = selected[selected["county_label"].isin(TRAIN_COUNTIES)].copy()
    valid = selected[~selected["county_label"].isin(TRAIN_COUNTIES)].copy()

    train.to_csv(TRAIN_FILE, index=False)
    valid.to_csv(VALID_FILE, index=False)
    return train, valid


def load_model_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv(TRAIN_FILE)
    valid = pd.read_csv(VALID_FILE)

    train_before = len(train)
    valid_before = len(valid)
    train = train.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    valid = valid.dropna(subset=FEATURE_COLS + [TARGET]).copy()

    dropped = {
        "train_rows_before_dropna": train_before,
        "train_rows_used": len(train),
        "train_rows_dropped_missing_features": train_before - len(train),
        "validation_rows_before_dropna": valid_before,
        "validation_rows_used": len(valid),
        "validation_rows_dropped_missing_features": valid_before - len(valid),
    }
    if train.empty or valid.empty:
        raise ValueError("Filtered training or validation set is empty after dropping missing features.")
    return train, valid, dropped


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


def add_county_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["county_name"] = out["county_label"].map(COUNTY_NAMES).fillna(out["county_label"].astype(str))
    return out


def diagnostics(mape_pct: float) -> list[str]:
    if mape_pct <= 5.0:
        return []
    return [
        "Validation counties are entirely unseen during training, so distribution shift across counties is the main risk.",
        "The requested five-feature set is narrow; adding price_income_ratio, carrier_usage_ratio, or seasonality terms is the first tuning direction.",
        "Lag features lose rows around time gaps and series starts; more usable history features or gap-aware imputation may stabilize validation.",
        "Tune XGBoost hyperparameters such as max_depth, n_estimators, learning_rate, subsample, and colsample_bytree for cross-county generalization.",
    ]


def run() -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    build_filtered_tables()
    train, valid, dropped = load_model_frames()

    model = build_model()
    model.fit(train[FEATURE_COLS], train[TARGET].to_numpy())
    preds = model.predict(valid[FEATURE_COLS])

    scored = add_county_name(valid[META_COLS + FEATURE_COLS].copy())
    scored["predicted_donation_ratio"] = preds
    scored["abs_error"] = np.abs(scored[TARGET] - scored["predicted_donation_ratio"])
    scored["ape_percent"] = (
        scored["abs_error"] / np.maximum(np.abs(scored[TARGET]), EPSILON) * 100.0
    )

    mape_pct = float(scored["ape_percent"].mean())
    verdict = "SUCCESS" if mape_pct <= 5.0 else "FAIL"
    diag = diagnostics(mape_pct)

    scored.to_csv(OUTDIR / "xgboost_validation_predictions.csv", index=False)
    feature_gain = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_gain.to_csv(OUTDIR / "xgboost_feature_importance.csv", index=False)

    summary = {
        "train_counties": sorted(TRAIN_COUNTIES),
        "validation_counties": sorted(int(c) for c in valid["county_label"].unique().tolist()),
        "feature_columns": FEATURE_COLS,
        "target": TARGET,
        "metric": "MAPE_percent",
        "mape_percent": mape_pct,
        "threshold_percent": 5.0,
        "epsilon": EPSILON,
        "verdict": verdict,
        **dropped,
        "diagnostics": diag,
    }
    (OUTDIR / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Training CSV: {TRAIN_FILE}")
    print(f"Validation CSV: {VALID_FILE}")
    print(f"Validation rows used: {len(valid)}")
    print(f"MAPE: {mape_pct:.4f}%")
    print(f"Verdict: {verdict}")
    if diag:
        print("Adjustment directions:")
        for item in diag:
            print(f"- {item}")

    return summary


if __name__ == "__main__":
    run()
