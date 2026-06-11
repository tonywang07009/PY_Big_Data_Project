"""
Root-cause analysis for XGBoost pipeline choices.

This script is diagnostic only. Its results helped identify why the old
raw-row approach was unstable, but it does not generate the formal Gate 0-7
artifacts. The formal mainline is `train_donation_count_xgboost_ensemble.py`.

This script compares three pipeline families on the full dataset:
  1. county-month baseline
  2. raw-row baseline
  3. hybrid county-month with richer raw-derived structural features

It evaluates two targets:
  - donation_ratio
  - donation_count

Outputs:
  - model_outputs/root_cause_analysis/data_profile.json
  - model_outputs/root_cause_analysis/pipeline_results.csv
  - reports/xgboost_root_cause_analysis.md
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "model_outputs" / "root_cause_analysis"
REPORT = ROOT / "reports" / "xgboost_root_cause_analysis.md"
RAW = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
sys.path.insert(0, str(ROOT / "src"))

import features  # noqa: E402

EPSILON = 1e-6


@dataclass
class EvalResult:
    pipeline: str
    target: str
    transform: str
    train_rows: int
    valid_rows: int
    feature_count: int
    rmse: float
    mae: float
    smape: float
    epsilon_mape: float
    r2: float
    note: str


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


def add_target_lags(frame: pd.DataFrame, target: str, key_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    key = out[key_cols + ["date", target]].copy()
    for lag in (1, 2, 3):
        shifted = key.copy()
        shifted["date"] = shifted["date"] + pd.DateOffset(months=lag)
        shifted = shifted.rename(columns={target: f"{target}_lag{lag}"})
        out = out.merge(
            shifted[key_cols + ["date", f"{target}_lag{lag}"]],
            on=key_cols + ["date"],
            how="left",
        )
    return out


def profile_raw_data(df: pd.DataFrame) -> dict:
    county_month_sizes = df.groupby(["county_label", "month"]).size()
    const_cols = [
        "housing_burden",
        "price_income_ratio",
        "total_county_invoice_count",
        "total_county_invoice_amount",
        "donation_count",
        "donation_ratio",
    ]
    constancy = {
        col: {
            "county_month_groups_with_multiple_values": int(
                (df.groupby(["county_label", "month"])[col].nunique() > 1).sum()
            )
        }
        for col in const_cols
    }
    targets = {}
    for target in ["donation_ratio", "donation_count"]:
        series = df[target]
        targets[target] = {
            "zero_rows": int((series == 0).sum()),
            "zero_ratio": float((series == 0).mean()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "skew": float(series.skew()),
        }
    return {
        "raw_rows": int(len(df)),
        "county_month_groups": int(len(county_month_sizes)),
        "avg_rows_per_county_month": float(county_month_sizes.mean()),
        "min_rows_per_county_month": int(county_month_sizes.min()),
        "max_rows_per_county_month": int(county_month_sizes.max()),
        "constancy": constancy,
        "targets": targets,
    }


def build_county_month_base(raw_df: pd.DataFrame) -> pd.DataFrame:
    cm = features.build_county_month(str(RAW))
    cm = add_target_lags(cm, "donation_count", ["county_label"])
    return cm


def _hybrid_stats(sub: pd.DataFrame) -> pd.Series:
    total = sub["industry_invoice_count"].sum()
    carrier_total = sub.loc[sub["carrier_type_label"] == 0, "industry_invoice_count"].sum()
    by_ind = sub.groupby("industry_label")["industry_invoice_count"].sum().sort_values(ascending=False)
    shares = by_ind / by_ind.sum() if by_ind.sum() else by_ind
    entropy = float(-(shares * np.log(shares + 1e-12)).sum()) if len(shares) else 0.0
    top1 = float(shares.iloc[0]) if len(shares) >= 1 else 0.0
    top3 = float(shares.iloc[:3].sum()) if len(shares) >= 1 else 0.0
    hhi = float((shares ** 2).sum()) if len(shares) else 0.0
    return pd.Series(
        {
            "carrier_usage_ratio": carrier_total / total if total else np.nan,
            "n_active_industries": int((by_ind > 0).sum()),
            "industry_hhi": hhi,
            "industry_entropy": entropy,
            "top1_industry_share": top1,
            "top3_industry_share": top3,
        }
    )


def build_hybrid_county_month(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_fields(raw_df)
    grp = df.groupby(["county_label", "month"], sort=True)
    const_cols = [
        "housing_burden",
        "price_income_ratio",
        "total_county_invoice_count",
        "total_county_invoice_amount",
        "donation_count",
        "donation_ratio",
        "date",
        "year",
        "month_num",
        "month_sin",
        "month_cos",
    ]
    base = grp[const_cols].first()
    agg = grp.apply(_hybrid_stats)
    cm = base.join(agg).reset_index()
    cm = add_target_lags(cm, "donation_ratio", ["county_label"])
    cm = add_target_lags(cm, "donation_count", ["county_label"])
    return cm


def build_raw_row_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_fields(raw_df)
    base = (
        df[["county_label", "month", "date", "donation_ratio", "donation_count"]]
        .drop_duplicates(["county_label", "month"])
        .reset_index(drop=True)
    )
    ratio_lags = add_target_lags(base[["county_label", "month", "date", "donation_ratio"]], "donation_ratio", ["county_label"])
    count_lags = add_target_lags(base[["county_label", "month", "date", "donation_count"]], "donation_count", ["county_label"])
    merged = df.merge(
        ratio_lags[["county_label", "month", "donation_ratio_lag1", "donation_ratio_lag2", "donation_ratio_lag3"]],
        on=["county_label", "month"],
        how="left",
    )
    merged = merged.merge(
        count_lags[["county_label", "month", "donation_count_lag1", "donation_count_lag2", "donation_count_lag3"]],
        on=["county_label", "month"],
        how="left",
    )
    return merged


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + EPSILON
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / denom))


def epsilon_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), EPSILON)
    return float(np.mean(100.0 * np.abs(y_true - y_pred) / denom))


def fit_eval(
    df: pd.DataFrame,
    pipeline_name: str,
    target: str,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    transform: str = "identity",
    note: str = "",
) -> EvalResult:
    categorical_cols = categorical_cols or []
    needed = feature_cols + categorical_cols + [target, "year"]
    clean = df.dropna(subset=needed).copy()
    train = clean[clean["year"] <= 2022].copy()
    valid = clean[clean["year"] >= 2023].copy()
    if train.empty or valid.empty:
        raise ValueError(f"{pipeline_name}/{target} produced empty split after dropna.")

    if categorical_cols:
        train_cat = pd.get_dummies(train[categorical_cols].astype(str), columns=categorical_cols, prefix=categorical_cols)
        valid_cat = pd.get_dummies(valid[categorical_cols].astype(str), columns=categorical_cols, prefix=categorical_cols)
        valid_cat = valid_cat.reindex(columns=train_cat.columns, fill_value=0)
        Xtr = pd.concat([train[feature_cols].reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
        Xva = pd.concat([valid[feature_cols].reset_index(drop=True), valid_cat.reset_index(drop=True)], axis=1)
    else:
        Xtr = train[feature_cols].copy()
        Xva = valid[feature_cols].copy()

    ytr = train[target].to_numpy()
    yva = valid[target].to_numpy()
    if transform == "log1p":
        model_ytr = np.log1p(ytr)
    else:
        model_ytr = ytr

    model = build_model()
    model.fit(Xtr, model_ytr)
    pred = model.predict(Xva)
    if transform == "log1p":
        pred = np.expm1(pred)
        pred = np.maximum(pred, 0.0)

    return EvalResult(
        pipeline=pipeline_name,
        target=target,
        transform=transform,
        train_rows=len(train),
        valid_rows=len(valid),
        feature_count=Xtr.shape[1],
        rmse=float(math.sqrt(mean_squared_error(yva, pred))),
        mae=float(mean_absolute_error(yva, pred)),
        smape=smape(yva, pred),
        epsilon_mape=epsilon_mape(yva, pred),
        r2=float(r2_score(yva, pred)),
        note=note,
    )


def build_report(profile: dict, results: pd.DataFrame) -> str:
    best_ratio = results[results["target"] == "donation_ratio"].sort_values("smape").iloc[0]
    best_count = results[results["target"] == "donation_count"].sort_values("smape").iloc[0]
    display = results.copy()
    float_cols = ["rmse", "mae", "smape", "epsilon_mape", "r2"]
    for col in float_cols:
        display[col] = display[col].map(lambda x: f"{x:.6f}")
    table_cols = [
        "pipeline",
        "target",
        "transform",
        "train_rows",
        "valid_rows",
        "feature_count",
        "rmse",
        "mae",
        "smape",
        "epsilon_mape",
        "r2",
    ]
    header = "| " + " | ".join(table_cols) + " |"
    divider = "| " + " | ".join(["---"] * len(table_cols)) + " |"
    body = [
        "| " + " | ".join(str(row[col]) for col in table_cols) + " |"
        for _, row in display[table_cols].iterrows()
    ]
    lines = [
        "# XGBoost Root Cause Analysis",
        "",
        "## Data Profile",
        "",
        f"- Raw rows: `{profile['raw_rows']:,}`",
        f"- County-month groups: `{profile['county_month_groups']:,}`",
        f"- Avg rows per county-month: `{profile['avg_rows_per_county_month']:.2f}`",
        f"- donation_ratio zero-row ratio: `{profile['targets']['donation_ratio']['zero_ratio']:.4f}`",
        f"- donation_count zero-row ratio: `{profile['targets']['donation_count']['zero_ratio']:.4f}`",
        "",
        "## Pipeline Results",
        "",
        header,
        divider,
        *body,
        "",
        "## Stepwise Decision Table",
        "",
        "| Step | Question | Evidence | Decision |",
        "| --- | --- | --- | --- |",
        f"| 1 | Should we use raw-row directly? | Raw data repeats the same county-month target across ~{profile['avg_rows_per_county_month']:.1f} rows on average. Raw-row pipelines did not dominate the best metric rows consistently. | Do not default to raw-row direct modeling; treat it as a weighting-heavy alternative. |",
        f"| 2 | Which target is more stable for XGBoost? | Best `donation_ratio` pipeline: `{best_ratio['pipeline']}` with sMAPE `{best_ratio['smape']:.4f}`. Best `donation_count` pipeline: `{best_count['pipeline']}` with sMAPE `{best_count['smape']:.4f}`. | Prefer the target with the lower and more stable error across multiple pipelines. |",
        f"| 3 | Is hybrid county-month worth it? | Hybrid keeps county-month target semantics while adding raw-derived structure. Compare its result rows against plain county-month and raw-row. | If hybrid beats both baseline variants on the chosen target, use hybrid as the next implementation baseline. |",
        "| 4 | Should normalization be a first-order concern? | All tested pipelines use XGBoost, which is usually insensitive to monotonic scaling. | Focus on grain, target, lag design, and structural features before normalization. |",
        "| 5 | Should we split by time first or by group? | A single full-data time split keeps comparisons fair across target/grain choices. | Finalize a full-data baseline first, then revisit grouping only if it improves holdout metrics. |",
        "",
        "## Recommended Next Step",
        "",
        f"- Use `{best_ratio['pipeline']}` or `{best_count['pipeline']}` as the next baseline candidate depending on business preference between ratio and count targets.",
        "- If `donation_ratio` remains unstable under percentage-based metrics, compare it against a `donation_count`-first pipeline and derive ratio later.",
        "- If raw-row does not clearly win, keep the target at county-month granularity and enrich structural features rather than repeating the same target across many rows.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW)
    profile = profile_raw_data(raw_df)
    (OUTDIR / "data_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    county_month = build_county_month_base(raw_df)
    hybrid = build_hybrid_county_month(raw_df)
    raw_frame = build_raw_row_frame(raw_df)

    results = []

    # county-month baseline
    ratio_features_cm = list(features.FEATURE_COLS) + ["year", "month_num"]
    count_features_cm = [
        "housing_burden",
        "price_income_ratio",
        "log_total_count",
        "log_total_amount",
        "avg_invoice_value",
        "carrier_usage_ratio",
        "n_active_industries",
        "industry_hhi",
        "donation_seasonal",
        "month_sin",
        "month_cos",
        "donation_count_lag1",
        "donation_count_lag2",
        "donation_count_lag3",
        "year",
        "month_num",
    ]
    results.append(fit_eval(county_month, "county_month_baseline", "donation_ratio", ratio_features_cm, transform="identity", note="Existing county-month matrix"))
    results.append(fit_eval(county_month, "county_month_baseline", "donation_count", count_features_cm, transform="identity", note="Existing county-month matrix"))
    results.append(fit_eval(county_month, "county_month_baseline", "donation_count", count_features_cm, transform="log1p", note="Existing county-month matrix with log1p target"))

    # raw-row baseline
    ratio_features_raw = [
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
    count_features_raw = [
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
    results.append(fit_eval(raw_frame, "raw_row_baseline", "donation_ratio", ratio_features_raw, categorical_cols=["industry_label", "carrier_type_label"], transform="identity", note="Raw rows with one-hot categories"))
    results.append(fit_eval(raw_frame, "raw_row_baseline", "donation_count", count_features_raw, categorical_cols=["industry_label", "carrier_type_label"], transform="identity", note="Raw rows with one-hot categories"))
    results.append(fit_eval(raw_frame, "raw_row_baseline", "donation_count", count_features_raw, categorical_cols=["industry_label", "carrier_type_label"], transform="log1p", note="Raw rows with one-hot categories and log1p target"))

    # hybrid county-month
    ratio_features_hybrid = [
        "housing_burden",
        "price_income_ratio",
        "total_county_invoice_count",
        "total_county_invoice_amount",
        "carrier_usage_ratio",
        "n_active_industries",
        "industry_hhi",
        "industry_entropy",
        "top1_industry_share",
        "top3_industry_share",
        "month_sin",
        "month_cos",
        "donation_ratio_lag1",
        "donation_ratio_lag2",
        "donation_ratio_lag3",
        "year",
        "month_num",
    ]
    count_features_hybrid = [
        "housing_burden",
        "price_income_ratio",
        "total_county_invoice_count",
        "total_county_invoice_amount",
        "carrier_usage_ratio",
        "n_active_industries",
        "industry_hhi",
        "industry_entropy",
        "top1_industry_share",
        "top3_industry_share",
        "month_sin",
        "month_cos",
        "donation_count_lag1",
        "donation_count_lag2",
        "donation_count_lag3",
        "year",
        "month_num",
    ]
    results.append(fit_eval(hybrid, "hybrid_county_month", "donation_ratio", ratio_features_hybrid, transform="identity", note="County-month target with richer structural stats"))
    results.append(fit_eval(hybrid, "hybrid_county_month", "donation_count", count_features_hybrid, transform="identity", note="County-month target with richer structural stats"))
    results.append(fit_eval(hybrid, "hybrid_county_month", "donation_count", count_features_hybrid, transform="log1p", note="County-month target with richer structural stats and log1p target"))

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(["target", "smape", "rmse"])
    results_df.to_csv(OUTDIR / "pipeline_results.csv", index=False)
    REPORT.write_text(build_report(profile, results_df), encoding="utf-8")
    print(results_df.to_string(index=False))
    print(f"\nWrote report: {REPORT}")


if __name__ == "__main__":
    main()
