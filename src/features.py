"""
MecDonate - Step 1: Build the county x month feature matrix.

The raw `encoded_ml_dataset.csv` is at the
    county x industry x carrier-type x month
granularity. Several columns (housing_burden, price_income_ratio,
total_county_invoice_count/amount, donation_count, donation_ratio) are
already constant within a county-month -- these are the county-month
economic indicators and the target. The industry/carrier rows are
aggregated into behavioural features (carrier-usage ratio, industry
diversity / concentration), and lag + STL-seasonal features are added so
the matrix matches the proposal's "county-level time-series matrix".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# --- county type mapping (per proposal: metropolitan / remote / agricultural) ---
# Codes come from label_mapping_table_county.csv
METROPOLITAN = {1, 2, 3, 11, 14, 18}      # 6 special municipalities
REMOTE = {4, 15}                          # 台東縣, 花蓮縣 (east / mountainous)
# everything else -> agricultural / general county


def county_type(code: int) -> str:
    if code in METROPOLITAN:
        return "metropolitan"
    if code in REMOTE:
        return "remote"
    return "agricultural"


def build_county_month(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # carrier_type_label: 0 = uses e-carrier, 1 = no carrier.
    df["carrier_count"] = np.where(df["carrier_type_label"] == 0,
                                   df["industry_invoice_count"], 0)

    grp = df.groupby(["county_label", "month"], sort=True)

    # county-month constant columns -> take first
    const_cols = ["housing_burden", "price_income_ratio",
                  "total_county_invoice_count", "total_county_invoice_amount",
                  "donation_count", "donation_ratio"]
    base = grp[const_cols].first()

    # behavioural / structural aggregates from the industry rows
    agg = grp.apply(_county_month_aggregates, include_groups=False)

    cm = base.join(agg).reset_index()

    # time fields
    cm["date"] = pd.to_datetime(cm["month"] + "-01")
    cm["year"] = cm["date"].dt.year
    cm["month_num"] = cm["date"].dt.month
    cm["month_sin"] = np.sin(2 * np.pi * cm["month_num"] / 12)
    cm["month_cos"] = np.cos(2 * np.pi * cm["month_num"] / 12)

    # derived economic ratios
    cm["avg_invoice_value"] = (cm["total_county_invoice_amount"]
                               / cm["total_county_invoice_count"])
    cm["log_total_count"] = np.log1p(cm["total_county_invoice_count"])
    cm["log_total_amount"] = np.log1p(cm["total_county_invoice_amount"])

    # county type + one-hot county
    cm["county_type"] = cm["county_label"].map(county_type)

    cm = cm.sort_values(["county_label", "date"]).reset_index(drop=True)

    # NOTE: the series is split into two blocks (2019-01..2020-01 and
    # 2022-01..2024-12) with a 23-month gap. STL needs a regular index, so
    # we reindex each county to a continuous monthly range, interpolate the
    # gap, fit STL, then map the seasonal component back by *date* (the
    # fabricated gap months are simply not joined back).
    cm["donation_seasonal"] = np.nan
    cm["donation_trend"] = np.nan
    for code, sub in cm.groupby("county_label"):
        s = sub.set_index("date")["donation_ratio"].asfreq("MS").interpolate()
        res = STL(s, period=12, robust=True).fit()
        seasonal = res.seasonal.reindex(sub["date"].values)
        trend = res.trend.reindex(sub["date"].values)
        cm.loc[sub.index, "donation_seasonal"] = seasonal.values
        cm.loc[sub.index, "donation_trend"] = trend.values

    # --- date-based lag & rolling features of the target -----------------
    # Joining on (county, date - L months) makes cross-gap lags NaN instead
    # of silently using a value 23 months away.
    key = cm[["county_label", "date", "donation_ratio"]]
    for lag in (1, 2, 3):
        shifted = key.copy()
        shifted["date"] = shifted["date"] + pd.DateOffset(months=lag)
        shifted = shifted.rename(columns={"donation_ratio": f"donation_ratio_lag{lag}"})
        cm = cm.merge(shifted[["county_label", "date", f"donation_ratio_lag{lag}"]],
                      on=["county_label", "date"], how="left")
    cm["donation_ratio_roll3"] = cm[[f"donation_ratio_lag{l}" for l in (1, 2, 3)]].mean(axis=1)

    return cm


def _county_month_aggregates(sub: pd.DataFrame) -> pd.Series:
    tot = sub["industry_invoice_count"].sum()
    carrier_ratio = sub["carrier_count"].sum() / tot if tot else np.nan

    # industry concentration: HHI over industry shares of invoice count
    by_ind = sub.groupby("industry_label")["industry_invoice_count"].sum()
    shares = by_ind / by_ind.sum() if by_ind.sum() else by_ind
    hhi = float((shares ** 2).sum())

    return pd.Series({
        "carrier_usage_ratio": carrier_ratio,
        "n_active_industries": int((by_ind > 0).sum()),
        "industry_hhi": hhi,
    })


FEATURE_COLS = [
    # economic indicators
    "housing_burden", "price_income_ratio",
    "log_total_count", "log_total_amount", "avg_invoice_value",
    # behavioural / structural
    "carrier_usage_ratio", "n_active_industries", "industry_hhi",
    "donation_seasonal",
    # seasonality
    "month_sin", "month_cos",
    # lag / rolling behaviour
    "donation_ratio_lag1", "donation_ratio_lag2", "donation_ratio_lag3",
    "donation_ratio_roll3",
]
TARGET = "donation_ratio"


if __name__ == "__main__":
    cm = build_county_month("encoded_ml_dataset.csv")
    cm.to_csv("outputs/county_month_matrix.csv", index=False)
    print("county-month matrix:", cm.shape)
    print(cm[["county_label", "month", "donation_ratio",
              "carrier_usage_ratio", "n_active_industries"]].head())
