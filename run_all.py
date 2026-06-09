"""
MecDonate - end-to-end pipeline.

Runs the full framework from the proposal:
  1. build scoped county x month feature matrix
  2. K-Means clustering (Elbow / Silhouette / PCA)
  3. regression model comparison (time-based split, RMSE & R2)
  4. Ridge coefficients, XGBoost SHAP, priority scores
  5. gate verdicts, final report, static dashboard, Tour Guide

Usage:  python run_all.py
Formal artefacts are written to ./model_outputs/ and ./reports/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

import features
import clustering
import model
import shap_analysis
import final_delivery
from project_config import filter_scope_counties, scope_readiness

OUT = Path("model_outputs")
REPORTS = Path("reports")
METRICS = OUT / "metrics"
SHAP = OUT / "shap"
RAW = "data/raw/encoded_ml_dataset.csv"


def main():
    OUT.mkdir(exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    SHAP.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    raw_counties = pd.read_csv(RAW, usecols=["county_label"])["county_label"].unique()
    readiness = scope_readiness(raw_counties)
    missing = [m["county_name"] for m in readiness["missing_scope_counties"]]

    print("=" * 60, "\n[1/5] Building scoped county-month feature matrix")
    full_cm = features.build_county_month(RAW)
    cm = filter_scope_counties(full_cm)
    cm.to_csv(OUT / "county_month_matrix.csv", index=False)
    print(f"      matrix shape = {cm.shape}")
    if missing:
        print(f"      data-readiness warning: missing requested county data = {missing}")

    print("=" * 60, "\n[2/5] K-Means clustering")
    clustering.run(cm, OUT)
    clustering.run_v2(cm, OUT)

    print("=" * 60, "\n[3/5] Regression model comparison and priority scoring")
    model_summary = model.run(cm, METRICS, priority_path=OUT / "priority_scores.csv")

    print("=" * 60, "\n[4/5] SHAP analysis")
    shap_importance = shap_analysis.run(cm, SHAP)

    print("=" * 60, "\n[5/5] Gate verdicts, report, dashboard, and Tour Guide")
    cluster_profiles = pd.read_csv(OUT / "cluster_assignments.csv")
    gates = final_delivery.write_all(
        OUT,
        REPORTS,
        cm,
        readiness,
        cluster_profiles.set_index("county_label"),
        model_summary,
        shap_importance,
    )
    print("      gates =", ", ".join(f"Gate {g['gate']} {g['verdict']}" for g in gates))

    print("=" * 60, "\nDone. Formal artefacts in ./model_outputs/ and ./reports/")


if __name__ == "__main__":
    main()
