"""
MecDonate - end-to-end pipeline.

Runs the full framework from the proposal:
  1. build county x month feature matrix
  2. K-Means clustering (Elbow / Silhouette / PCA)
  3. regression model comparison (time-based split, RMSE & R2)
  4. SHAP feature-importance analysis

Usage:  python run_all.py
All artefacts are written to ./outputs/
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd

import features
import clustering
import model
import shap_analysis

OUT = "outputs"
RAW = "encoded_ml_dataset.csv"


def main():
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60, "\n[1/4] Building county-month feature matrix")
    cm = features.build_county_month(RAW)
    cm.to_csv(f"{OUT}/county_month_matrix.csv", index=False)
    print(f"      matrix shape = {cm.shape}")

    print("=" * 60, "\n[2/4] K-Means clustering")
    clustering.run(cm, OUT)

    print("=" * 60, "\n[3/4] Regression model comparison")
    model.run(cm, OUT)

    print("=" * 60, "\n[4/4] SHAP analysis")
    shap_analysis.run(cm, OUT)

    print("=" * 60, "\nDone. Artefacts in ./outputs/")


if __name__ == "__main__":
    main()
