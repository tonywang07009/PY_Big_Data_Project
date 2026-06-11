"""
MecDonate formal end-to-end pipeline.

Formal mainline:
  1. train county-month XGBoost ensemble with monthly walk-forward
  2. audit ensemble predictions, SHAP outputs, and Gate 7 acceptance
  3. regenerate Gate 0 through Gate 7 verdicts and human-facing reports

Usage: python run_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import analyze_donation_count_xgboost_ensemble
import final_delivery
import train_donation_count_xgboost_ensemble

OUT = Path("model_outputs")
REPORTS = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    print("=" * 60, "\n[1/3] Training formal monthly XGBoost ensemble")
    train_donation_count_xgboost_ensemble.main()

    print("=" * 60, "\n[2/3] Auditing formal ensemble and Gate 7 acceptance")
    analyze_donation_count_xgboost_ensemble.main()

    print("=" * 60, "\n[3/3] Gate verdicts, final report, and dashboard")
    gates = final_delivery.write_all(OUT, REPORTS)
    print("      gates =", ", ".join(f"Gate {g['gate']} {g['verdict']}" for g in gates))

    print("=" * 60, "\nDone. Formal artifacts are in ./model_outputs/ and ./reports/")


if __name__ == "__main__":
    main()
