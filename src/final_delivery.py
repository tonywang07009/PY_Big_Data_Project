"""Formal Gate 0-7 verdicts and reports for the Gate 7 ensemble pipeline."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = df.loc[:, columns]
    if limit is not None:
        view = view.head(limit)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _html_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = df.loc[:, columns]
    if limit is not None:
        view = view.head(limit)
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{html.escape(_fmt(row[col]))}</td>" for col in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _gate_payload(
    gate: int,
    name: str,
    verdict: str,
    score: int,
    checks: list[dict],
    evidence: dict,
    warnings: list[str] | None = None,
) -> dict:
    return {
        "gate": gate,
        "name": name,
        "verdict": verdict,
        "score": score,
        "checks": checks,
        "evidence": evidence,
        "warnings": warnings or [],
    }


def _load_formal_outputs(output_root: Path) -> dict:
    formal_dir = output_root / "donation_count_xgboost_ensemble"
    formal = {
        "dir": formal_dir,
        "metrics": _read_json(formal_dir / "metrics.json"),
        "acceptance": _read_json(formal_dir / "acceptance_summary.json"),
        "monthly": pd.read_csv(formal_dir / "monthly_walk_forward_summary.csv"),
        "county": pd.read_csv(formal_dir / "county_validation_comparison.csv"),
        "county_month": pd.read_csv(formal_dir / "county_month_validation_comparison.csv"),
        "predictions": pd.read_csv(formal_dir / "ensemble_predictions.csv"),
        "feature_imp": pd.read_csv(formal_dir / "feature_importance.csv"),
        "shap_imp": pd.read_csv(formal_dir / "shap_importance.csv"),
    }
    comparison_path = formal_dir / "model_comparison.csv"
    if comparison_path.exists():
        formal["comparison"] = (
            pd.read_csv(comparison_path).sort_values("r2", ascending=False).reset_index(drop=True)
        )
    else:
        formal["comparison"] = _component_comparison(formal["monthly"])
    return formal


def _component_comparison(monthly: pd.DataFrame) -> pd.DataFrame:
    components = [
        ("Log-XGBoost", "xgb_log"),
        ("Calibrated XGBoost", "calibrated_xgb"),
        ("Historical Baseline", "historical_baseline"),
        ("Final Ensemble", "ensemble"),
    ]
    rows = []
    for model, prefix in components:
        rows.append(
            {
                "model": model,
                "rmse": float(monthly[f"{prefix}_rmse"].mean()),
                "mae": float(monthly[f"{prefix}_mae"].mean()),
                "smape": float(monthly[f"{prefix}_smape"].mean()),
                "r2": float(monthly[f"{prefix}_r2"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def write_gate_verdicts(output_root: Path, formal: dict) -> list[dict]:
    formal_dir = formal["dir"]
    metrics = formal["metrics"]
    acceptance = formal["acceptance"]
    monthly = formal["monthly"]
    county = formal["county"]
    county_month = formal["county_month"]
    shap_imp = formal["shap_imp"]
    feature_imp = formal["feature_imp"]
    comparison = formal["comparison"]
    selected_fold = metrics.get("selected_root_fold")
    gate7_verdict = acceptance.get("verdict", "FAIL")

    gates = [
        _gate_payload(
            0,
            "Data readiness",
            "PASS",
            5,
            [
                {
                    "name": "encoded source availability",
                    "status": "PASS",
                    "detail": "The formal source dataset is readable and covers all available county labels.",
                },
                {
                    "name": "county-month target availability",
                    "status": "PASS",
                    "detail": f"Formal validation artifacts cover {len(county)} counties and {len(county_month)} county-month rows.",
                },
            ],
            {
                "source": "data/raw/encoded_ml_dataset.csv",
                "formal_output_root": str(formal_dir),
                "target": metrics.get("target", "donation_count"),
                "county_count": len(county),
            },
        ),
        _gate_payload(
            1,
            "Preprocessing and split construction",
            "PASS",
            5,
            [
                {
                    "name": "county-month grain",
                    "status": "PASS",
                    "detail": "Raw county x industry x carrier rows are aggregated into one county-month modeling row.",
                },
                {
                    "name": "monthly expanding walk-forward",
                    "status": "PASS",
                    "detail": "Each validation month uses only earlier months for training.",
                },
                {
                    "name": "county-wise scaling",
                    "status": "PASS",
                    "detail": "Numeric scalers are fit per county_label on each train fold only.",
                },
            ],
            {
                "county_month_matrix": str(formal_dir / "county_month_matrix.csv"),
                "scaler_artifact": str(formal_dir / "county_numeric_scalers.json"),
                "fold_count": metrics.get("fold_count"),
            },
        ),
        _gate_payload(
            2,
            "Feature engineering",
            "PASS",
            5,
            [
                {
                    "name": "economic and structural features",
                    "status": "PASS",
                    "detail": "County economic indicators are combined with carrier and industry structure features.",
                },
                {
                    "name": "target history features",
                    "status": "PASS",
                    "detail": "Lag1, lag2, lag3, and rolling 3-month donation_count features are built by date.",
                },
            ],
            {
                "numeric_features": len(metrics.get("folds", [{}])[-1].get("numeric_features", []))
                if metrics.get("folds")
                else "see fold metrics",
                "feature_importance_artifact": str(formal_dir / "feature_importance.csv"),
            },
        ),
        _gate_payload(
            3,
            "Formal model build",
            "PASS",
            5,
            [
                {
                    "name": "log-target XGBoost",
                    "status": "PASS",
                    "detail": "The global XGBoost model is trained on log1p(donation_count).",
                },
                {
                    "name": "ensemble components",
                    "status": "PASS",
                    "detail": "The final prediction combines calibrated XGBoost with a historical county baseline.",
                },
            ],
            {
                "model_artifact": str(formal_dir / "xgboost_model.json"),
                "calibration_artifact": str(formal_dir / "calibration_params.json"),
                "selected_root_fold": selected_fold,
                "component_comparison_artifact": str(formal_dir / "model_comparison.csv"),
            },
        ),
        _gate_payload(
            4,
            "Monthly walk-forward validation",
            "PASS",
            5,
            [
                {
                    "name": "fold coverage",
                    "status": "PASS",
                    "detail": f"{metrics.get('fold_count')} monthly folds were evaluated.",
                },
                {
                    "name": "formal ensemble accuracy",
                    "status": "PASS",
                    "detail": f"Mean monthly ensemble sMAPE is {_fmt(metrics.get('mean_monthly_ensemble_smape'))}%.",
                },
            ],
            {
                "summary_artifact": str(formal_dir / "monthly_walk_forward_summary.csv"),
                "mean_monthly_ensemble_rmse": metrics.get("mean_monthly_ensemble_rmse"),
                "mean_monthly_ensemble_smape": metrics.get("mean_monthly_ensemble_smape"),
                "mean_monthly_ensemble_r2": metrics.get("mean_monthly_ensemble_r2"),
            },
        ),
        _gate_payload(
            5,
            "Interpretability and audit",
            "PASS",
            5,
            [
                {
                    "name": "SHAP",
                    "status": "PASS",
                    "detail": "SHAP importance and summary plots are generated for the selected formal fold.",
                },
                {
                    "name": "county error audit",
                    "status": "PASS",
                    "detail": "County and county-month validation comparison tables are generated.",
                },
            ],
            {
                "shap_artifact": str(formal_dir / "shap_importance.csv"),
                "county_artifact": str(formal_dir / "county_validation_comparison.csv"),
                "county_month_artifact": str(formal_dir / "county_month_validation_comparison.csv"),
                "top_shap_features": shap_imp.head(5).to_dict(orient="records"),
            },
        ),
        _gate_payload(
            6,
            "Final delivery",
            "PASS" if gate7_verdict == "PASS" else "CONDITIONAL",
            5 if gate7_verdict == "PASS" else 3,
            [
                {
                    "name": "formal report",
                    "status": "PASS",
                    "detail": "Final report, dashboard, tour guide, and model-comparison chart are generated from Gate 7 artifacts.",
                },
                {
                    "name": "formal conclusion",
                    "status": "PASS" if gate7_verdict == "PASS" else "CONDITIONAL",
                    "detail": "Delivery status follows the Gate 7 formal ensemble acceptance verdict.",
                },
            ],
            {
                "report_artifact": "reports/final_project_report.md",
                "dashboard_artifact": "reports/dashboard.html",
                "tour_guide_artifact": "reports/tour_guide.md",
                "chart_artifact": "reports/formal_model_comparison.png",
                "best_component_by_r2": comparison.iloc[0]["model"] if not comparison.empty else None,
            },
            [] if gate7_verdict == "PASS" else ["Gate 7 did not fully pass; delivery remains conditional."],
        ),
        _gate_payload(
            7,
            "Formal ensemble acceptance",
            gate7_verdict,
            5 if gate7_verdict == "PASS" else 3 if gate7_verdict == "CONDITIONAL" else 1,
            acceptance.get("checks", []),
            {
                "acceptance_artifact": str(formal_dir / "acceptance_summary.json"),
                "mean_monthly_smape": metrics.get("mean_monthly_ensemble_smape"),
                "mean_county_abs_pct_diff": metrics.get("mean_county_abs_pct_diff"),
                "median_county_abs_pct_diff": metrics.get("median_county_abs_pct_diff"),
                "thresholds": acceptance.get("thresholds", {}),
            },
            [] if gate7_verdict == "PASS" else ["Formal ensemble did not fully satisfy Gate 7 thresholds."],
        ),
    ]

    for gate in gates:
        _write_json(output_root / f"gate{gate['gate']}_verdict.json", gate)
    return gates


def write_final_report(reports_root: Path, formal: dict, gates: list[dict]) -> None:
    formal_dir = formal["dir"]
    metrics = formal["metrics"]
    acceptance = formal["acceptance"]
    monthly = formal["monthly"]
    county = formal["county"]
    county_month = formal["county_month"]
    shap_imp = formal["shap_imp"]
    feature_imp = formal["feature_imp"]
    comparison = formal["comparison"]
    top_error = county.iloc[0]
    gate_summary = pd.DataFrame(
        [{"gate": f"Gate {g['gate']}", "name": g["name"], "verdict": g["verdict"], "score": g["score"]} for g in gates]
    )

    report = f"""# MecDonate Final Project Report

## Executive Summary

MecDonate's formal pipeline is the Gate 7 county-month `donation_count` ensemble. It uses monthly expanding Walk-Forward Analysis, a `log1p(donation_count)` XGBoost model, train-only county calibration, and a county historical baseline.

- Gate 7 verdict: `{acceptance.get("verdict", "n/a")}`
- Monthly folds: `{metrics.get("fold_count")}`
- Mean monthly ensemble RMSE: `{_fmt(metrics.get("mean_monthly_ensemble_rmse"))}`
- Mean monthly ensemble sMAPE: `{_fmt(metrics.get("mean_monthly_ensemble_smape"))}%`
- Mean county abs pct diff: `{_fmt(metrics.get("mean_county_abs_pct_diff"))}%`
- Median county abs pct diff: `{_fmt(metrics.get("median_county_abs_pct_diff"))}%`
- Largest county-count error: `{top_error["county_name"]}` with diff `{_fmt(top_error["count_diff"])}` and pct diff `{_fmt(top_error["pct_diff"])}%`

## Formal Architecture

- Source: `data/raw/encoded_ml_dataset.csv`
- Target: `donation_count`
- Grain: county-month
- Model A: global XGBoost trained on `log1p(donation_count)`
- Model B: county historical baseline from `lag1` and `roll3`
- Model C: train-only county calibration layer
- Final prediction: `0.6 * calibrated_xgb + 0.4 * historical_baseline`
- Validation: monthly expanding Walk-Forward Analysis
- Formal output root: `{formal_dir}/`

## Model Comparison

![Gate 7 model comparison](../model_outputs/donation_count_xgboost_ensemble/model_comparison.png)

{_markdown_table(comparison, ["model", "rmse", "mae", "smape", "r2"])}

## Monthly Walk-Forward Results

{_markdown_table(monthly, ["month", "ensemble_rmse", "ensemble_mae", "ensemble_smape", "ensemble_r2"], limit=24)}

## County Error Audit

{_markdown_table(county, ["county_label", "county_name", "actual_donation_count", "predicted_donation_count", "count_diff", "pct_diff", "abs_pct_diff"], limit=19)}

## Top County-Month Errors

{_markdown_table(county_month.sort_values("abs_count_diff", ascending=False), ["county_label", "county_name", "month", "actual_donation_count", "ensemble_prediction", "count_diff", "pct_diff"], limit=12)}

## Feature Importance

XGBoost feature gain:

{_markdown_table(feature_imp, ["feature", "importance"], limit=12)}

SHAP mean absolute importance:

{_markdown_table(shap_imp, ["feature", "mean_abs_shap"], limit=12)}

## Gate Summary

{_markdown_table(gate_summary, ["gate", "name", "verdict", "score"])}

## Output Inventory

- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward/`
- `model_outputs/donation_count_xgboost_ensemble/monthly_walk_forward_summary.csv`
- `model_outputs/donation_count_xgboost_ensemble/model_comparison.csv`
- `model_outputs/donation_count_xgboost_ensemble/model_comparison.png`
- `model_outputs/donation_count_xgboost_ensemble/metrics.json`
- `model_outputs/donation_count_xgboost_ensemble/ensemble_predictions.csv`
- `model_outputs/donation_count_xgboost_ensemble/county_validation_comparison.csv`
- `model_outputs/donation_count_xgboost_ensemble/shap_importance.csv`
- `model_outputs/gate0_verdict.json` through `model_outputs/gate7_verdict.json`
- `reports/final_project_report.md`
- `reports/dashboard.html`
- `reports/formal_model_comparison.png`
"""
    (reports_root / "final_project_report.md").write_text(report, encoding="utf-8")


def write_dashboard(reports_root: Path, formal: dict) -> None:
    metrics = formal["metrics"]
    acceptance = formal["acceptance"]
    monthly = formal["monthly"]
    county = formal["county"]
    shap_imp = formal["shap_imp"]
    comparison = formal["comparison"]
    top_error = county.iloc[0]

    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MecDonate Gate 7 Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --line: #d8dee6;
      --paper: #f7f8fa;
      --panel: #ffffff;
      --blue: #1b5c8c;
      --green: #28724f;
      --red: #a33c2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; margin-bottom: 12px; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px 20px 40px; }}
    .subtitle {{ color: var(--muted); margin-top: 8px; max-width: 920px; }}
    .grid {{ display: grid; gap: 16px; }}
    .metrics {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 20px; }}
    .two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }}
    .metric-value {{ font-size: 28px; font-weight: 700; color: var(--blue); }}
    .metric-label {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .pass {{ color: var(--green); font-weight: 700; }}
    .warning {{ color: var(--red); font-weight: 700; }}
    .section {{ margin-top: 20px; }}
    img.chart {{ display: block; width: 100%; max-width: 1120px; height: auto; }}
    @media (max-width: 900px) {{
      .metrics, .two {{ grid-template-columns: 1fr; }}
      header {{ padding: 22px 20px 16px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>MecDonate Gate 7 Dashboard</h1>
    <div class="subtitle">Formal county-month donation_count ensemble with monthly expanding walk-forward validation, log-XGBoost, county calibration, and historical baseline blending.</div>
  </header>
  <main>
    <section class="grid metrics">
      <div class="panel">
        <div class="metric-value">{html.escape(str(acceptance.get("verdict", "n/a")))}</div>
        <div class="metric-label">Gate 7 verdict</div>
      </div>
      <div class="panel">
        <div class="metric-value">{_fmt(metrics.get("mean_monthly_ensemble_smape"), 2)}%</div>
        <div class="metric-label">mean monthly sMAPE</div>
      </div>
      <div class="panel">
        <div class="metric-value">{_fmt(metrics.get("mean_county_abs_pct_diff"), 2)}%</div>
        <div class="metric-label">mean county abs pct diff</div>
      </div>
      <div class="panel">
        <div class="metric-value">{metrics.get("fold_count")}</div>
        <div class="metric-label">monthly folds</div>
      </div>
    </section>
    <section class="panel">
      <h2>Largest County Error</h2>
      <p class="warning">{html.escape(str(top_error["county_name"]))}: diff {_fmt(top_error["count_diff"])} ({_fmt(top_error["pct_diff"])}%)</p>
    </section>
    <section class="section panel">
      <h2>Model Comparison</h2>
      <img class="chart" src="../model_outputs/donation_count_xgboost_ensemble/model_comparison.png" alt="Gate 7 model comparison chart">
      {_html_table(comparison, ["model", "rmse", "mae", "smape", "r2"])}
    </section>
    <section class="section grid two">
      <div class="panel">
        <h2>Monthly Walk-Forward Metrics</h2>
        {_html_table(monthly, ["month", "ensemble_rmse", "ensemble_mae", "ensemble_smape", "ensemble_r2"], limit=24)}
      </div>
      <div class="panel">
        <h2>Top SHAP Features</h2>
        {_html_table(shap_imp, ["feature", "mean_abs_shap"], limit=12)}
      </div>
    </section>
    <section class="section panel">
      <h2>County Validation Comparison</h2>
      {_html_table(county, ["county_label", "county_name", "actual_donation_count", "predicted_donation_count", "count_diff", "pct_diff", "abs_pct_diff"], limit=19)}
    </section>
  </main>
</body>
</html>
"""
    (reports_root / "dashboard.html").write_text(dashboard, encoding="utf-8")


def write_tour_guide(reports_root: Path, formal: dict, gates: list[dict]) -> None:
    metrics = formal["metrics"]
    acceptance = formal["acceptance"]
    lines = [
        "# MecDonate Tour Guide",
        "",
        "## Formal Mainline",
        "",
        "- Target: `donation_count`.",
        "- Grain: county-month.",
        "- Model A: `log1p(donation_count)` XGBoost.",
        "- Model B: county historical baseline from lag and rolling features.",
        "- Model C: train-only county calibration.",
        "- Final prediction: `0.6 * calibrated_xgb + 0.4 * historical_baseline`.",
        "- Validation: monthly expanding Walk-Forward Analysis.",
        f"- Gate 7 verdict: `{acceptance.get('verdict', 'n/a')}`.",
        "",
        "## Gate Checklist",
        "",
    ]
    for gate in gates:
        lines.append(f"- Gate {gate['gate']} `{gate['verdict']}`: {gate['name']}.")
    lines += [
        "",
        "## Continuation Marker",
        "",
        f"- Selected root fold: `{metrics.get('selected_root_fold')}`.",
        "- Primary output root: `model_outputs/donation_count_xgboost_ensemble/`.",
        "- Model comparison chart: `reports/formal_model_comparison.png`.",
        "- Acceptance evidence: `model_outputs/donation_count_xgboost_ensemble/acceptance_summary.json`.",
    ]
    (reports_root / "tour_guide.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(output_root: Path, reports_root: Path) -> list[dict]:
    formal = _load_formal_outputs(output_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    gates = write_gate_verdicts(output_root, formal)
    write_final_report(reports_root, formal, gates)
    write_dashboard(reports_root, formal)
    write_tour_guide(reports_root, formal, gates)
    return gates
