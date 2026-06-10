# Gate 6 - Final Delivery

## Purpose

Convert model outputs into reviewable deliverables, operational priority scores, and formal gate verdict artifacts.

## Current Main Implementation

- Entrypoint: `run_all.py`
- Main code path:
  - `src/final_delivery.write_all()`
- Sub-components:
  - `write_gate_verdicts()`
  - `write_final_report()`
  - `write_dashboard()`
  - `write_tour_guide()`

## Current Delivery Flow

1. Read upstream clustering, modeling, SHAP, and readiness results.
2. Write `gate0` through `gate6` verdict JSON files.
3. Write the final Markdown report.
4. Write the static HTML dashboard.
5. Write the Tour Guide document.
6. Persist priority scores and reference them from the human-facing outputs.

## Current Main Output Artifacts

- `model_outputs/priority_scores.csv`
- `model_outputs/gate0_verdict.json` through `model_outputs/gate6_verdict.json`
- `reports/final_project_report.md`
- `reports/dashboard.html`
- `reports/tour_guide.md`

## Current Status

- Implemented and active.
- Current final delivery is still **CONDITIONAL** when requested scope counties are missing from the source data.
