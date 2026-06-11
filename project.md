# MecDonate System Overview

MecDonate is a Big Data Analytics final project for county-level invoice donation-count prediction in Taiwan. This file is the short routing entry point for Codex and human reviewers. Detailed policies, gate plans, and working notes live under `agent_doc/`.

## Core Identity

- **Research source**: `source_materials/group_six.md`
- **Formal target**: `donation_count`
- **Formal scope**: all `county_label` values available in `data/raw/encoded_ml_dataset.csv`
- **Formal model**: Gate 7 county-month XGBoost ensemble
- **Formal validation policy**: monthly expanding Walk-Forward Analysis
- **Scaling policy**: numeric features are standardized per `county_label` using train-fold rows only
- **Final outputs**: monthly walk-forward metrics, model comparison chart, county validation audit, SHAP interpretation, gate verdicts, final report, dashboard

## Formal Gate 7 Model

- Model A: global XGBoost trained on `log1p(donation_count)`
- Model B: county historical baseline from `lag1` and `roll3`
- Model C: train-only county calibration
- Final prediction: `0.6 * calibrated_xgb + 0.4 * historical_baseline`

## Language Rule

- Persisted project documents are written in English.
- Questions asked directly to the human are written in Traditional Chinese.
- Human decisions are recorded in English under `agent_doc/decisions/`.

## Read Router

| Need | Read |
|---|---|
| Project summary | `agent_doc/project_brief.md` |
| Gate 7 architecture | `agent_doc/xgboost_ensemble_architecture.md` |
| QFD mapping | `agent_doc/qfd_matrix.md` |
| Diver/Counter and human brainstorming rules | `agent_doc/review_protocol.md` |
| Testing and acceptance policy | `agent_doc/testing_workflow.md` |
| Documentation paths and naming | `agent_doc/documentation_policy.md` |
| Human-approved decisions | `agent_doc/decisions/decision_log.md` |
| Gate implementation details | `agent_doc/gates/gate{N}_*.md` |

## Gate Router

| Gate | Topic | Route |
|---:|---|---|
| 0 | Data readiness | `agent_doc/gates/gate0_data_readiness.md` |
| 1 | Preprocessing and split construction | `agent_doc/gates/gate1_preprocessing.md` |
| 2 | Feature engineering | `agent_doc/gates/gate2_feature_engineering.md` |
| 3 | Formal model build | `agent_doc/gates/gate3_model_build.md` |
| 4 | Monthly walk-forward validation | `agent_doc/gates/gate4_validation.md` |
| 5 | Interpretability and audit | `agent_doc/gates/gate5_interpretability.md` |
| 6 | Final delivery | `agent_doc/gates/gate6_final_delivery.md` |
| 7 | Formal ensemble acceptance | `agent_doc/gates/gate7_ensemble_acceptance.md` |

## Directory Overview

```text
source_materials/      source proposal and original reference files
data/raw/              read-only data inputs
pipeline_tools/        placeholder modules for future extraction
model_outputs/         machine-readable artifacts, metrics, models, SHAP files
reports/               human-facing deliverables
agent_doc/             agent routing, decisions, gate specs, templates, memory
```

## Operating Rules

- Treat `train_donation_count_xgboost_ensemble.py` as the formal training entry point.
- Treat `analyze_donation_count_xgboost_ensemble.py` as the formal audit and Gate 7 acceptance entry point.
- Treat `run_all.py` as the end-to-end orchestrator.
- Keep older ratio, clustering, grouping, raw-row, and enhanced scripts clearly labeled as legacy or diagnostic paths.
