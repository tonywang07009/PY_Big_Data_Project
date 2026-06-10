# MecDonate System Overview

MecDonate is a Big Data Analytics final project for county-level invoice donation ratio prediction in Taiwan. This file is the short routing entry point for Codex and human reviewers. Detailed policies, gate plans, and working notes live under `agent_doc/`.

## Core Identity

- **Research source**: `source_materials/group_six.md`
- **Target**: `donation_ratio = donated_count / total_count`
- **County scope**: Taipei, Taichung, Kaohsiung, Lienchiang, Hualien, Yunlin
- **Time policy**: train on 2019-2022 and test on 2023 onward; never shuffle time-series rows
- **Final outputs**: clustering, model metrics, SHAP interpretation, priority scores, final report

## Language Rule

- Persisted project documents are written in English.
- Questions asked directly to the human are written in Traditional Chinese.
- Human decisions are recorded in English under `agent_doc/decisions/`.

## Read Router

Read only the files needed for the current task:

| Need | Read |
|---|---|
| Project summary | `agent_doc/project_brief.md` |
| QFD mapping | `agent_doc/qfd_matrix.md` |
| Diver/Counter and human brainstorming rules | `agent_doc/review_protocol.md` |
| Testing and acceptance policy | `agent_doc/testing_workflow.md` |
| Documentation paths and naming | `agent_doc/documentation_policy.md` |
| Human-approved decisions | `agent_doc/decisions/decision_log.md` |
| Gate implementation details | `agent_doc/gates/gate{N}_*.md` |
| Tool design notes | `agent_doc/tool_designs/` |

## Gate Router

| Gate | Topic | Route |
|---:|---|---|
| 0 | Data readiness | `agent_doc/gates/gate0_data_readiness.md` |
| 1 | Preprocessing | `agent_doc/gates/gate1_preprocessing.md` |
| 2 | Feature engineering | `agent_doc/gates/gate2_feature_engineering.md` |
| 3 | Model build | `agent_doc/gates/gate3_model_build.md` |
| 4 | Validation | `agent_doc/gates/gate4_validation.md` |
| 5 | Interpretability | `agent_doc/gates/gate5_interpretability.md` |
| 6 | Final delivery | `agent_doc/gates/gate6_final_delivery.md` |

## Directory Overview

```text
source_materials/      source proposal and original reference files
data/raw/              read-only data inputs
data/county_zscore_split/ standalone county-based scaling outputs
src/                   executable modules used by run_all.py
model_outputs/         machine-readable outputs, metrics, models, SHAP files
reports/               human-facing deliverables
agent_doc/             agent routing, decisions, gate specs, templates, memory
```

## Operating Rules

- Use QFD to connect research needs to engineering contracts.
- Use Diver/Counter review for every micro-task.
- After each micro-task, check project fit, runtime stability, and human maintainability.
- Start a human brainstorming discussion in Traditional Chinese when a high-impact ambiguity appears.
- Keep this file short. Add detailed rules to `agent_doc/`, then route to them from here.
