# MecDonate Decision Log

This file records human-approved project decisions that affect research direction, system design, documentation, or maintainability.

## Decision 001 - Project Control Document Rewrite

- **Date**: 2026-06-04
- **Source**: Human discussion based on `source_materials/group_six.md`
- **Decision**: Rewrite `project.md` as the central English project control document.
- **Rationale**: English documentation improves searchability, reduces token usage, and keeps future agent work consistent.
- **Impact**:
  - `project.md` becomes the primary routing overview.
  - `source_materials/group_six.md` remains the research proposal source.
  - Future persisted project documents should be written in English.

## Decision 002 - Human Question Language

- **Date**: 2026-06-04
- **Decision**: Questions asked directly to the human must use Traditional Chinese.
- **Rationale**: The human collaboration interface should remain easy to read and review.
- **Impact**:
  - Codex must use Traditional Chinese when asking for clarification, brainstorming, or approval.
  - The recorded decision summary remains English in this file or the relevant gate document.

## Decision 003 - Documentation Split

- **Date**: 2026-06-04
- **Decision**: Use `agent_doc/` for project governance documents, decisions, gate plans, templates, and agent memory.
- **Rationale**: Human review documents and agent memory logs serve different maintenance purposes.
- **Impact**:
  - `agent_doc/gates/` stores gate implementation discussion.
  - `agent_doc/decisions/` stores human-approved decisions.
  - `agent_doc/agent_memory/` stores daily execution traces, blockers, and next actions.

## Decision 004 - Diver / Counter Review Model

- **Date**: 2026-06-04
- **Decision**: Every micro-task should use a Diver/Counter review model.
- **Rationale**: Diver/Counter review supports brainstorming, counter-checking, and system engineering discipline.
- **Impact**:
  - Diver proposes task method, assumptions, model_outputs, and risks.
  - Counter checks project fit, operational stability, maintainability, leakage risk, and artifact validity.
  - Each micro-task must end with `PASS`, `CONDITIONAL`, or `FAIL`.

## Decision 005 - Task-Level Engineering Checks

- **Date**: 2026-06-04
- **Decision**: Every micro-task must check project fit, operational stability, and human maintainability before the next task begins.
- **Rationale**: Small task-level checks improve testing efficiency, runtime reliability, and long-term maintainability.
- **Impact**:
  - Each task receives `fit_score`, `stability_score`, and `maintainability_score`.
  - Any score below 3 blocks progress.
  - Any score equal to 3 requires conditional documentation and human discussion before high-impact continuation.

## Decision 006 - Router-Based Documentation

- **Date**: 2026-06-04
- **Decision**: Keep `project.md` as a short system overview and route detailed content to `agent_doc/`.
- **Rationale**: A compact router follows better agent documentation practice, reduces repeated token loading, and improves searchability.
- **Impact**:
  - `project.md` should stay short and only route to detailed documents.
  - QFD, testing, review protocol, gate plans, templates, and decisions live under `agent_doc/`.
  - Future gate discussions should update the relevant file under `agent_doc/gates/`.

## Decision 007 - Human-Centered Folder Names

- **Date**: 2026-06-04
- **Decision**: Rename folders to improve readability and reduce confusion.
- **Rationale**: Folder names should reflect user intent and maintenance purpose.
- **Impact**:
  - `source_materials/` was renamed to `source_materials/`.
  - `tools/` was renamed to `pipeline_tools/`.
  - `model_outputs/` was renamed to `model_outputs/`.
  - `agent_doc/tool_designs/` content moved to `agent_doc/agent_doc/tool_designss/`.
  - `test_log/` content moved into `agent_doc/`.
