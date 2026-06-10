# Gate 0 - Data Readiness

## Purpose

Confirm that the encoded source dataset is readable and that the requested formal county scope can actually be supported by the raw data.

## Current Implementation

- Entrypoint: `run_all.py`
- Main code path:
  - `pd.read_csv(..., usecols=["county_label"])`
  - `src/project_config.scope_readiness()`
  - `src/final_delivery.write_gate_verdicts()`
- The old `pipeline_tools/` discovery/profiling path is retired and is no longer part of the current executable repository structure.

## What This Gate Checks

- `data/raw/encoded_ml_dataset.csv` can be read.
- Raw county labels exist and can be compared against the formal scope.
- Missing requested counties are surfaced explicitly instead of being imputed.

## Current Output Artifacts

- `model_outputs/gate0_verdict.json`
- readiness evidence embedded into:
  - `reports/final_project_report.md`
  - `reports/tour_guide.md`
  - `reports/dashboard.html`

## Current Status

- Status is currently **CONDITIONAL** in the generated verdicts because the requested formal scope includes counties that are not present in the raw mapping/data.
- This gate is a scope-coverage audit, not a generic schema-profiler workflow.
