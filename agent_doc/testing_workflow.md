# Testing Workflow

Testing is organized at three levels: task, gate, and system.

## Task-Level Checks

Run after each micro-task.

| Check | Question | Evidence |
|---|---|---|
| Fit | Does the output satisfy QFD and downstream needs? | task verdict JSON |
| Stability | Can the task be rerun with the same inputs? | hashes, row counts, deterministic config |
| Maintainability | Can a human inspect and revise it later? | short English notes and clear paths |
| Leakage | Does it preserve time order and avoid target leakage? | split audit or preprocessing note |
| Artifact | Are outputs present and non-empty? | file existence and size check |

## Gate-Level Checks

At the end of each gate:

- validate all task verdicts
- validate required artifacts
- write `model_outputs/gateN_verdict.json`
- update the relevant `agent_doc/gates/gateN_*.md`
- write a report under `reports/` when the gate produces human-facing results

## System-Level Checks

After Gate 7:

- Gate 0 through Gate 7 are `PASS` or human-accepted `CONDITIONAL`
- final model uses monthly expanding Walk-Forward Analysis
- numeric feature scalers are fit per `county_label` on train-fold rows only
- county calibration is fit from train-fold months only
- model comparison chart is generated for log-XGBoost, calibrated XGBoost, historical baseline, and final ensemble
- Gate 7 validates ensemble metrics against formal thresholds
- final outputs trace back to source data and model artifacts
- reports are readable by future human maintainers
