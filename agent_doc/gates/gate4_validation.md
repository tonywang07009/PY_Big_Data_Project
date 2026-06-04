# Gate 4 - Validation

## Purpose

Confirm model performance with time-aware validation and leakage control.

## Planned Micro-Tasks

- T4-A: create time-based train/test split
- T4-B: run `TimeSeriesSplit` cross-validation
- T4-C: summarize RMSE, MAE, and R2
- T4-D: audit leakage and runtime stability
- T4-E: produce Diver/Counter validation verdict

## Expected Outputs

- `model_outputs/metrics/validation_metrics.json`
- `model_outputs/validation_audit.json`
- `model_outputs/gate4_verdict.json`
- `reports/step4_report.md`

## Discussion Status

Implementation details are not finalized. Discuss validation windows, pass thresholds, and leakage audit rules before coding.

