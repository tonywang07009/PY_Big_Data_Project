# Gate 1 - Preprocessing

## Purpose

Build a clean monthly analysis matrix from mixed-frequency data.

## Planned Micro-Tasks

- T1-A: define missing value policy
- T1-B: align annual indicators to monthly rows
- T1-C: apply STL to high-frequency monthly features only
- T1-D: normalize numeric features
- T1-E: produce Diver/Counter preprocessing verdict

## Expected Outputs

- `model_outputs/preprocessing_summary.json`
- `model_outputs/normalized_matrix.parquet`
- `model_outputs/gate1_verdict.json`
- `reports/step1_report.md`

## Discussion Status

Implementation details are not finalized. Discuss interpolation policy, STL eligibility, and normalization scope before coding.

