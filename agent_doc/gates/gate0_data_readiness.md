# Gate 0 - Data Readiness

## Purpose

Confirm that source data is present, readable, and semantically ready before preprocessing or modeling.

## Planned Micro-Tasks

- T0-A: discover files under `data/raw/`
- T0-B: profile row counts, columns, dtypes, null rates, and date coverage
- T0-C: validate semantic rules for county, month, counts, and target fields
- T0-D: produce Diver/Counter data-readiness verdict

## Expected Outputs

- `model_outputs/file_manifest.json`
- `model_outputs/column_profile.json`
- `model_outputs/gate0_verdict.json`
- `reports/step0_report.md`

## Discussion Status

Implementation details are not finalized. Discuss source file schema, required columns, and pass/fail thresholds before coding.

