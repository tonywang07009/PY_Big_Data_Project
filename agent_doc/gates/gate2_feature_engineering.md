# Gate 2 - Feature Engineering

## Purpose

Represent county heterogeneity and prepare model-ready features.

## Planned Micro-Tasks

- T2-A: build county-level feature matrix
- T2-B: encode county identity and county type
- T2-C: run K-Means with Elbow and Silhouette checks
- T2-D: create PCA visualization for cluster interpretation
- T2-E: produce Diver/Counter feature verdict

## Expected Outputs

- `model_outputs/feature_matrix.parquet`
- `model_outputs/cluster_profile.json`
- `model_outputs/gate2_verdict.json`
- `reports/step2_report.md`

## Discussion Status

Implementation details are not finalized. Discuss county type definitions, cluster feature set, and K selection before coding.

