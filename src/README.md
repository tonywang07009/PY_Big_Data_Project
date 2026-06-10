# src Module Guide

This document explains the public entry functions used by the main MecDonate pipeline. It focuses on callable entrypoints and workflow roles, while internal `_helper` functions are grouped by behavior instead of documented one by one.

## Pipeline Flow

`run_all.py` calls the `src/` modules in this order:

1. `features.build_county_month()`
2. `project_config.filter_scope_counties()` and `project_config.scope_readiness()`
3. `clustering.run()` and `clustering.run_v2()`
4. `model.run()`
5. `shap_analysis.run()`
6. `final_delivery.write_all()`

## project_config.py

### `ScopeCounty`
- Purpose: immutable definition of a county in the formal project scope.
- Role in pipeline: provides the English name, Chinese name, and county code used by scope filtering and readiness reporting.

### `add_county_name(df)`
- Purpose: return a copy of a DataFrame with a stable `county_name` column.
- Input: any frame containing `county_label`.
- Output: same rows plus mapped Chinese county names.
- Role in pipeline: keeps downstream clustering, reporting, and dashboard outputs human-readable.

### `filter_scope_counties(df)`
- Purpose: keep only counties in the formal delivery scope.
- Input: engineered county-month matrix.
- Output: filtered county-month matrix with `county_name`.
- Role in pipeline: restricts `run_all.py` outputs to the approved final-delivery counties.

### `scope_readiness(raw_county_labels)`
- Purpose: summarize requested, available, and missing counties before the formal pipeline runs.
- Input: iterable of raw county codes found in the source data.
- Output: readiness dictionary used by final reporting.
- Role in pipeline: drives readiness warnings in `run_all.py` and gate-verdict content in `final_delivery.py`.

## features.py

### `county_type(code)`
- Purpose: map county code to a coarse county type category.
- Input: integer county code.
- Output: one of `metropolitan`, `remote`, or `agricultural`.
- Role in pipeline: provides a categorical feature for the regression stage.

### `build_county_month(raw_path)`
- Purpose: transform row-level encoded invoice data into a county-month modeling matrix.
- Input: path to `encoded_ml_dataset.csv`.
- Output: engineered county-month DataFrame.
- Main steps:
  - read raw encoded rows
  - aggregate county-month constant indicators
  - derive carrier usage and industry concentration features
  - add time fields and cyclic month features
  - compute STL seasonal/trend features for `donation_ratio`
  - compute date-safe lag and rolling target features
- Role in pipeline: this is the primary feature-engineering entrypoint for `run_all.py`.

### Internal helper groups
- `_county_month_aggregates()` builds grouped behavioral features per county-month.
- module constants `FEATURE_COLS` and `TARGET` define the modeling contract reused by `model.py` and `shap_analysis.py`.

## clustering.py

### `county_profiles(cm)`
- Purpose: compress the county-month matrix into one mean profile per county.
- Input: scoped county-month matrix.
- Output: county-level profile table with county names.
- Role in pipeline: supports the legacy clustering workflow.

### `county_representatives(cm)`
- Purpose: build county representatives using feature mean and standard deviation.
- Input: scoped county-month matrix.
- Output: county-level representative table.
- Role in pipeline: supports the second clustering workflow that uses richer county summaries.

### `run(cm, outdir)`
- Purpose: execute the legacy clustering workflow.
- Input: scoped county-month matrix and output directory.
- Output: cluster assignment table plus elbow, silhouette, and PCA artifacts written to disk.
- Main steps:
  - standardize county mean profiles
  - search for the best `K`
  - fit K-Means
  - run PCA for visualization
  - save assignments and charts

### `run_v2(cm, outdir)`
- Purpose: execute the richer clustering workflow used for grouped PCA views.
- Input: scoped county-month matrix and output directory.
- Output: county-cluster table, county-month PCA points, grouped PCA figures, and metrics JSON.
- Main steps:
  - build county representatives from mean and std features
  - choose `K` with silhouette search
  - cluster counties in representative space
  - center county-month points by cluster centroid
  - run PCA on centered points
  - auto-group counties into figure groups and save outputs

### Internal helper groups
- `_k_search()` and `_plot_k_search()` handle model selection diagnostics.
- `_auto_plot_groups()` and `_plot_grouped_pca()` manage grouped PCA figure generation.

## model.py

### `split(cm)`
- Purpose: perform the formal time split for modeling.
- Input: scoped county-month matrix.
- Output: train frame with `year <= 2022` and test frame with `year >= 2023`.
- Role in pipeline: enforces the project’s no-shuffle time policy.

### `one_hot_encoder()`
- Purpose: create a version-compatible `OneHotEncoder`.
- Output: configured encoder with `handle_unknown="ignore"`.
- Role in pipeline: shared categorical preprocessing for linear and tree models.

### `build_models()`
- Purpose: define the candidate model set.
- Output: dictionary of named sklearn/XGBoost pipelines.
- Models included: Linear Regression, Ridge, Lasso, Random Forest, XGBoost.

### `ridge_coefficients(pipe)`
- Purpose: extract standardized Ridge coefficients after preprocessing.
- Input: trained Ridge pipeline.
- Output: sorted coefficient table.
- Role in pipeline: provides interpretable linear-feature importance output.

### `priority_scores(test, predictions, model_name)`
- Purpose: convert model predictions into county-month priority rankings.
- Input: test frame, predictions, and best model name.
- Output: ranked priority-score table.
- Role in pipeline: creates the actionable prioritization artifact written to `model_outputs/priority_scores.csv`.

### `run(cm, outdir, priority_path=None)`
- Purpose: run the full model-comparison stage.
- Input: scoped county-month matrix, output directory, optional priority output path.
- Output: comparison DataFrame, best-model metadata, priority scores, and Ridge coefficients; also writes plots and JSON/CSV outputs to disk.
- Main steps:
  - split train/test by year
  - fit all candidate models
  - compare RMSE and R2
  - save comparison chart
  - save predicted-vs-actual scatter for the best model
  - export Ridge coefficients and priority scores

### Internal helper groups
- `_linear_pipe()` and `_tree_pipe()` define preprocessing per model family.
- `_clean_feature_name()` normalizes transformed feature names for readable reporting.

## shap_analysis.py

### `run(cm, outdir)`
- Purpose: compute SHAP explanations for the tree-model feature space.
- Input: scoped county-month matrix and output directory.
- Output: SHAP importance DataFrame plus beeswarm/bar plots written to disk.
- Main steps:
  - reuse the formal time split from `model.split()`
  - build the design matrix from numeric features and one-hot county type
  - fit an XGBoost regressor
  - compute SHAP values on the test set
  - export mean absolute SHAP importance and plots

## final_delivery.py

### `write_gate_verdicts(output_root, cm, readiness, model_summary, shap_importance)`
- Purpose: build gate verdict payloads for the formal delivery.
- Input: core outputs from feature engineering, readiness checking, modeling, and SHAP analysis.
- Output: list of gate dictionaries and gate JSON files on disk.

### `write_final_report(output_root, reports_root, cm, readiness, cluster_profiles, model_summary, shap_importance, gates)`
- Purpose: write the final Markdown project report.
- Role in pipeline: converts machine-readable outputs into the main human-facing narrative.

### `write_dashboard(output_root, reports_root, cm, readiness, cluster_profiles, model_summary, shap_importance, gates)`
- Purpose: build the static HTML dashboard.
- Role in pipeline: provides an inspectable summary view for the project outputs.

### `write_tour_guide(output_root, reports_root, cm, readiness, cluster_profiles, model_summary, shap_importance, gates)`
- Purpose: write the Tour Guide document that explains how to review the final deliverables.

### `write_all(output_root, reports_root, cm, readiness, cluster_profiles, model_summary, shap_importance)`
- Purpose: orchestrate all final-delivery outputs from a single call.
- Input: results from all earlier pipeline stages.
- Output: gate verdicts plus report, dashboard, and tour guide files.
- Main steps:
  - assemble gate payloads
  - write gate JSON files
  - write final report
  - write dashboard
  - write tour guide

### Internal helper groups
- JSON helpers normalize numpy/pandas values before serialization.
- Markdown/HTML table helpers convert DataFrames into final report/dashboard fragments.
- formatting helpers keep repeated text rendering consistent across deliverables.

## plotting_setup.py

### `configure_matplotlib_cache()`
- Purpose: point Matplotlib and font caches at writable temp directories.
- Role in pipeline: prevents plotting failures in sandboxed or restricted local environments.
