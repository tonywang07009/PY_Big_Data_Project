# Gate 2 - Feature Engineering

## Purpose

Represent county-level structure, time behavior, and clustering-ready profiles from the preprocessed county-month matrix.

## Current Main Implementation

- Entrypoint: `run_all.py`
- Main code path:
  - `src/features.build_county_month()`
  - `src/clustering.run()`
  - `src/clustering.run_v2()`

## Current Feature Contracts

- Main regression target in the formal project pipeline:
  - `donation_ratio`
- Main engineered numeric features include:
  - `log_total_count`
  - `log_total_amount`
  - `avg_invoice_value`
  - `carrier_usage_ratio`
  - `n_active_industries`
  - `industry_hhi`
  - `donation_seasonal`
  - `month_sin`
  - `month_cos`
  - `donation_ratio_lag1`
  - `donation_ratio_lag2`
  - `donation_ratio_lag3`
  - `donation_ratio_roll3`
- Main categorical feature:
  - `county_type`

## Current Clustering Workflows

### Legacy clustering

- Entry: `src/clustering.run()`
- Uses county mean profiles.
- Produces:
  - `model_outputs/cluster_assignments.csv`
  - `model_outputs/clustering_pca.png`
  - `model_outputs/clustering_elbow_silhouette.png`
  - `model_outputs/clustering_metrics.json`

### Representative-space clustering

- Entry: `src/clustering.run_v2()`
- Uses county representative features built from profile mean and std.
- Produces:
  - `model_outputs/county_cluster_v2.csv`
  - `model_outputs/county_month_pca_points_v2.csv`
  - grouped PCA figures
  - `model_outputs/clustering_v2_metrics.json`

## Current Status

- Implemented and active.
- Gate 2 in this repository covers both engineered feature creation and county clustering outputs.
