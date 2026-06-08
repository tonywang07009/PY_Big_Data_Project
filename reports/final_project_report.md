# MecDonate Final Project Report

## Executive Summary

MecDonate predicts monthly invoice donation ratios for the project scope counties and converts those predictions into operational priority scores. The formal scope is Taipei, Taichung, Kaohsiung, Lienchiang, Hualien, and Yunlin. The encoded dataset contains 5 of those six counties: 台中市, 台北市, 花蓮縣, 雲林縣, 高雄市.

Data readiness is **CONDITIONAL** because the source mapping and raw encoded data do not include 連江縣. No synthetic county rows were added and no substitute county was used.

The best empirical model on the time split is **Ridge** with test R2 0.9186 and test RMSE 0.0022. This differs from the proposal expectation that tree models would lead: under the small scoped sample and no-shuffle temporal validation, regularized linear models generalize more stably than XGBoost or Random Forest.

## Data Scope and Feature Matrix

- Formal machine outputs: `model_outputs/`
- Human-facing reports: `reports/`
- Scoped matrix: `model_outputs/county_month_matrix.csv`
- Matrix shape: 245 rows x 27 columns
- Month coverage in scoped matrix: 2019-01 to 2024-12
- Train years: 2019, 2020, 2022
- Test years: 2023, 2024

STL seasonal features are computed per county. Lag features are joined by calendar date, so lags that would cross the source data coverage gap remain missing and are excluded from model training.

## Clustering

K-Means clustering uses time-averaged county profiles from the in-scope available counties only. The valid K search is bounded by the scoped county count, avoiding invalid silhouette scores for the five-county sample.

- Best K: 3
- Cluster artifact: `model_outputs/cluster_assignments.csv`
- PCA plot: `model_outputs/clustering_pca.png`

| county_label | county_english_name | county_name | cluster | pc1 | pc2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Taichung | 台中市 | 2 | 0.3017 | 1.6500 |
| 2 | Taipei | 台北市 | 0 | 4.3788 | -1.1928 |
| 15 | Hualien | 花蓮縣 | 1 | -2.5040 | -1.5328 |
| 17 | Yunlin | 雲林縣 | 1 | -2.0590 | -0.5220 |
| 18 | Kaohsiung | 高雄市 | 2 | -0.1174 | 1.5975 |

## Model Comparison

The validation policy is fixed: train on 2019-2022, test on 2023 onward, and never shuffle time-series rows.

| model | test_RMSE | test_R2 | train_R2 |
| --- | --- | --- | --- |
| Ridge | 0.0022 | 0.9186 | 0.9437 |
| Lasso | 0.0027 | 0.8756 | 0.9305 |
| Linear Regression | 0.0036 | 0.7725 | 0.9545 |
| XGBoost | 0.0056 | 0.4488 | 0.9974 |
| Random Forest | 0.0067 | 0.2229 | 0.9868 |

The proposal expected XGBoost and Random Forest to outperform the linear baselines. The formal run shows the opposite: Ridge/Lasso are more robust on the scoped sample, while the tree models fit the training window more aggressively and lose accuracy on the future test window. This result is consistent with limited rows, strong autocorrelation, and a time coverage gap.

## Priority Scores

Priority score is computed as:

`expected_donated_count = predicted_donation_ratio * total_county_invoice_count`

The expected count is then min-max normalized and sorted descending. The formal priority artifact is `model_outputs/priority_scores.csv`.

| rank | county_name | month | actual_donation_ratio | predicted_donation_ratio | invoice_count | expected_donated_count | normalized_priority_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 台北市 | 2024-05 | 0.0229 | 0.0242 | 178,599,146 | 4323073.7100 | 1.0000 |
| 2 | 台北市 | 2024-07 | 0.0238 | 0.0230 | 179,297,688 | 4119145.0423 | 0.9527 |
| 3 | 台北市 | 2024-11 | 0.0239 | 0.0221 | 185,696,567 | 4109742.1368 | 0.9505 |
| 4 | 台北市 | 2024-06 | 0.0239 | 0.0236 | 174,431,178 | 4108210.2147 | 0.9502 |
| 5 | 台北市 | 2024-08 | 0.0234 | 0.0224 | 183,443,397 | 4108094.2107 | 0.9502 |
| 6 | 台北市 | 2023-05 | 0.0238 | 0.0237 | 168,545,559 | 3994290.3600 | 0.9238 |
| 7 | 台北市 | 2024-12 | 0.0237 | 0.0213 | 186,912,671 | 3985178.2825 | 0.9216 |
| 8 | 台北市 | 2023-07 | 0.0230 | 0.0229 | 172,601,473 | 3960142.2642 | 0.9158 |
| 9 | 台北市 | 2024-04 | 0.0232 | 0.0233 | 169,902,851 | 3955955.8349 | 0.9149 |
| 10 | 台北市 | 2023-08 | 0.0231 | 0.0226 | 174,927,157 | 3948332.6812 | 0.9131 |
| 11 | 台北市 | 2024-03 | 0.0236 | 0.0225 | 175,684,907 | 3947824.9380 | 0.9130 |
| 12 | 台北市 | 2024-10 | 0.0239 | 0.0221 | 176,730,529 | 3907875.8928 | 0.9037 |

## Interpretability

Ridge standardized coefficients are the primary explanation for the best-performing model family. XGBoost SHAP is retained as a nonlinear contrast.

Top Ridge coefficients:

| feature | standardized_coefficient | abs_standardized_coefficient |
| --- | --- | --- |
| donation_ratio_lag3 | 0.0029 | 0.0029 |
| avg_invoice_value | -0.0020 | 0.0020 |
| carrier_usage_ratio | 0.0012 | 0.0012 |
| donation_ratio_roll3 | 0.0010 | 0.0010 |
| donation_seasonal | 0.0009 | 0.0009 |
| log_total_count | 0.0007 | 0.0007 |
| industry_hhi | -0.0005 | 0.0005 |
| month_cos | -0.0004 | 0.0004 |
| county_type=remote | -0.0004 | 0.0004 |
| county_type=metropolitan | 0.0003 | 0.0003 |

Top XGBoost SHAP features:

| feature | mean_abs_shap |
| --- | --- |
| housing_burden | 0.0028 |
| donation_ratio_lag1 | 0.0006 |
| industry_hhi | 0.0005 |
| donation_ratio_lag3 | 0.0002 |
| donation_ratio_lag2 | 0.0002 |
| donation_seasonal | 0.0001 |
| log_total_count | 0.0001 |
| month_sin | 0.0001 |
| month_cos | 0.0001 |
| n_active_industries | 0.0000 |

## Limitations

- Lienchiang County is part of the requested project scope but is absent from the source mapping and encoded dataset.
- The scoped run has a small training sample, so simpler regularized models are less fragile than high-capacity tree ensembles.
- Source months are not fully continuous, which limits year-over-year features and requires careful date-based lag handling.
- Clustering summarizes county averages and should not be read as month-specific behavior.

## Output Inventory

- `model_outputs/county_month_matrix.csv`
- `model_outputs/cluster_assignments.csv`
- `model_outputs/metrics/model_comparison.csv`
- `model_outputs/metrics/model_results.json`
- `model_outputs/metrics/ridge_coefficients.csv`
- `model_outputs/shap/shap_importance.csv`
- `model_outputs/priority_scores.csv`
- `model_outputs/gate0_verdict.json` through `model_outputs/gate6_verdict.json`
- `reports/final_project_report.md`
- `reports/dashboard.html`
- `reports/tour_guide.md`
