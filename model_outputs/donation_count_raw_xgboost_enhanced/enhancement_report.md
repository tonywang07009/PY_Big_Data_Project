# Enhanced donation_count XGBoost Report

## Scope

- Source data: `data/raw/encoded_ml_dataset.csv`
- Target: `donation_count`
- Grain: raw encoded rows; no county-month aggregation for training rows
- Train split: `year <= 2022`
- Validation split: `year >= 2023`
- The unrelated Trimmomatic/HISAT2/featureCounts table was not used.

## Best Enhanced Model

- Variant: `unweighted_regularized`
- RMSE: `215,163.860`
- MAE: `110,063.289`
- sMAPE: `40.856%`
- epsilon-MAPE: `32.095%`
- R2: `0.960`
- Mean county abs pct diff: `21.958%`

## Baseline Comparison

| Metric | Baseline | Enhanced | Delta |
|---|---:|---:|---:|
| rmse | 249,590.301 | 215,163.860 | -34,426.442 |
| mae | 120,502.594 | 110,063.289 | -10,439.305 |
| smape | 38.882 | 40.856 | 1.974 |
| epsilon_mape | 30.697 | 32.095 | 1.398 |
| r2 | 0.946 | 0.960 | 0.014 |
| mean_county_abs_pct_diff | 20.678 | 21.958 | 1.280 |

## Model Variants

| Variant | Selection score | RMSE | MAE | sMAPE | epsilon-MAPE | R2 | Mean county abs pct diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| unweighted_regularized | 1.333 | 215,163.860 | 110,063.289 | 40.856% | 32.095% | 0.960 | 21.958% |
| weighted_regularized | 1.667 | 314,912.415 | 108,607.016 | 34.732% | 34.086% | 0.914 | 28.404% |

## Inner 2022 Parameter Selection

| Weighting | Candidate | Inner sMAPE | Inner MAE | Inner R2 |
|---|---|---:|---:|---:|
| unweighted | baseline_like | 97.205% | 437,558.750 | -0.149 |
| unweighted | conservative | 144.766% | 415,293.750 | -0.052 |
| unweighted | deeper | 89.892% | 416,182.375 | -0.045 |
| unweighted | regularized | 70.005% | 402,029.719 | -0.019 |
| weighted | baseline_like | 88.102% | 432,011.969 | -0.143 |
| weighted | conservative | 141.765% | 405,550.094 | -0.018 |
| weighted | deeper | 89.183% | 412,193.125 | -0.043 |
| weighted | regularized | 65.663% | 386,033.719 | 0.053 |

## Top SHAP Features

| Feature | mean_abs_SHAP |
|---|---:|
| housing_burden | 470,132.219 |
| industry_entropy | 65,761.438 |
| donation_count_lag1 | 65,201.305 |
| donation_count_lag2 | 46,700.371 |
| month_num | 29,369.738 |
| total_county_invoice_count | 23,398.199 |
| donation_count_roll3_mean | 21,199.984 |
| top3_industry_share | 11,350.293 |
| carrier_usage_ratio | 10,481.577 |
| total_county_invoice_amount | 10,179.962 |

## Largest County-Level Validation Errors

| County | Actual | Predicted | Diff | Pct diff |
|---|---:|---:|---:|---:|
| 高雄市 | 14,839,807.000 | 8,904,207.000 | -5,935,600.000 | -39.998% |
| 台中市 | 9,753,670.000 | 15,316,745.000 | 5,563,075.000 | 57.036% |
| 台北市 | 97,085,454.000 | 92,539,490.000 | -4,545,966.000 | -4.682% |
| 桃園市 | 9,407,614.000 | 7,222,736.500 | -2,184,877.500 | -23.225% |
| 新北市 | 31,756,823.000 | 33,168,990.000 | 1,412,167.000 | 4.447% |
| 台南市 | 4,857,047.000 | 4,092,084.200 | -764,962.750 | -15.750% |
| 基隆市 | 2,048,990.000 | 1,514,220.900 | -534,769.125 | -26.099% |
| 彰化縣 | 2,132,807.000 | 1,651,673.100 | -481,133.875 | -22.559% |

## Interpretation

- County identity is now explicit, so stable county-specific bias can be learned instead of forced through common economic variables only.
- County-month structure features preserve raw-row detail while giving the model month-level composition information.
- Weighted raw-row training tests whether repeated county-month targets were dominating the fit through row count alone.
- Use `county_year_validation_comparison.csv` to identify whether remaining error is concentrated in 2023 or 2024.
