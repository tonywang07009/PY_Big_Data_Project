# Possible Modification Directions

## Current Baseline

Current formal pipeline:

- Source data: `data/raw/encoded_ml_dataset.csv`
- Target: `donation_count`
- Modeling grain: raw rows
- Validation split:
  - train `<= 2022`
  - validation `>= 2023`
- Model: `XGBoost`
- Current validation summary:
  - `RMSE = 249,590.301350`
  - `MAE = 120,502.593750`
  - `sMAPE = 38.881973`
  - `epsilon-MAPE = 30.697399`
  - `R2 = 0.946285`

## Main Findings

1. Raw-row modeling is useful for `donation_count`, but it does not solve every county's bias.
2. The largest practical county-level errors remain concentrated in a few counties:
   - `台北市`
   - `高雄市`
   - `台中市`
   - `新北市`
3. The model relies heavily on:
   - `housing_burden`
   - `donation_count_lag1`
   - `donation_count_lag2`
   - `month_num`
   - `total_county_invoice_count`
   - `price_income_ratio`
4. Some counties show directional bias across many months, not just isolated outliers:
   - `台中市`: long-run overprediction
   - `高雄市`: long-run underprediction
   - `新北市`: regime shift between 2023 and 2024
   - `台北市`: comparatively stable, but still underpredicts in some periods

## Recommended Modification Directions

### 1. Improve county discrimination

Current raw-row pipeline uses industry/carrier detail, but it still lacks strong county identity signals.

Recommended actions:

- Add `county_label` as a categorical feature with one-hot encoding
- Compare against `county_type` style grouping features
- Test whether explicit county identity reduces the stable bias seen in `台中市`, `高雄市`, and `新北市`

Expected effect:

- Better separation of county-specific structural patterns
- Less tendency to force one common rule across counties with different dynamics

### 2. Strengthen historical behavior features

Current lag features only include:

- `donation_count_lag1`
- `donation_count_lag2`
- `donation_count_lag3`

Recommended actions:

- Add rolling statistics:
  - rolling mean
  - rolling std
  - rolling max/min
- Add year-over-year month comparison if data continuity permits
- Add lag interaction with county or season

Expected effect:

- Better capture of momentum and local trend persistence
- Reduced month-to-month drift when the target level changes across 2023-2024

### 3. Add stronger structural aggregation features

Raw rows preserve detail, but the model may still miss higher-level composition signals.

Recommended actions:

- Add county-month structure summaries back into the raw-row pipeline:
  - `carrier_usage_ratio`
  - `n_active_industries`
  - `industry_hhi`
  - `industry_entropy`
  - `top1_industry_share`
  - `top3_industry_share`
- Merge these county-month statistics onto each raw row

Expected effect:

- Keep raw-row sample size
- Also give the model county-month structure information that raw rows alone do not summarize clearly

### 4. Reduce raw-row repeated-target side effects

Even though raw-row modeling is the current best path for `donation_count`, one county-month target is still repeated across many raw rows.

Recommended actions:

- Compare current raw-row training against:
  - sample weighting by `1 / row_count_per_county_month`
  - capped sampling per county-month
  - hybrid pipeline with county-month target and raw-derived structure features

Expected effect:

- Reduce the risk that months with many raw rows dominate the fit only because they have more duplicated targets

### 5. Tune XGBoost specifically for cross-time stability

Current parameters are still a generic baseline.

Recommended actions:

- Grid or manual search over:
  - `max_depth`
  - `n_estimators`
  - `learning_rate`
  - `subsample`
  - `colsample_bytree`
  - `min_child_weight`
  - `reg_alpha`
  - `reg_lambda`
- Compare tuning by:
  - overall metrics
  - county-level absolute percent difference

Expected effect:

- Lower risk of overfitting the training era while drifting on 2023-2024

### 6. Revisit validation slicing

Current validation uses all of `2023-2024` together.

Recommended actions:

- Split evaluation into:
  - `2023` only
  - `2024` only
- Report county-level error separately for each year

Expected effect:

- Detect whether the model is failing because of a genuine 2024 regime shift
- Help explain why some counties switch from underprediction to overprediction

### 7. Add post-model error audit by county

This should become part of the standard workflow, not just a one-off inspection.

Recommended actions:

- Always output:
  - county-month comparison table
  - county comparison table
  - top overpredicted counties
  - top underpredicted counties
- Add threshold alerts when county-level percent difference exceeds a chosen level

Expected effect:

- Faster debugging
- Easier model iteration using business-relevant error patterns

## Suggested Execution Order

Recommended order of experiments:

1. Add `county_label` one-hot
2. Add county-month structural summary features to the raw-row pipeline
3. Tune XGBoost hyperparameters
4. Compare weighted raw-row training vs current unweighted raw-row training
5. Split validation reporting into `2023` and `2024`

## Most Likely High-Impact Next Step

If only one next change is allowed, the most promising direction is:

`raw-row donation_count pipeline + county_label one-hot + county-month structural summary features`

Reason:

- It preserves the best-performing raw-row donation_count path
- It directly targets the current weakness: insufficient county-specific and county-month structural information
- It is more likely to reduce the large directional errors in `台中市`, `高雄市`, and `新北市` than normalization alone
