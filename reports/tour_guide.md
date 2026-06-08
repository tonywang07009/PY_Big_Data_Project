# MecDonate Tour Guide

This guide records the final alignment work at micro-task level. It does not replace the final report; it explains what was done, why it was done, what files were read or written, and how each task was judged.

### T0-A - Project Scope Audit

**Objective:** Confirm the formal county, target, split, and output requirements.

**Action Taken:** Read `project.md`, project governance notes, and the existing runner before modifying code.

**Decision Logic:** The project control file is the highest-level routing source, so implementation follows its scope and folder contracts.

**Inputs and Outputs:** Input: `project.md`, `agent_doc/project_brief.md`; Output: scoped delivery plan used by `run_all.py`.

**Diver Notes:** The task narrows the project to six named counties and a no-shuffle temporal validation policy.

**Counter Notes:** The scope is clear enough to implement without asking a new question because missing county data can be documented conditionally.

**Verdict:** PASS (5/5)

### T0-B - County Coverage Check

**Objective:** Determine whether all six requested counties exist in the encoded source data.

**Action Taken:** Compared the requested scope against the county mapping and raw county labels; missing county data: 連江縣.

**Decision Logic:** A missing requested county is a data-readiness issue, not a modeling choice. No synthetic or substitute records should be introduced.

**Inputs and Outputs:** Input: `label_mapping_table_county.csv`, `encoded_ml_dataset.csv`; Output: `model_outputs/gate0_verdict.json`.

**Diver Notes:** The pipeline keeps available in-scope counties and records the absent county as a formal warning.

**Counter Notes:** Gate 0 is conditional when a requested county is absent, but downstream work can proceed for the available scope.

**Verdict:** CONDITIONAL (3/5)

### T1-A - County-Month Matrix

**Objective:** Build the modeling matrix at county-month granularity.

**Action Taken:** Aggregated county-industry-carrier raw rows into monthly county features.

**Decision Logic:** The target and economic fields are constant within county-month; behavioral structure is summarized through carrier ratio and industry concentration.

**Inputs and Outputs:** Input: `encoded_ml_dataset.csv`; Output: `model_outputs/county_month_matrix.csv`.

**Diver Notes:** STL seasonality and lag features preserve the proposal's time-series framing.

**Counter Notes:** The matrix is deterministic and can be regenerated from the raw encoded file.

**Verdict:** PASS (5/5)

### T1-B - Scope Filter

**Objective:** Ensure formal artifacts contain only requested in-scope counties that exist in the data.

**Action Taken:** Filtered the county-month matrix to Taichung, Taipei, Hualien, Yunlin, and Kaohsiung.

**Decision Logic:** Filtering after per-county feature construction is valid because no feature borrows information across counties.

**Inputs and Outputs:** Input: full county-month matrix in memory; Output: scoped `model_outputs/county_month_matrix.csv`.

**Diver Notes:** The output intentionally excludes all non-scope counties.

**Counter Notes:** Validation checks should fail if any formal output contains a county code outside the allowed set.

**Verdict:** PASS (5/5)

### T1-C - Time Features

**Objective:** Preserve temporal ordering and avoid gap leakage.

**Action Taken:** Kept the date-join lag policy and the train/test split by year.

**Decision Logic:** Date joins prevent lags from silently crossing the source coverage gap.

**Inputs and Outputs:** Input: scoped feature matrix; Output: split metadata in `model_outputs/metrics/model_results.json`.

**Diver Notes:** The split remains train years 2019-2022 and test years 2023 onward.

**Counter Notes:** No shuffled validation is used, matching the project control document.

**Verdict:** PASS (5/5)

### T2-A - Cluster Profiles

**Objective:** Represent county heterogeneity for the scoped counties.

**Action Taken:** Averaged economic, donation, and behavioral features by county.

**Decision Logic:** K-Means needs one row per county profile rather than one row per county-month observation.

**Inputs and Outputs:** Input: `model_outputs/county_month_matrix.csv`; Output: `model_outputs/cluster_assignments.csv`.

**Diver Notes:** The profile table includes PCA coordinates for review.

**Counter Notes:** The cluster assignment is reproducible with `random_state=42`.

**Verdict:** PASS (5/5)

### T2-B - K Selection

**Objective:** Choose a valid cluster count for the smaller formal scope.

**Action Taken:** Bounded candidate K values by the available scoped county count before computing silhouette scores.

**Decision Logic:** The old K range could exceed the number of scoped samples; dynamic bounds prevent invalid clustering.

**Inputs and Outputs:** Input: scoped county profiles; Output: `model_outputs/clustering_metrics.json`.

**Diver Notes:** Silhouette remains the primary model-selection signal.

**Counter Notes:** The method is stable if the available county count changes in a future dataset.

**Verdict:** PASS (5/5)

### T2-C - Cluster Visualization

**Objective:** Produce inspectable clustering artifacts.

**Action Taken:** Saved elbow/silhouette and PCA charts under `model_outputs/`.

**Decision Logic:** Static images are sufficient for final project review and avoid adding dashboard framework dependencies.

**Inputs and Outputs:** Input: clustering model and PCA transform; Output: `model_outputs/clustering_pca.png`, `model_outputs/clustering_elbow_silhouette.png`.

**Diver Notes:** County names are retained for reviewer readability.

**Counter Notes:** Artifacts are path-stable and referenced by the report and dashboard.

**Verdict:** PASS (5/5)

### T3-A - Model Family Build

**Objective:** Train the proposal's linear and nonlinear regression candidates.

**Action Taken:** Fit Linear Regression, Ridge, Lasso, Random Forest, and XGBoost on the scoped training set.

**Decision Logic:** The comparison preserves baseline accountability before judging nonlinear models.

**Inputs and Outputs:** Input: scoped matrix; Output: `model_outputs/metrics/model_comparison.csv`.

**Diver Notes:** Linear models use train-fitted scaling and categorical one-hot encoding inside pipelines.

**Counter Notes:** Tree models are compared on the same split, so metric differences are attributable to model behavior.

**Verdict:** PASS (5/5)

### T3-B - Best Model Selection

**Objective:** Select the model used for priority scoring.

**Action Taken:** Sorted models by test R2; selected `Ridge`.

**Decision Logic:** Priority recommendations should use the strongest future-period model, not the strongest training fit.

**Inputs and Outputs:** Input: model predictions and metrics; Output: `model_outputs/metrics/model_results.json`.

**Diver Notes:** The selected model reflects empirical performance under the formal split.

**Counter Notes:** The report explicitly explains when the empirical result differs from the proposal expectation.

**Verdict:** PASS (5/5)

### T3-C - Ridge Coefficients

**Objective:** Provide the main explanation for the best linear model family.

**Action Taken:** Exported standardized Ridge coefficients sorted by absolute magnitude.

**Decision Logic:** The best empirical model is regularized linear, so its coefficient ranking is the primary explanation.

**Inputs and Outputs:** Input: fitted Ridge pipeline; Output: `model_outputs/metrics/ridge_coefficients.csv`.

**Diver Notes:** Standardized numeric inputs make coefficient magnitudes comparable.

**Counter Notes:** Categorical one-hot coefficients are included and labeled.

**Verdict:** PASS (5/5)

### T4-A - Split Audit

**Objective:** Verify that training and testing obey the time policy.

**Action Taken:** Recorded train years, test years, and no-shuffle policy in model results and Gate 4.

**Decision Logic:** Temporal prediction requires future rows to be held out by date.

**Inputs and Outputs:** Input: `year` column from the scoped matrix; Output: `model_outputs/gate4_verdict.json`.

**Diver Notes:** Train rows use 2019-2022; test rows use 2023 onward.

**Counter Notes:** A validation script can re-read the output and assert the same year boundaries.

**Verdict:** PASS (5/5)

### T4-B - Metric Review

**Objective:** Summarize generalization performance.

**Action Taken:** Computed test RMSE, test R2, and train R2 for each model.

**Decision Logic:** Train R2 is retained to reveal overfitting, especially in tree models.

**Inputs and Outputs:** Input: model predictions; Output: `model_outputs/metrics/model_comparison.csv` and `.png`.

**Diver Notes:** The comparison shows whether nonlinear models actually improve future prediction.

**Counter Notes:** The final report treats Ridge/Lasso outperforming tree models as an empirical finding, not an error.

**Verdict:** PASS (5/5)

### T5-A - XGBoost SHAP

**Objective:** Retain nonlinear interpretability for comparison.

**Action Taken:** Fit XGBoost on the scoped training set and computed SHAP values on the test set.

**Decision Logic:** Tree SHAP provides a nonlinear contrast even though Ridge is the best model.

**Inputs and Outputs:** Input: scoped matrix; Output: `model_outputs/shap/shap_importance.csv`, SHAP plots.

**Diver Notes:** Mean absolute SHAP values support a compact feature importance table.

**Counter Notes:** The SHAP result is not presented as the sole explanation for the selected model.

**Verdict:** PASS (5/5)

### T5-B - Interpretability Synthesis

**Objective:** Connect Ridge coefficients and SHAP into one review story.

**Action Taken:** The final report and dashboard include both coefficient and SHAP tables.

**Decision Logic:** This satisfies the proposal's SHAP requirement while respecting the empirical best model.

**Inputs and Outputs:** Input: `ridge_coefficients.csv`, `shap_importance.csv`; Output: report and dashboard interpretability sections.

**Diver Notes:** Linear and nonlinear explanations are shown side by side.

**Counter Notes:** Reviewer confusion is reduced by clearly labeling Ridge as primary and SHAP as contrast.

**Verdict:** PASS (5/5)

### T6-A - Priority Scoring

**Objective:** Convert predictions into actionable ranking.

**Action Taken:** Computed expected donated count as predicted donation ratio times invoice count, then min-max normalized it.

**Decision Logic:** The formula follows the human-approved plan and favors high predicted ratio in high-volume counties.

**Inputs and Outputs:** Input: best-model test predictions; Output: `model_outputs/priority_scores.csv`.

**Diver Notes:** The output includes county code/name, month, actual and predicted ratio, invoice count, expected count, normalized score, rank, and model name.

**Counter Notes:** Rank ordering is deterministic and can be recomputed from the score column.

**Verdict:** PASS (5/5)

### T6-B - Final Report

**Objective:** Create the human-facing written project report.

**Action Taken:** Generated `reports/final_project_report.md` from current artifacts.

**Decision Logic:** The report should be reproducible from model outputs and should not depend on stale legacy results.

**Inputs and Outputs:** Input: all formal model outputs; Output: `reports/final_project_report.md`.

**Diver Notes:** The report includes scope, model metrics, priority, clustering, Ridge explanation, SHAP, and caveats.

**Counter Notes:** It explicitly reconciles proposal expectations with empirical Ridge/Lasso results.

**Verdict:** PASS (5/5)

### T6-C - Static Dashboard

**Objective:** Provide a compact visual overview for review.

**Action Taken:** Generated a single-file static HTML dashboard under `reports/`.

**Decision Logic:** A static dashboard is enough for final delivery and avoids deployment complexity.

**Inputs and Outputs:** Input: formal tables and plots; Output: `reports/dashboard.html`.

**Diver Notes:** The dashboard covers priority, clustering, model metrics, interpretability, and limitations.

**Counter Notes:** All referenced assets live under `model_outputs/` with stable relative paths.

**Verdict:** PASS (5/5)

### T6-D - Gate Package

**Objective:** Write final gate verdicts and acceptance caveats.

**Action Taken:** Generated Gate 0 through Gate 6 verdict JSON files.

**Decision Logic:** Machine-readable gate files make the final delivery auditable.

**Inputs and Outputs:** Input: readiness, artifacts, split, metrics; Output: `model_outputs/gate0_verdict.json` through `gate6_verdict.json`.

**Diver Notes:** The package is complete for available in-scope counties.

**Counter Notes:** The only conditional caveat is missing data for 連江縣.

**Verdict:** CONDITIONAL (3/5)
