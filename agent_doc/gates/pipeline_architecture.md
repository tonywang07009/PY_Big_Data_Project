# Pipeline Architecture (Gate 1 to Gate 6)

This diagram reflects the current repository structure and excludes Gate 0 by design.

```mermaid
flowchart TD
    A[data/raw/encoded_ml_dataset.csv] --> B[Gate 1<br/>src/features.py]
    B --> C[model_outputs/county_month_matrix.csv]
    C --> D[Gate 2<br/>src/clustering.py]
    C --> E[Gate 3<br/>src/model.py]
    C --> F[Gate 5<br/>src/shap_analysis.py]
    D --> G[cluster_assignments.csv<br/>county_cluster_v2.csv<br/>PCA figures]
    E --> H[model comparison<br/>prediction scatter<br/>priority scores]
    F --> I[shap_importance.csv<br/>shap_summary.png]
    G --> J[Gate 6<br/>src/final_delivery.py]
    H --> J
    I --> J
    E --> K[Gate 4<br/>time-based split evidence]
    K --> J

    A --> L[Supplementary preprocessing<br/>split_scale_by_county.py]
    L --> M[data/county_zscore_split/]
    M --> N[Supplementary model<br/>train_donation_ratio_county_scaled_xgboost.py]
    M --> O[Supplementary rolling CV<br/>train_donation_ratio_county_scaled_year_cv.py]
    N --> P[model_outputs/donation_ratio_county_scaled_xgboost/]
    O --> Q[model_outputs/donation_ratio_county_scaled_year_cv/]

    J --> R[reports/final_project_report.md]
    J --> S[reports/dashboard.html]
    J --> T[reports/tour_guide.md]
```
