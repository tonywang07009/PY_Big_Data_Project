

# **MecDonate:**  **A Machine Learning Framework for**  **County-Level Invoice Donation**  **Ratio Prediction in Taiwan**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

## Project proposal

Group: 6th  
Student ID, Name  
114C73021, 凃辛河  
114C73016, 張侑傑  
114C73032, 王界棠  
114C73008**,** 湯雁茹   
114C73037, 徐立帆

# **Contents** {#contents}

[Contents	2](#contents)

[1\. Introduction	1](#1.-introduction)

[2\.  Materials and Methods	2](#2.-materials-and-methods)

[2.1 Database	2](#2.1-database)

2[.1.1 Database Sources	2](#2.1.1.-database-sources)

[2.1.2 Data Specifications	2](#2.1.2.-data-specifications)

2[.1.2 Data Preprocessing Plan	3](#2.1.3.-data-preprocessing-plan)

[2.2 Model	3](#2.2-model)

[2.3 Methods	4](#2.3-methods)

[3\. Expected Result	5](#3.-expected-result)

[3.1 Clustering Results	5](#3.1-clustering-results)

[3.2 Prediction Performance	5](#3.2-prediction-performance)

[3.3 Feature Importance (SHAP)	5](#3.3-feature-importance-\(shap\))

[Reference	6](#reference)

# **1\. Introduction**  {#1.-introduction}

Taiwan's electronic invoice system has been widely adopted since 2009, with over 630,000 businesses participating by the end of 2024\. Built on this infrastructure, the Love Thermometer (愛心溫度計) \[1\] program allows consumers to donate invoices to non-profit organizations at the point of purchase.

However, donation behavior varies considerably across counties, shaped by differences in income levels, housing burden, and digital infrastructure. Zhang et al. (2023) demonstrate that household income positively influences charitable giving, while housing burden negatively moderates donation willingness — supporting both as causal antecedents in our framework\[2\]. Urban counties such as Taipei generate far higher invoice volumes than remote regions such as Lienchiang and Yunlin, creating significant data imbalance across regions.

These distributional differences pose a modeling challenge: a single pooled model risks overlooking county-level heterogeneity, while county-specific models suffer from data scarcity in low-volume regions. To address this, we propose a unified framework combining K-Means clustering and supervised regression, with SHAP analysis ensuring model interpretability.\[3\] The framework is validated on six counties — three metropolitan (Taipei, Taichung, Kaohsiung) and three non-metropolitan (Lienchiang, Hualien, Yunlin) — to derive priority scores supporting charitable collection route planning.

# 

# **2\.  Materials and Methods** {#2.-materials-and-methods}

## **2.1 Database**  {#2.1-database}

This study utilizes four public datasets released on Taiwan's government open data platforms. The data has been structurally processed and is characterized by periodic updates, ensuring both data freshness and reproducibility for model training. Given the inherent lag in government statistical indicators, this study adopts a "mixed-frequency data processing" strategy: real-time invoice statistics (2019–2024) serve as dynamic predictive features, while annually released income and expenditure indicators (currently updated to 2025)\[4\] are interpolated using annual constant values or linear trend prediction methods to align with the high-frequency time-series model.

### **2.1.1. Database Sources**  {#2.1.1.-database-sources}

* **High-Frequency Data (Real-time updates):**  
  * **Consumption Channel Invoice Statistics:** From the Ministry of Finance's E-Invoice Platform. Captures consumption intensity across channels and counties, used to eliminate seasonal interference\[5\].  
  * **Love Thermometer — County/City E-invoice Issuance Statistics:** From the Ministry of Finance's "Love Thermometer" platform. Serves as the target variable, representing the regional invoice donation quantity\[6\].  
* **Low-Frequency Data (Annual releases):**  
  * **Housing Affordability Indicators:** From the Ministry of the Interior (MOI). Contains mortgage burden rates across major municipalities\[7\].

### **2.1.2. Data Specifications** {#2.1.2.-data-specifications}

* **Records:** This study integrates structured data spanning several years across all counties and cities. After applying lag feature processing, the effective sample size is moderately reduced, yielding a sufficient dataset for model training.


### **2.1.3. Data Preprocessing Plan** {#2.1.3.-data-preprocessing-plan}

* **Imputation & Normalization:** Missing values are handled via linear interpolation, followed by Z-score standardization applied to each county/city's feature matrix.  
* **Feature Optimization:** STL decomposition is applied to isolate and remove seasonal and festival effects, enabling cleaner analysis of how economic and housing factors influence donation behavior.

## **2.2 Model** {#2.2-model}

This study uses two types of models. For clustering, **K-Means** groups counties by feature similarity. The optimal K is selected using the Elbow Method and validated with Silhouette Score. PCA is then applied for 2D visualization. For prediction, three regression models are compared: **Linear Regression** as the baseline, **Ridge/Lasso** to handle multicollinearity, and **XGBoost/Random Forest** to capture non-linear relationships.

## 

## **2.3 Methods** {#2.3-methods}

Data is collected from Taiwan's Ministry of Finance electronic invoice records (2019–2024).\[6\] The prediction target is the monthly invoice donation ratio per county. Preprocessing includes missing value imputation, STL seasonal decomposition, and Z-score normalization.

Since the six counties vary significantly in population, economic activity, and digital infrastructure — ranging from metropolitan cities such as Taipei to remote counties such as Lienchiang — a single model trained on pooled data may fail to capture these distributional differences. To address this, county type (metropolitan / remote / agricultural) and county one-hot encoding are included as additional features, allowing the model to account for county-level heterogeneity without training separate models per county.

Features are aggregated into a county-level time-series matrix and fed into a single global model. A time-based train/test split is used (2019–2022 for training, 2023 onward for testing) to prevent data leakage. Models are evaluated using RMSE and R². SHAP analysis is applied to interpret feature importance and explain model predictions across different county types.

# **3\. Expected Result** {#3.-expected-result}

## **3.1 Clustering Results** {#3.1-clustering-results}

We expect K-Means clustering to group the six counties into two or three distinct clusters. Metropolitan counties (Taipei, Taichung, Kaohsiung) are likely to form one group due to their higher invoice volumes and donation ratios, while remote and agricultural counties (Lienchiang, Hualien, Yunlin) may form separate groups reflecting lower activity and different behavioral patterns. PCA visualization should show clear separation between these groups.

## **3.2 Prediction Performance** {#3.2-prediction-performance}

We expect XGBoost or Random Forest to outperform the linear baselines (Ridge, Lasso) in terms of RMSE and R², as donation behavior likely involves non-linear relationships with economic and behavioral features. The model is expected to achieve a reasonably high R² on the test set (2023 onward), indicating that county-level donation ratios can be predicted with acceptable accuracy using the selected features.

## **3.3 Feature Importance (SHAP)** {#3.3-feature-importance-(shap)}

SHAP analysis is expected to show that behavioral features are the most influential predictors, suggesting that past donation behavior is the strongest indicator of future behavior. Economic indicators such as average income and housing burden ratio are expected to have moderate importance, while county type is expected to help the model distinguish between urban and rural donation patterns.

# **Reference** {#reference}

\[1\] Ministry of Finance, Taiwan, "A Study on the Use of Electronic Invoices,"   2025

\[2\] Zhang, K., Cao, B., Zhang, Y., & Han, Y. (2023). A study on the influence of personality characteristics on household charitable donation behavior in China. *PLOS ONE, 18*(5), e0284798. https://doi.org/10.1371/journal.pone.0284798

\[3\] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems (NIPS), vol. 30, Long Beach, CA, USA, Dec. 2017, pp. 4765–4774.

\[4\] Ministry of Finance, Taiwan, "主題分析-消費通路發票統計 (統計111年1月至114年11月)," E-Invoice Integration Service Platform, \[Online\]. Available: https://www.einvoice.nat.gov.tw/portal/ods/ODS303E/main/0DBDAF6E-5E44-49A8-8528-E22648B2F32E/131

\[5\] Ministry of Finance, Taiwan, "縣市電子發票消費通路資料," E-Invoice Integration Service Platform, \[Online\]. Available: https://www.einvoice.nat.gov.tw/portal/ods/ods303E\_detail/main/55B7CB0E-8182-4FDE-A5A0-085A64625D02/132/3975BF8A-2CF0-449C-A64D-0E121C6566EF/136

\[6\] Ministry of Finance, Taiwan, "縣市電子發票愛心捐贈統計 (Love Thermometer Data)," E-Invoice Integration Service Platform, \[Online\]. Available: https://www.einvoice.nat.gov.tw/portal/ods/ods303E\_detail/main/55B7CB0E-8182-4FDE-A5A0-085A64625D02/132/428D97C9-6BA9-48E5-AC33-8CA1C651905B/13

\[7\] Directorate-General of Budget, Accounting and Statistics (DGBAS), Taiwan, "家庭收支調查報告 (Family Income and Expenditure Survey)," Government Open Data Platform, \[Online\]. Available: https://data.gov.tw/dataset/24826.

\[8\] Ministry of the Interior, Taiwan, "房價負擔能力統計," Real Estate Information Platform, \[Online\]. Available: https://pip.moi.gov.tw/Publicize/Info/E1050 

\[9\] National Development Council, Taiwan, ,Government Open Data Platform, \[Online\]. Available:https://data.gov.tw/datasets/search?p=1\&size=10\&s=\_score\_desc\&rtt=1751  
