# Project Brief

## Purpose

MecDonate predicts monthly invoice donation ratios at the county level in Taiwan and converts predictions into priority scores for charitable collection route planning.

## Source Material

- Main proposal: `source_materials/group_six.md`
- PDF proposal: `source_materials/group_six.pdf`
- Existing transformed data utility: `source_materials/data_transformer.py`

## Research Scope

- Counties: Taipei, Taichung, Kaohsiung, Lienchiang, Hualien, Yunlin
- Target: `donation_ratio = donated_count / total_count`
- Validation: train on 2019-2022, test on 2023 onward
- Methods: K-Means, PCA, linear baselines, Ridge, Lasso, XGBoost, Random Forest, SHAP

## Non-Negotiable Constraints

- Do not shuffle time-series rows.
- Do not modify files under `data/raw/`.
- Ask the human in Traditional Chinese when a high-impact ambiguity appears.
- Keep persisted project documents in English.

