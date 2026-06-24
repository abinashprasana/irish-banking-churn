# Model Card: Irish Banking Customer Churn Predictor

This model card details the model type, training characteristics, performance metrics, and ethical considerations for the customer churn classifier built for Irish retail banking.

## Model Details
- **Model name:** Irish Banking Customer Churn Predictor
- **Model type:** XGBoost Binary Classifier
- **Version:** 1.0
- **Date:** June 2026
- **Framework:** XGBoost 2.x, scikit-learn, imbalanced-learn
- **License:** MIT

## Intended Use
- **Primary use:** Predicting retail banking customer churn risk for retail banks in the Republic of Ireland.
- **Intended users:** Data science/analytics teams and client relationship managers in retail banking and fintech.
- **Out-of-scope:** Not intended for automated credit underwriting, loan approvals, credit scoring, or any financial transaction requiring authorization by the Central Bank of Ireland or the UK Financial Conduct Authority (FCA).

## Training Data
- **Source:** Synthetic dataset containing 10,000 customer records generated using statistical parameters modeled on actual published Irish retail banking statistics.
- **Context:** Reflects the KBC Bank Ireland and Ulster Bank market exits (2022-2023) where ~1.2 million accounts migrated. In 2025-2026, those customers are now in their third or fourth year with a new provider; the synthetic population models the persisting lower-loyalty behaviours that characterise this ongoing transition period, as institutional trust typically takes 3-5 years to rebuild after a forced migration.
- **Sampling Strategy:** Class imbalance handled via SMOTEENN (SMOTE + Edited Nearest Neighbors) on training data only. Resampled from a starting distribution of `6,320` negative and `1,680` positive samples to a balanced set of `2,724` negative and `3,662` positive samples.

## Evaluation Data
- **Size:** 20% stratified holdout test split (2,000 records: `1,580` retained, `420` churned).
- **Distribution:** Original imbalanced target distribution (no resampling applied to the evaluation set to ensure realistic performance expectations).

## Performance
The model was evaluated against baseline classifiers (Logistic Regression and Random Forest) on the original, imbalanced holdout test set. Metric values achieved by the deployed XGBoost model are:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Selected)** | **0.8990** | **0.7080** | **0.8833** | **0.7860** | **0.9593** | **0.8420** |
| Random Forest | 0.8790 | 0.6660 | 0.8500 | 0.7469 | 0.9438 | 0.7708 |
| Logistic Regression | 0.8385 | 0.5883 | 0.7690 | 0.6667 | 0.9011 | 0.7403 |

*Note: For imbalanced data, PR-AUC (0.8420) is the key metric showing high precision and recall stability compared to the baseline.*

### Top 5 Most Important Features (by Mean Absolute SHAP Value)
1. `num_products` (Mean Absolute SHAP: 2.841)
2. `months_since_switching` (Mean Absolute SHAP: 1.028)
3. `has_direct_debits` (Mean Absolute SHAP: 0.883)
4. `tenure_months` (Mean Absolute SHAP: 0.838)
5. `has_savings_goal` (Mean Absolute SHAP: 0.529)

## Ethical Considerations
- **Data Privacy:** Synthetic data only. No personally identifiable information (PII) or real customer data was used, eliminating privacy breach risks.
- **Explainability (EU AI Act Article 86):** Model interpretations are structured using SHAP (global feature importance, local waterfall plots) and DICE counterfactuals. This provides a transparent mathematical framework to satisfy customers' rights to an explanation for automated financial assessments.
- **EBA Guidelines on Internal Governance:** Under European Banking Authority guidelines, automated AI decisions must have human-in-the-loop oversight. Model predictions should act as decision-support alerts for relationship managers, rather than triggering automated account freezing or negative customer treatment.

## Limitations
- **Synthetic Nature:** While parameters are realistic, real-world customer behaviors may deviate from synthetic distributions.
- **Geographic Lock-in:** Modeled specifically on the retail banking dynamics of the Republic of Ireland (e.g. unique switching friction from the dual bank exits). It will not generalize to other EU, UK, or global retail banking markets without retraining.
- **Macroeconomic Factors:** The current iteration does not incorporate interest rate fluctuations, housing market parameters, or inflationary pressures, which significantly drive financial migrations. As the Irish market normalises post-2027, the relevance of migration-specific variables will gradually decay and model weights will require recalibration.

## Caveats and Recommendations
- **Retraining Cadence:** Retrain model quarterly to reflect changing customer movement trends. As the post-2022 switching landscape continues to settle through 2025-2026 and beyond, the signal strength of migration-specific features (`months_since_switching`, `was_kbc_ulster_customer`) will decay and model weights will need recalibration to avoid over-weighting stale signals.
- **Data Drift:** Closely monitor features related to secondary digital accounts (e.g., uses_digital_bank_secondary). Revolut/N26 usage levels in Ireland are shifting rapidly.
- **Human Validation:** All counterfactual suggestions (e.g., asking a client to set up a savings goal or add a mortgage) are mathematical suggestions and must be verified by a banking advisor before direct client outreach.
