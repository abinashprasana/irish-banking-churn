# Model Card: Irish Banking Customer Churn Predictor

This model card details the model type, training characteristics, performance metrics, and ethical considerations for the customer churn classifier built for Irish retail banking.

## Model Details
- **Model name:** Irish Banking Customer Churn Predictor
- **Model type:** XGBoost Binary Classifier
- **Version:** 1.0
- **Date:** June 2026
- **Framework:** XGBoost through its scikit-learn API, with scikit-learn and imbalanced-learn

## Intended Use
- **Primary use:** Demonstrating churn modelling, explanation, and governed recommendation workflows in a synthetic Irish retail banking scenario.
- **Intended users:** People reviewing machine learning and decision-support system design.
- **Out-of-scope:** Not intended for automated credit underwriting, loan approval, credit scoring, customer contact, or initiating any financial transaction.

## Training Data
- **Source:** A locally generated synthetic dataset containing 10,000 customer records. The generator borrows the [CCPC figure of 60 percent](https://www.ccpc.ie/about-us/advocacy-and-research/research/publication-details/ccpc-switching-research-%28phase-2%29) as the switching difficulty probability for migration flagged synthetic records. The CCPC respondents had an open KBC or Ulster current account, or had closed one within the previous six months. Applying their survey figure to the generated subgroup is a modelling assumption, not a subgroup estimate reported by the CCPC. Central Bank data supplies historical account migration context. The 15 percent migration flag, 21 percent churn target, other distributions, and churn label rule are also constructed assumptions.
- **Context:** Uses the KBC Bank Ireland and Ulster Bank market exits as its setting. The [Central Bank of Ireland](https://www.centralbank.ie/statistics/data-and-analysis/credit-and-banking-statistics/account-migration-statistics) recorded 1,167,219 current and deposit account closures at the two exiting banks between the start of 2022 and the end of June 2023. The synthetic population explores possible post migration loyalty patterns; it does not measure the current behaviour of those account holders.
- **Sampling Strategy:** Class imbalance handled via SMOTEENN (SMOTE + Edited Nearest Neighbors) on training data only. The training set changed from `6,320` negative and `1,680` positive samples to a resampled set of `2,652` negative and `3,624` positive samples.

## Evaluation Data
- **Size:** 20% stratified holdout test split (2,000 records: `1,580` retained, `420` churned).
- **Distribution:** Original generated target distribution. No resampling was applied to the evaluation set.

## Performance
The model was evaluated against baseline classifiers (Logistic Regression and Random Forest) on the original, imbalanced holdout test set. Metric values achieved by the deployed XGBoost model are:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Average precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Selected)** | **0.8085** | **0.5377** | **0.6286** | **0.5796** | **0.8239** | **0.5329** |
| Random Forest | 0.7985 | 0.5183 | 0.5738 | 0.5446 | 0.8159 | 0.4862 |
| Logistic Regression | 0.7525 | 0.4346 | 0.5929 | 0.5015 | 0.7717 | 0.4543 |

*Note: Average precision is reported because the positive churn class is less common. XGBoost scores 0.5329 on the holdout sample.*

### Real Data Benchmark
The same recipe run on the UCI Bank Marketing dataset (Moro, Cortez and Rita, 2014, CC BY 4.0; 45,211 real customers, 11.7 percent positive, target is term deposit subscription, `duration` dropped) gave XGBoost a ROC-AUC of 0.7779 and an average precision of 0.3829. This checks the method on real bank data. It is not a churn result. Full results are in `results/benchmark_uci_bank_marketing.json`.

### Top 5 Most Important Features (by Mean Absolute SHAP Value)
1. `num_products` (Mean Absolute SHAP: 1.943)
2. `account_type` (Mean Absolute SHAP: 0.719)
3. `has_direct_debits` (Mean Absolute SHAP: 0.683)
4. `months_since_switching` (Mean Absolute SHAP: 0.521)
5. `tenure_months` (Mean Absolute SHAP: 0.476)

## Ethical Considerations
- **Data Privacy:** The dataset is synthetic and contains no real customer records or personally identifiable information.
- **Explainability (EU AI Act Article 86):** [Article 86](https://eur-lex.europa.eu/eli/reg/2024/1689/oj#art_86) sets out a right to a clear and meaningful explanation for some decisions based on the output of an AI system listed in Annex III and classified as high risk. The right only applies when the other conditions in the Article are met. The Act has a general application date of 2 August 2026. [Regulation (EU) 2026/1744](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32026R1744) entered into force on 27 July 2026 and moved the application date for Sections 1 to 3 of Chapter III, except Article 6(5), for systems classified as high risk under Article 6(2) and Annex III to 2 December 2027. This prototype does not claim to fall within Article 86. SHAP and DiCE are model inspection tools, not evidence of legal compliance.
- **EBA Guidelines on Internal Governance:** The [in-force Guidelines on internal governance under CRD](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/internal-governance/guidelines-internal-governance-under-crd) apply to institutions within their stated scope and address responsibilities, risk management, and internal controls. They do not prescribe this retention workflow. Advisor review and the policy gate are engineering choices in this application and have not been assessed as evidence of regulatory compliance.

## Limitations
- **Synthetic Nature:** One switching difficulty probability borrows a published survey figure, while its subgroup mapping, the migration share, churn target, many other distributions, and the churn label rule were constructed for this study. They should not be treated as observed customer behaviour.
- **Geographic Lock-in:** The generated scenario uses Irish banking context. Performance should not be assumed to generalise to any live Irish or international customer population.
- **Macroeconomic Factors:** The current iteration does not include interest rates, housing market conditions, or inflation. If the model were adapted to real data, those omitted factors and the continuing relevance of migration-specific fields would need to be tested.

## Caveats and Recommendations
- **Retraining Cadence:** No fixed cadence can be justified from synthetic data alone. A production owner should set retraining and review intervals from observed drift, model performance, and governance requirements.
- **Data Drift:** If adapted to live data, monitor all features for drift, including `months_since_switching`, `was_kbc_ulster_customer`, and `uses_digital_bank_secondary`.
- **Human Validation:** Counterfactual examples are mathematical candidates rather than prescribed customer actions. Any real use would require suitability, eligibility, and governance review outside this application before customer contact.
