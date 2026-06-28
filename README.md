<div align="center">

# 🏦 Irish Banking Customer Churn Predictor

**An explainable machine learning system built around the largest account migration event in Irish banking history.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosted-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://abinashprasana-irish-banking-churn-app-aidovf.streamlit.app/)
[![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.959-2ea44f?style=for-the-badge)](.)
[![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=for-the-badge)](.)

<br/>

*KBC Bank Ireland & Ulster Bank exits 2022–2023 · 1.2M accounts migrated · SMOTEENN · SHAP · DiCE · EU AI Act Article 86*

</div>

---

## 📖 What This Project Is

Between 2022 and 2023, KBC Bank Ireland and Ulster Bank (NatWest Group) both pulled out of the Irish retail banking market. That forced over 1.2 million customers to close their accounts and find a new bank, all within a short window. It caused real chaos: around 60% of people who switched reported serious problems, things like direct debits failing, money not transferring correctly, and poor customer support throughout.

I built this project because those customers haven't just settled in and moved on. Research shows institutional trust takes 3 to 5 years to rebuild after a forced switch, and we're only in year 3 or 4 now. That means Irish banks are still sitting on an unusually high churn risk that won't calm down before 2027. I wanted to build something that actually reflects that. Not a generic churn model, but one calibrated to what's happening in this specific market right now.

The result is an XGBoost classifier that predicts which customers are most likely to leave, SHAP values that explain why, and DiCE counterfactuals that suggest what a relationship manager could actually do about it. The whole thing runs in a five-tab Streamlit dashboard.

---

## 🎬 Live Demo

**[→ Open the live app on Streamlit Community Cloud](https://abinashprasana-irish-banking-churn-app-aidovf.streamlit.app/)**

No setup needed, runs directly in the browser.

---

## 🗃️ Dataset

<div align="center">

| Detail | Value |
|:---|:---|
| 📦 Type | Fully synthetic, statistically calibrated |
| 📋 Total records | 10,000 customer profiles |
| 🎯 Churn rate | ~21% (2,100 churners, 7,900 retained) |
| 🏗️ Features | 19 input variables |
| 📐 Train / test split | 80% training (8,000) / 20% test (2,000), stratified |
| 🏦 Migration flag | ~15% former KBC Bank Ireland or Ulster Bank customers |
| 😤 Switching difficulty | 60% of migrated customers experienced friction |
| 📊 Sources | Central Bank of Ireland, CCPC 2022 account migration survey |

</div>

All records are synthetic. No real customer data was used. I built the statistical parameters around actual published figures from the Central Bank of Ireland and the CCPC's 2022 account migration survey, so the distributions reflect what the real market looks like rather than being made up.

What makes this dataset different from a standard churn dataset is the Irish-specific columns: `was_kbc_ulster_customer`, `months_since_switching`, `experienced_switching_difficulty`, and `uses_digital_bank_secondary` (Revolut / N26 usage). Those four features are what lets the model capture the migration-driven risk that standard banking churn models would miss entirely.

---

## 🧠 Pipeline Architecture

```mermaid
flowchart TD
    A["📁 Data Generation\ngenerate_data.py\n10,000 synthetic records · 19 features\nCalibrated to CBI & CCPC statistics"]
    B["🔧 Preprocessing\nLabelEncoder · Boolean cast to int\nStratified 80/20 train / test split"]
    C["⚖️ SMOTEENN\nTraining set only\n6,320 neg + 1,680 pos  →  2,724 neg + 3,662 pos\nTest set left at original 79% / 21%"]
    D1["Logistic Regression\nBaseline"]
    D2["Random Forest\nEnsemble baseline"]
    D3["⚡ XGBoost\nSelected model\n200 est · depth 6 · lr 0.05"]
    E["📊 Model Comparison\nAccuracy · Precision · Recall\nF1 · ROC-AUC · PR-AUC"]
    F1["🔍 SHAP TreeExplainer\nGlobal beeswarm & bar plots\nLocal waterfall chart"]
    F2["🎲 DiCE Counterfactuals\nXGBoostClassifierWrapper guard\n3 diverse scenarios per high-risk customer\nLocked: age · switching history"]
    G["🏦 Streamlit Dashboard\nTab 1 Overview  ·  Tab 2 Data Explorer\nTab 3 Model Performance\nTab 4 SHAP Explainability  ·  Tab 5 Risk Predictor"]

    A --> B --> C
    C --> D1 & D2 & D3
    D1 & D2 & D3 --> E
    D3 --> F1 & F2
    E & F1 & F2 --> G

    style A fill:#1f4e79,color:#ffffff,stroke:#1f4e79
    style B fill:#2e75b6,color:#ffffff,stroke:#2e75b6
    style C fill:#c55a11,color:#ffffff,stroke:#c55a11
    style D1 fill:#404040,color:#ffffff,stroke:#404040
    style D2 fill:#404040,color:#ffffff,stroke:#404040
    style D3 fill:#375623,color:#ffffff,stroke:#375623
    style E fill:#375623,color:#ffffff,stroke:#375623
    style F1 fill:#7030a0,color:#ffffff,stroke:#7030a0
    style F2 fill:#7030a0,color:#ffffff,stroke:#7030a0
    style G fill:#c00000,color:#ffffff,stroke:#c00000
```

---

## 📊 Model Performance

I trained three classifiers and compared them on the original imbalanced test set. XGBoost came out clearly ahead on every metric.

<div align="center">

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Selected)** | **0.8990** | **0.7080** | **0.8833** | **0.7860** | **0.9593** | **0.8420** |
| Random Forest | 0.8790 | 0.6660 | 0.8500 | 0.7469 | 0.9438 | 0.7708 |
| Logistic Regression | 0.8385 | 0.5883 | 0.7690 | 0.6667 | 0.9011 | 0.7403 |

</div>

I used PR-AUC as the primary metric rather than accuracy because the test set is imbalanced. A model that labels every customer as "retained" would still hit 79% accuracy, which is useless. PR-AUC measures how well the model performs across all possible decision thresholds without that distortion. XGBoost's **0.842** PR-AUC is a **+0.102 gain** over Logistic Regression, which is a meaningful jump on an imbalanced problem.

<div align="center">

| Metric vs. LR Baseline | XGBoost Gain |
|:---|:---:|
| F1 Score | **+0.119** |
| ROC-AUC | **+0.058** |
| PR-AUC | **+0.102** |

</div>

---

## 🔍 Feature Importance (SHAP)

SHAP Shapley values were computed using `TreeExplainer` on the full 2,000-record test set. The top 5 features driving churn predictions across the portfolio are:

<div align="center">

| Rank | Feature | Mean Absolute SHAP | What It Means |
|:---:|:---|:---:|:---|
| 1 | `num_products` | **2.841** | The single strongest retention anchor. Customers with only one product have nothing tying them to the bank. |
| 2 | `months_since_switching` | **1.028** | How recently the customer was forced to switch. More recent = higher risk. |
| 3 | `has_direct_debits` | **0.883** | Direct debits create real friction to leave. No direct debits is a clear warning sign. |
| 4 | `tenure_months` | **0.838** | Longer relationships reduce switching intent, regardless of how they started. |
| 5 | `has_savings_goal` | **0.529** | Customers with a savings goal are more engaged and less likely to leave. |

</div>

The fact that `num_products` and `months_since_switching` are the top two features is exactly what I expected. Product depth is the main thing keeping customers in place, and the recency of the forced migration is still the dominant risk factor, which is why this model makes sense specifically for the current Irish market moment.

---

## ⚡ Sample Counterfactual Explanations (DiCE)

For any customer the model flags above 50% churn probability, the Risk Predictor tab generates three counterfactual scenarios: the smallest set of changes that would bring them below the churn threshold. These are meant to give relationship managers something concrete to work with, not just a risk score.

<details>
<summary>🔴 Sample output — high-risk customer at 87% churn probability</summary>

```
Input profile:
  age=34 · tenure=8 months · num_products=1 · has_direct_debits=False
  was_kbc_ulster_customer=True · months_since_switching=9
  has_savings_goal=False · credit_score_band=Low

Scenario 1 — Add products and set up direct debits:
  num_products:       1  →  3
  has_direct_debits:  0  →  1
  direct_debit_count: 0  →  4
  → Predicted outcome: Retained (12% risk)

Scenario 2 — Open a savings goal and increase transaction activity:
  has_savings_goal:          0  →  1
  monthly_transaction_count: 11  →  34
  → Predicted outcome: Retained (31% risk)

Scenario 3 — Increase balance and transaction volume:
  monthly_balance_eur:       420  →  3,100
  monthly_transaction_count: 11   →  52
  → Predicted outcome: Retained (44% risk)
```

> These are model-generated suggestions. A relationship manager should review them before any customer contact.
</details>

---

## 📁 Project Structure

```
irish-banking-churn/
├── 📄 app.py                         Streamlit five-tab dashboard
├── 📋 requirements.txt               Project dependencies
├── 📄 model_card.md                  Model card (metrics, limitations, regulatory context)
│
├── 📂 data/
│   ├── generate_data.py              Synthetic dataset generator (10,000 records)
│   └── irish_banking_churn.csv       Generated dataset [git-ignored]
│
├── 📂 models/
│   ├── train_model.py                Training pipeline — preprocessing, SMOTEENN,
│   │                                 model comparison, SHAP, DiCE verification
│   └── xgboost_churn_model.pkl       Serialized model bundle [git-ignored]
│
└── 📂 assets/
    ├── shap_summary_plot.png         Global SHAP beeswarm plot [git-ignored]
    └── shap_bar_plot.png             Global SHAP bar plot [git-ignored]
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/abinashprasana/irish-banking-churn.git
cd irish-banking-churn
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

**3. Generate the synthetic dataset**
```bash
python data/generate_data.py
```
This creates `data/irish_banking_churn.csv` with 10,000 records at a ~21% churn rate.

**4. Train the model**
```bash
python models/train_model.py
```
Trains all three classifiers, prints the comparison table, saves the XGBoost model to `models/xgboost_churn_model.pkl`, and exports both SHAP plots to `assets/`.

**5. Launch the dashboard**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. The Risk Predictor tab needs the trained model file to be present before it will run predictions.

---

## ⚠️ Limitations

This is a student portfolio project, so I want to be upfront about what it is and isn't.

The data is synthetic. I calibrated it carefully against real published statistics, but synthetic data still can't replicate the full complexity of real customer behaviour. Before this model could be used in production, it would need to be retrained on actual bank data.

The model also doesn't know anything about the broader economy. Interest rates, the housing market, inflation: all of these push people to switch banks, and none of that is in here. That's a gap.

The two Irish-specific features (`was_kbc_ulster_customer`, `months_since_switching`) will become less useful over time as the post-2022 migration period settles. Once the market normalises post-2027, those signals will decay and the model weights will need recalibrating.

<div align="center">

| 🔧 If I were to extend this | 📈 What it would add |
|:---|:---|
| Real bank transaction data | Actual behavioural signal instead of simulated |
| Macroeconomic features | Sensitivity to interest rates and housing market |
| Quarterly retraining | Keeps up as the market normalises post-migration |
| Larger feature set | More granular engagement and product-use signals |
| Online learning | Catches drift without needing full retrains |

</div>

---

## 🏛️ Regulatory Context

Under **Article 86 of the EU AI Act**, customers have a right to an explanation when an automated system makes a significant decision about their financial situation. Flagging someone as high-risk for churn can lead to changes in what products they're offered or how they're treated, so that needs to be explainable. The SHAP waterfall chart in the Risk Predictor tab gives a mathematical breakdown of exactly what pushed any individual prediction in either direction, and the DiCE counterfactuals show what would have to change to get a different outcome.

The **EBA Guidelines on Internal Governance** also require human oversight for automated decisions in financial services. The dashboard is built as a decision-support tool for relationship managers, not a system that takes automatic action. Every counterfactual output carries a note making that clear.

Full details on the model, its validation, and ethical considerations are in [model_card.md](model_card.md).

---

## 👤 Author

**Abinash Prasana Selvanathan**

*If you found this useful, feel free to ⭐ star the repo.*
