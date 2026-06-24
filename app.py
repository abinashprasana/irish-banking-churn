import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
import shap
import dice_ml

st.set_page_config(
    page_title="Irish Banking Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    padding: 1rem 1.25rem;
    border-radius: 8px;
    background: #f8f9fa;
    border: 1px solid #e9ecef;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
div[data-testid="stInfo"] {
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join('models', 'xgboost_churn_model.pkl')
DATA_PATH = os.path.join('data', 'irish_banking_churn.csv')

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    st.error("Model file or dataset not found. Run models/train_model.py first.")
    st.stop()

payload = joblib.load(MODEL_PATH)
xgb_model = payload['model']
encoders = payload['encoders']
feature_names = payload['feature_names']
continuous_features = payload['continuous_features']

df_data = pd.read_csv(DATA_PATH)


class XGBoostClassifierWrapper:
    """
    Casts DataFrame columns back to their training dtypes before each prediction.
    DiCE mutates column types during counterfactual search, which breaks the native
    XGBoost predictor without this guard.
    """
    def __init__(self, model, feature_names, dtypes):
        self.model = model
        self.feature_names = feature_names
        self.dtypes = dtypes
        self.classes_ = model.classes_

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_cast = X.copy()
        for col in self.feature_names:
            X_cast[col] = pd.to_numeric(X_cast[col], errors='coerce').fillna(0).astype(self.dtypes[col])
        return self.model.predict_proba(X_cast)

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_cast = X.copy()
        for col in self.feature_names:
            X_cast[col] = pd.to_numeric(X_cast[col], errors='coerce').fillna(0).astype(self.dtypes[col])
        return self.model.predict(X_cast)


@st.cache_data
def get_test_predictions():
    df_model = df_data.drop(columns=['customer_id'])
    categorical_cols = ['account_type', 'credit_score_band']
    boolean_cols = [
        'has_direct_debits', 'uses_digital_bank_secondary', 'was_kbc_ulster_customer',
        'experienced_switching_difficulty', 'has_complaint_history', 'has_mortgage', 'has_savings_goal'
    ]
    for col in categorical_cols:
        df_model[col] = encoders[col].transform(df_model[col])
    for col in boolean_cols:
        df_model[col] = df_model[col].astype(int)

    X = df_model.drop(columns=['churn'])
    y = df_model['churn']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    return X_test, y_test, y_prob, X.dtypes


X_test, y_test, y_prob, train_dtypes = get_test_predictions()


@st.cache_resource
def init_dice_explainer():
    template_df = X_test.copy()
    template_df['churn'] = y_test
    d = dice_ml.Data(
        dataframe=template_df,
        continuous_features=continuous_features,
        outcome_name='churn'
    )
    wrapped_model = XGBoostClassifierWrapper(xgb_model, feature_names, train_dtypes)
    m = dice_ml.Model(model=wrapped_model, backend="sklearn")
    return dice_ml.Dice(d, m, method="random")


dice_explainer = init_dice_explainer()

CHURN_COLOR = '#EF553B'
RETAIN_COLOR = '#636EFA'

st.title("Irish Banking Customer Churn Predictor")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏦 Overview",
    "📊 Data Explorer",
    "📈 Model Performance",
    "🔍 SHAP Explainability",
    "⚡ Risk Predictor"
])


with tab1:
    st.header("🏦 The Irish Banking Context")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Accounts Migrated", value="1.2M+")
        st.caption("Forced out of KBC Bank Ireland and Ulster Bank between 2022 and 2023.")
    with col2:
        st.metric(label="Switchers Faced Difficulty", value="60%")
        st.caption("Source: CCPC 2022 survey — direct debit failures, delays, and poor support.")
    with col3:
        st.metric(label="Top Reason for Bank Choice", value="Branch Location")
        st.caption("Digital-only alternatives failed to gain trust during the migration crisis.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "The 2022-2023 exits of KBC Bank Ireland and Ulster Bank (NatWest Group) remain the defining disruption "
        "in modern Irish retail banking. Over 1.2 million customers were forced to migrate to AIB, Bank of Ireland, "
        "or Permanent TSB within a compressed two-year window. Now in 2025-2026, those customers are entering their "
        "third or fourth year with a new provider. Behavioural research suggests institutional trust takes 3-5 years "
        "to rebuild after a forced migration — the Irish market is still in an elevated churn risk window that will "
        "not normalise before 2027. This project models those persisting switching behaviours to help retail banks "
        "identify at-risk customers before they leave."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**What this project builds**")
    st.markdown("<br>", unsafe_allow_html=True)

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown("**XGBoost + SMOTEENN**")
        st.write(
            "An XGBoost gradient boosted classifier predicts churn probability for each customer. "
            "Class imbalance (21% positive rate) is handled with SMOTEENN on the training set only. "
            "Evaluated on a clean 20% stratified holdout."
        )
    with lc2:
        st.markdown("**SHAP Explainability**")
        st.write(
            "Shapley values explain what drives each prediction — globally across the portfolio with "
            "beeswarm and bar plots, and locally for individual customers with a waterfall chart."
        )
    with lc3:
        st.markdown("**DICE Counterfactuals**")
        st.write(
            "Diverse counterfactual explanations show what would need to change for a high-risk customer "
            "to drop below the churn threshold — giving relationship managers specific, actionable targets."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander("EU AI Act — Article 86"):
            st.markdown(
                "Under Article 86 of the EU AI Act, customers have a right to explanation when automated systems "
                "make significant decisions about their access to financial services. Flagging a customer as high-risk "
                "for churn could affect product offers, credit availability, or how they are treated by relationship "
                "managers. Both SHAP and DICE provide the auditable, customer-level justifications that satisfy this requirement."
            )
    with exp_col2:
        with st.expander("EBA Guidelines on Internal Governance"):
            st.markdown(
                "Under European Banking Authority guidelines on internal governance, automated AI systems in banking "
                "must maintain clear human-in-the-loop oversight. This model is intentionally a decision-support tool, "
                "not an autonomous action-taker. All counterfactual recommendations and risk flags are designed to alert "
                "and assist advisors — final customer treatments require human validation before execution."
            )


with tab2:
    st.header("📊 Churn Rate by Segment")
    st.markdown("Statistical distributions from the 10,000-record synthetic dataset.")
    st.markdown("<br>", unsafe_allow_html=True)

    overall_churn = df_data['churn'].mean() * 100
    kbc_mask = df_data['was_kbc_ulster_customer'] == True
    kbc_churn = df_data[kbc_mask]['churn'].mean() * 100
    other_churn = df_data[~kbc_mask]['churn'].mean() * 100
    churn_ratio = kbc_churn / other_churn

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Overall Churn Rate", value=f"{overall_churn:.1f}%")
    with m2:
        st.metric(label="Former KBC / Ulster Customers", value=f"{kbc_churn:.1f}%")
    with m3:
        st.metric(label="All Other Customers", value=f"{other_churn:.1f}%")

    st.info(
        f"Former KBC and Ulster Bank customers churn at {kbc_churn:.1f}% versus {other_churn:.1f}% for other customers "
        f"— {churn_ratio:.1f}x higher. This gap narrows as months since switching increases, "
        f"but remains elevated across the dataset, reflecting the ongoing post-migration loyalty deficit in 2025-2026."
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        acct_churn = df_data.groupby('account_type')['churn'].mean().reset_index()
        acct_churn['churn'] *= 100
        fig_acct = px.bar(
            acct_churn,
            x='account_type',
            y='churn',
            labels={'account_type': 'Account type', 'churn': 'Churn rate (%)'},
            title='Churn rate by account type',
            color_discrete_sequence=[CHURN_COLOR]
        )
        fig_acct.update_layout(showlegend=False, height=320, margin=dict(l=60, r=20, t=40, b=60))
        st.plotly_chart(fig_acct, use_container_width=True)
        st.caption(
            "Customers with only a current or savings account churn at a noticeably higher rate than those "
            "holding a mortgage. The mortgage acts as a strong retention anchor in the Irish market."
        )

        prod_churn = df_data.groupby('num_products')['churn'].mean().reset_index()
        prod_churn['churn'] *= 100
        fig_prod = px.bar(
            prod_churn,
            x='num_products',
            y='churn',
            labels={'num_products': 'Products held', 'churn': 'Churn rate (%)'},
            title='Churn rate by number of products held',
            color_discrete_sequence=[CHURN_COLOR]
        )
        fig_prod.update_layout(showlegend=False, height=320, margin=dict(l=60, r=20, t=40, b=60))
        st.plotly_chart(fig_prod, use_container_width=True)
        st.caption(
            "Each additional product substantially reduces churn risk. "
            "Customers with a single product have very little tying them to the bank."
        )

    with c2:
        df_hist = df_data.copy()
        df_hist['churn_label'] = df_hist['churn'].map({0: 'Retained', 1: 'Churned'})
        fig_tenure = px.histogram(
            df_hist,
            x='tenure_months',
            color='churn_label',
            barmode='overlay',
            labels={'tenure_months': 'Tenure (months)', 'count': 'Customers', 'churn_label': 'Status'},
            title='Tenure distribution by churn status',
            color_discrete_map={'Retained': RETAIN_COLOR, 'Churned': CHURN_COLOR}
        )
        fig_tenure.update_layout(height=320, margin=dict(l=60, r=20, t=40, b=60))
        st.plotly_chart(fig_tenure, use_container_width=True)
        st.caption(
            "Churn is concentrated in customers with short tenures. "
            "After roughly five years, customers become substantially more stable."
        )

        complaint_churn = df_data.groupby('has_complaint_history')['churn'].mean().reset_index()
        complaint_churn['churn'] *= 100
        complaint_churn['has_complaint_history'] = complaint_churn['has_complaint_history'].map(
            {True: 'Complaint on record', False: 'No complaints'}
        )
        fig_complaint = px.bar(
            complaint_churn,
            x='has_complaint_history',
            y='churn',
            labels={'has_complaint_history': 'Complaint status', 'churn': 'Churn rate (%)'},
            title='Churn rate by complaint history',
            color='has_complaint_history',
            color_discrete_map={'Complaint on record': CHURN_COLOR, 'No complaints': RETAIN_COLOR}
        )
        fig_complaint.update_layout(showlegend=False, height=320, margin=dict(l=60, r=20, t=40, b=60))
        st.plotly_chart(fig_complaint, use_container_width=True)
        st.caption(
            "Having even a single complaint on record is one of the strongest predictors of churn. "
            "These customers are already disengaged and are actively looking at alternatives."
        )


with tab3:
    st.header("📈 XGBoost Performance")
    st.markdown("All metrics are evaluated on the held-out 20% test set. The model never saw these records during training.")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="F1 Score", value="0.786", delta="+0.119 vs Logistic Regression")
        st.caption("Balances precision and recall. The model catches most churners without generating too many false alarms.")
    with c2:
        st.metric(label="ROC-AUC", value="0.959", delta="+0.058 vs Logistic Regression")
        st.caption("Measures how well the model separates churners from retained customers. Random guessing would score 0.50.")
    with c3:
        st.metric(label="PR-AUC", value="0.842", delta="+0.102 vs Logistic Regression")
        st.caption("PR-AUC matters more than accuracy here because only 21% of customers churn — accuracy alone would be misleading.")

    st.markdown("---")

    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        cm = np.array([[1427, 153], [49, 371]])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=['Predicted: retained', 'Predicted: churned'],
            y=['Actual: retained', 'Actual: churned'],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            color_continuous_scale='Blues',
            title='Confusion matrix'
        )
        fig_cm.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(
            "371 of 420 actual churners correctly identified. "
            "The 49 false negatives (bottom-left) are missed churners — the main cost in a retention context."
        )

    with ch2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = px.line(
            x=fpr,
            y=tpr,
            labels={'x': 'False positive rate', 'y': 'True positive rate'},
            title='ROC curve  (AUC = 0.959)'
        )
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(height=360, margin=dict(l=60, r=20, t=50, b=60))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption(
            "The curve hugging the top-left corner shows strong separation between classes. "
            "The dashed diagonal is random chance (AUC = 0.50)."
        )

    with ch3:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig_pr = px.line(
            x=recall,
            y=precision,
            labels={'x': 'Recall', 'y': 'Precision'},
            title='Precision-recall curve  (AUC = 0.842)'
        )
        fig_pr.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0.21, y1=0.21)
        fig_pr.update_layout(height=360, margin=dict(l=60, r=20, t=50, b=60))
        st.plotly_chart(fig_pr, use_container_width=True)
        st.caption(
            "The dashed baseline is the class prevalence (21%). "
            "The model stays well above this line across the full recall range."
        )


with tab4:
    st.header("🔍 Global Feature Importance")
    st.markdown("These plots show which features drive churn predictions across the entire customer portfolio.")
    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "SHAP (SHapley Additive Explanations) assigns each feature a contribution score for every individual prediction. "
        "Features are ranked by their average absolute impact across all 10,000 customers in the dataset. "
        "Positive SHAP values push toward churn; negative values push toward retention."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.markdown("**Beeswarm Plot — Feature Interactions**")
        beeswarm_path = os.path.join('assets', 'shap_summary_plot.png')
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, use_container_width=True)
            st.caption(
                "Each dot is one customer. Red = high feature value, blue = low. "
                "Dots pushed right increase churn probability — low product count (blue) is the strongest single signal."
            )
        else:
            st.warning("Beeswarm plot not found. Run models/train_model.py to generate it.")

    with shap_col2:
        st.markdown("**Bar Plot — Mean Absolute Impact**")
        bar_path = os.path.join('assets', 'shap_bar_plot.png')
        if os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True)
            st.caption(
                "Features ranked by their average impact across all 10,000 customers. "
                "A longer bar means the feature consistently shifts the predicted probability by a larger amount."
            )
        else:
            st.warning("Bar plot not found. Run models/train_model.py to generate it.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Under Article 86 of the EU AI Act, banks must be able to explain automated decisions that affect a "
        "customer's access to financial services. SHAP provides a mathematically grounded way to do that — "
        "each explanation is derived from cooperative game theory, making it auditable and defensible to regulators. "
        "Global monitoring also helps confirm that the model is not relying on proxy variables that could introduce bias."
    )


with tab5:
    st.header("⚡ Customer Risk Assessment")
    st.markdown("Enter a customer profile to generate a churn probability, a local SHAP explanation, and retention suggestions.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1.5])

    with col_input:

        st.markdown("**Customer Profile**")
        r1a, r1b = st.columns(2)
        with r1a:
            age = st.slider("Age", 18, 75, 42)
        with r1b:
            tenure_months = st.slider("Tenure (months)", 1, 180, 24)

        st.markdown("**Account Details**")
        r2a, r2b = st.columns(2)
        with r2a:
            account_type = st.selectbox(
                "Account type",
                options=['Current Account', 'Savings Account', 'Current + Savings', 'Current + Mortgage']
            )
        with r2b:
            monthly_balance_eur = st.slider("Monthly balance (EUR)", 0, 50000, 2500)

        r3a, r3b = st.columns(2)
        with r3a:
            num_products = st.slider("Products held", 1, 5, 2)
        with r3b:
            credit_score_band = st.selectbox("Credit score band", options=['Low', 'Medium', 'High'])

        r4a, r4b = st.columns(2)
        with r4a:
            monthly_transaction_count = st.slider("Monthly transactions", 5, 200, 45)
        with r4b:
            monthly_transaction_amount_eur = st.slider("Monthly spend (EUR)", 100, 8000, 1200)

        r5a, r5b = st.columns(2)
        with r5a:
            has_direct_debits = st.checkbox("Has direct debits", value=True)
        with r5b:
            direct_debit_count = st.slider(
                "Direct debit count",
                0, 15, 4 if has_direct_debits else 0,
                disabled=not has_direct_debits
            )

        st.markdown("**Switching History**")
        was_kbc_ulster_customer = st.checkbox("Former KBC Bank Ireland or Ulster Bank customer", value=False)
        if was_kbc_ulster_customer:
            sw1, sw2 = st.columns(2)
            with sw1:
                months_since_switching = st.slider("Months since switching", 1, 36, 12)
            with sw2:
                experienced_switching_difficulty = st.checkbox("Experienced switching difficulties")
        else:
            months_since_switching = 0
            experienced_switching_difficulty = False

        st.markdown("**Engagement Signals**")
        e1, e2 = st.columns(2)
        with e1:
            uses_digital_bank_secondary = st.checkbox("Uses Revolut / N26 as secondary bank")
            has_mortgage = st.checkbox("Has mortgage with this bank")
            if has_mortgage and account_type != 'Current + Mortgage':
                account_type = 'Current + Mortgage'
        with e2:
            has_complaint_history = st.checkbox("Has complaint on record")
            has_savings_goal = st.checkbox("Has active savings goal")

        e3, e4 = st.columns(2)
        with e3:
            branch_visits_monthly = st.slider("Branch visits per month", 0, 8, 1)
        with e4:
            customer_service_calls_6months = st.slider("Service calls (last 6 months)", 0, 12, 1)

        predict_btn = st.button("Predict churn risk", type="primary", use_container_width=True)

    with col_output:

        if predict_btn:
            acct_encoded = encoders['account_type'].transform([account_type])[0]
            credit_encoded = encoders['credit_score_band'].transform([credit_score_band])[0]

            input_dict = {
                'age': age,
                'tenure_months': tenure_months,
                'account_type': acct_encoded,
                'monthly_balance_eur': float(monthly_balance_eur),
                'num_products': num_products,
                'monthly_transaction_count': monthly_transaction_count,
                'monthly_transaction_amount_eur': float(monthly_transaction_amount_eur),
                'has_direct_debits': int(has_direct_debits),
                'direct_debit_count': int(direct_debit_count) if has_direct_debits else 0,
                'uses_digital_bank_secondary': int(uses_digital_bank_secondary),
                'was_kbc_ulster_customer': int(was_kbc_ulster_customer),
                'months_since_switching': months_since_switching,
                'experienced_switching_difficulty': int(experienced_switching_difficulty),
                'branch_visits_monthly': branch_visits_monthly,
                'customer_service_calls_6months': customer_service_calls_6months,
                'has_complaint_history': int(has_complaint_history),
                'credit_score_band': credit_encoded,
                'has_mortgage': int(has_mortgage),
                'has_savings_goal': int(has_savings_goal)
            }

            input_df = pd.DataFrame([input_dict])
            churn_prob = xgb_model.predict_proba(input_df)[0, 1]

            if churn_prob < 0.30:
                risk_label = "Low Risk"
                risk_msg = "This customer shows low switching risk. No immediate action needed."
                risk_fn = st.success
            elif churn_prob < 0.60:
                risk_label = "Medium Risk"
                risk_msg = "This customer shows moderate switching risk. Consider a proactive check-in."
                risk_fn = st.warning
            else:
                risk_label = "High Risk"
                risk_msg = "This customer is at high risk of leaving. Retention action is recommended."
                risk_fn = st.error

            st.metric(label="Churn Probability", value=f"{churn_prob * 100:.1f}%", delta=risk_label)
            st.progress(float(churn_prob))
            risk_fn(risk_msg)

            st.divider()

            st.markdown("**Local explanation (SHAP waterfall)**")
            explainer = shap.TreeExplainer(xgb_model)
            shap_val = explainer(input_df)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            shap.plots.waterfall(shap_val[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Red bars pushed this customer toward churn. Blue bars pulled toward retention. "
                "The longer the bar, the stronger the effect."
            )

            if churn_prob > 0.50:
                st.divider()

                st.markdown("**What could change this customer's outcome?**")
                st.caption("These are model-generated suggestions only. A relationship manager should review before any customer contact.")

                with st.spinner("Generating counterfactual scenarios..."):
                    try:
                        # lock features that cannot realistically change for an existing customer
                        cf = dice_explainer.generate_counterfactuals(
                            input_df,
                            total_CFs=3,
                            desired_class=0,
                            features_to_vary=[
                                c for c in feature_names
                                if c not in ['age', 'was_kbc_ulster_customer', 'experienced_switching_difficulty']
                            ]
                        )

                        if cf is not None and len(cf.cf_examples_list) > 0:
                            cf_df = cf.cf_examples_list[0].final_cfs_df
                            orig_row = cf_df.iloc[0]
                            cf_rows = cf_df.iloc[1:]

                            changes = []
                            for col in input_df.columns:
                                orig_val = orig_row[col]
                                cf_vals = cf_rows[col].unique()

                                if not np.all(cf_vals == orig_val):
                                    cf_disp_list = []
                                    for val in cf_vals:
                                        if val == orig_val:
                                            continue
                                        if col in encoders:
                                            cf_disp_list.append(
                                                str(encoders[col].inverse_transform([int(val)])[0])
                                            )
                                        elif col in ['has_direct_debits', 'uses_digital_bank_secondary',
                                                     'has_complaint_history', 'has_mortgage', 'has_savings_goal']:
                                            cf_disp_list.append("Yes" if val == 1 else "No")
                                        elif col in ['monthly_balance_eur', 'monthly_transaction_amount_eur']:
                                            cf_disp_list.append(f"€{val:,.2f}")
                                        else:
                                            cf_disp_list.append(str(int(val)))

                                    if col in encoders:
                                        orig_disp = encoders[col].inverse_transform([int(orig_val)])[0]
                                    elif col in ['has_direct_debits', 'uses_digital_bank_secondary',
                                                 'has_complaint_history', 'has_mortgage', 'has_savings_goal']:
                                        orig_disp = "Yes" if orig_val == 1 else "No"
                                    elif col in ['monthly_balance_eur', 'monthly_transaction_amount_eur']:
                                        orig_disp = f"€{orig_val:,.2f}"
                                    else:
                                        orig_disp = str(int(orig_val))

                                    changes.append({
                                        'Feature': col.replace('_', ' ').title(),
                                        'Current Value': orig_disp,
                                        'Suggested Change': " or ".join(cf_disp_list)
                                    })

                            if changes:
                                st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
                            else:
                                st.write("No distinct changes identified in the counterfactuals.")
                        else:
                            st.write("Could not generate counterfactual suggestions for this profile.")
                    except Exception as e:
                        st.error(f"Error generating suggestions: {e}")

            else:
                st.divider()
                st.success("This customer is below the churn threshold. No counterfactual suggestions needed.")

        else:
            st.info("Fill in the customer profile on the left and click 'Predict churn risk'.")
