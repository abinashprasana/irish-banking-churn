import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
import shap
import dice_ml

RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.20
SAMPLING_STRATEGY = 'auto'

XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_EVAL_METRIC = 'logloss'

RF_N_ESTIMATORS = 100
LR_MAX_ITER = 1000

DATA_PATH = os.path.join('data', 'irish_banking_churn.csv')
MODEL_SAVE_PATH = os.path.join('models', 'xgboost_churn_model.pkl')
ASSETS_DIR = 'assets'
MODELS_DIR = 'models'


class XGBoostClassifierWrapper:
    """
    Wraps the XGBoost model to cast DataFrame column types back to their training dtypes
    before each prediction — DiCE mutates types during counterfactual search which breaks
    the native XGBoost predictor without this guard.
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


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'PR-AUC': average_precision_score(y_test, y_prob),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"dataset not found at {DATA_PATH}. run generate_data.py first.")

    print("loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"shape: {df.shape}")
    print(f"churn distribution:\n{df['churn'].value_counts().to_string()}")

    print("preprocessing...")
    df_model = df.drop(columns=['customer_id'])

    categorical_cols = ['account_type', 'credit_score_band']
    boolean_cols = [
        'has_direct_debits', 'uses_digital_bank_secondary', 'was_kbc_ulster_customer',
        'experienced_switching_difficulty', 'has_complaint_history', 'has_mortgage', 'has_savings_goal'
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le
        print(f"encoded '{col}': {le.classes_}")

    for col in boolean_cols:
        df_model[col] = df_model[col].astype(int)

    X = df_model.drop(columns=['churn'])
    y = df_model['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    # apply smoteenn to training set only — never to test data
    print("resampling with smoteenn...")
    print(f"before: {np.bincount(y_train)}")
    smoteenn = SMOTEENN(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE)
    X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)
    print(f"after:  {np.bincount(y_train_res)}")

    print("training models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            eval_metric=XGB_EVAL_METRIC,
            random_state=RANDOM_STATE
        )
    }

    results = []
    for name, clf in models.items():
        clf.fit(X_train_res, y_train_res)
        metrics = evaluate_model(clf, X_test, y_test, name)
        results.append(metrics)
        print(f"\n{name}")
        print(f"  accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  precision: {metrics['Precision']:.4f}")
        print(f"  recall:    {metrics['Recall']:.4f}")
        print(f"  f1:        {metrics['F1 Score']:.4f}")
        print(f"  roc-auc:   {metrics['ROC-AUC']:.4f}")
        print(f"  pr-auc:    {metrics['PR-AUC']:.4f}")
        print(f"  confusion matrix:\n{metrics['Confusion Matrix']}")

    df_comparison = pd.DataFrame(results).drop(columns=['Confusion Matrix'])
    print("\nmodel comparison:")
    print(df_comparison.to_string(index=False))

    xgb_model = models['XGBoost']

    print("\ncomputing shap values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    beeswarm_path = os.path.join(ASSETS_DIR, 'shap_summary_plot.png')
    plt.title('SHAP Beeswarm Plot - Global Feature Impacts', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"saved beeswarm plot to {beeswarm_path}")

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    bar_path = os.path.join(ASSETS_DIR, 'shap_bar_plot.png')
    plt.title('SHAP Feature Importance (Mean Absolute Impact)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"saved bar plot to {bar_path}")

    mean_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Absolute_SHAP': mean_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)
    print("\ntop 5 features by mean absolute shap:")
    print(feature_importance.head(5).to_string(index=False))

    print("\nsaving model...")
    continuous_features = [c for c in X.columns if c not in (categorical_cols + boolean_cols)]
    model_payload = {
        'model': xgb_model,
        'encoders': encoders,
        'feature_names': list(X.columns),
        'categorical_features': categorical_cols + boolean_cols,
        'continuous_features': continuous_features
    }
    joblib.dump(model_payload, MODEL_SAVE_PATH)
    print(f"model saved to {MODEL_SAVE_PATH}")

    # dice verification — confirms the wrapper and explainer initialise correctly before deployment
    print("\nverifying dice setup...")
    train_df = X_train.copy()
    train_df['churn'] = y_train

    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        outcome_name='churn'
    )
    wrapped_model = XGBoostClassifierWrapper(xgb_model, list(X_train.columns), X_train.dtypes)
    m = dice_ml.Model(model=wrapped_model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    test_probs = xgb_model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.where((test_probs > 0.8) & (y_test == 1))[0]

    if len(high_risk_idx) > 0:
        query_idx = high_risk_idx[0]
        query_instance = X_test.iloc[query_idx:query_idx+1]
        print(f"test customer index {query_idx}, risk score {test_probs[query_idx]:.4f}")

        # lock features that cannot realistically change for an existing customer
        cf = exp.generate_counterfactuals(
            query_instance,
            total_CFs=3,
            desired_class=0,
            features_to_vary=[c for c in X_test.columns if c not in ['age', 'was_kbc_ulster_customer', 'experienced_switching_difficulty']]
        )

        if cf is not None and len(cf.cf_examples_list) > 0:
            cf_df = cf.cf_examples_list[0].final_cfs_df
            print("counterfactuals generated:")
            print(cf_df.to_string(index=False))
        else:
            print("no counterfactuals returned.")
    else:
        print("no high-risk customer found in test set.")

    print("\ndone.")


if __name__ == "__main__":
    main()
