"""Run the Phase 1 training recipe on real bank customer data.

Dataset: UCI Bank Marketing (Moro, Cortez and Rita, 2014, Decision Support
Systems 62:22-31), CC BY 4.0, https://archive.ics.uci.edu/dataset/222/bank+marketing
45,211 real customers of a Portuguese bank. The target is whether the customer
subscribed to a term deposit after a phone campaign. It is a churn proxy only in
the loose sense of a retention decision, so results are reported as a method
transfer check on real bank data, never as churn performance.

``duration`` is dropped because the call length is only known after the outcome,
which the dataset authors note makes it unsuitable for a realistic model.

Download the archive into data/external/ (gitignored) and unzip it so that
data/external/bank-full.csv exists, then run:
    python scripts/benchmark_real_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "external" / "bank-full.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "benchmark_uci_bank_marketing.json"

# Same settings as models/train_model.py so the comparison is like for like.
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"Skipped: {DATA_PATH.relative_to(PROJECT_ROOT)} is missing.")
        print("Download bank+marketing.zip from the UCI page in this file's docstring.")
        return 0

    df = pd.read_csv(DATA_PATH, sep=";")
    y = (df.pop("y") == "yes").astype(int)
    df = df.drop(columns=["duration"])
    for column in df.select_dtypes(include="object").columns:
        df[column] = LabelEncoder().fit_transform(df[column])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SPLIT_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_res, y_res = SMOTEENN(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_res, y_res)
        predicted = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": round(accuracy_score(y_test, predicted), 4),
                "precision": round(precision_score(y_test, predicted), 4),
                "recall": round(recall_score(y_test, predicted), 4),
                "f1": round(f1_score(y_test, predicted), 4),
                "roc_auc": round(roc_auc_score(y_test, scores), 4),
                "average_precision": round(average_precision_score(y_test, scores), 4),
            }
        )

    result = {
        "dataset": "UCI Bank Marketing (bank-full.csv)",
        "citation": "Moro, S., Cortez, P. and Rita, P. (2014). A data-driven approach to "
        "predict the success of bank telemarketing. Decision Support Systems 62, 22-31.",
        "licence": "CC BY 4.0",
        "source_sha256": _sha256(DATA_PATH),
        "target": "term deposit subscription (y); not churn",
        "dropped_columns": ["duration"],
        "records": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "train_before_resampling": [int((y_train == 0).sum()), int((y_train == 1).sum())],
        "train_after_smoteenn": [int((y_res == 0).sum()), int((y_res == 1).sum())],
        "test_records": int(len(y_test)),
        "metrics": rows,
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nrecords={result['records']} positive_rate={result['positive_rate']}")
    print(f"smoteenn: {result['train_before_resampling']} -> {result['train_after_smoteenn']}")
    print(f"wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
