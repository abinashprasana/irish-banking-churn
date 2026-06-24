import os
import numpy as np
import pandas as pd
from faker import Faker

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
fake = Faker()
Faker.seed(RANDOM_STATE)

NUM_RECORDS = 10000

# ~21% churn rate — estimated average for Irish retail banking during the migration event
TARGET_CHURN_RATE = 0.21

# ~15% of customers are former KBC/Ulster Bank — source: Central Bank of Ireland (~1.2M accounts migrated)
KBC_ULSTER_MIGRATION_PROB = 0.15

# 60% of switchers reported difficulties — source: CCPC 2022 survey
SWITCHING_DIFFICULTY_PROB = 0.60

AGE_MIN = 18
AGE_MAX = 75
AGE_BETA_A = 3.0
AGE_BETA_B = 4.0  # skewed towards 30-55 working-age range

MAX_SAVINGS_BALANCE = 50000.0
BALANCE_EXP_SCALE = 8000.0
PRODUCT_LIMIT = 5

NOISE_STD_DEV = 0.5
BASELINE_BIAS = -1.5


def generate_demographics(num_records):
    customer_ids = np.array([f"IRLBANK_{i+1:05d}" for i in range(num_records)])
    beta_samples = np.random.beta(a=AGE_BETA_A, b=AGE_BETA_B, size=num_records)
    ages = (AGE_MIN + beta_samples * (AGE_MAX - AGE_MIN)).astype(int)
    return customer_ids, ages


def generate_account_products(num_records):
    account_types = np.random.choice(
        ['Current Account', 'Savings Account', 'Current + Savings', 'Current + Mortgage'],
        size=num_records,
        p=[0.35, 0.10, 0.40, 0.15]
    )

    balances = np.random.exponential(scale=BALANCE_EXP_SCALE, size=num_records)
    balances = np.clip(balances, 0.0, MAX_SAVINGS_BALANCE)

    # product counts are correlated with account type to keep the data internally consistent
    product_counts = []
    for acct in account_types:
        if acct == 'Current Account':
            product_counts.append(np.random.choice([1, 2], p=[0.7, 0.3]))
        elif acct == 'Savings Account':
            product_counts.append(np.random.choice([1, 2], p=[0.8, 0.2]))
        elif acct == 'Current + Savings':
            product_counts.append(np.random.choice([2, 3, 4], p=[0.6, 0.3, 0.1]))
        else:  # Current + Mortgage
            product_counts.append(np.random.choice([3, 4, 5], p=[0.4, 0.4, 0.2]))

    return account_types, np.round(balances, 2), np.array(product_counts)


def generate_transactions_direct_debits(num_records, account_types):
    transaction_counts = []
    has_direct_debits = []

    for acct in account_types:
        if acct == 'Savings Account':
            transaction_counts.append(np.random.randint(5, 20))
            has_direct_debits.append(np.random.choice([True, False], p=[0.1, 0.9]))
        else:
            transaction_counts.append(np.random.randint(15, 201))
            has_direct_debits.append(np.random.choice([True, False], p=[0.85, 0.15]))

    transaction_counts = np.array(transaction_counts)
    has_direct_debits = np.array(has_direct_debits)

    # transaction amount scales with count to reflect realistic spend patterns
    transaction_amounts = transaction_counts * np.random.uniform(20.0, 50.0) + np.random.uniform(50.0, 500.0)
    transaction_amounts = np.clip(transaction_amounts, 100.0, 8000.0)

    direct_debit_counts = []
    for hd in has_direct_debits:
        if hd:
            direct_debit_counts.append(np.random.randint(1, 16))
        else:
            direct_debit_counts.append(0)

    return transaction_counts, np.round(transaction_amounts, 2), has_direct_debits, np.array(direct_debit_counts)


def generate_switching_status(num_records):
    was_kbc_ulster = np.random.choice(
        [True, False],
        size=num_records,
        p=[KBC_ULSTER_MIGRATION_PROB, 1 - KBC_ULSTER_MIGRATION_PROB]
    )

    months_since_switch = []
    experienced_difficulty = []

    for wk in was_kbc_ulster:
        if wk:
            months_since_switch.append(np.random.randint(1, 37))
            experienced_difficulty.append(
                np.random.choice([True, False], p=[SWITCHING_DIFFICULTY_PROB, 1 - SWITCHING_DIFFICULTY_PROB])
            )
        else:
            months_since_switch.append(0)
            experienced_difficulty.append(False)

    return was_kbc_ulster, np.array(months_since_switch), np.array(experienced_difficulty)


def generate_service_complaints_credit(num_records, ages):
    # older customers visit branches more frequently
    branch_visits = []
    for age in ages:
        if age > 55:
            branch_visits.append(
                np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], p=[0.1, 0.2, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02])
            )
        else:
            branch_visits.append(np.random.choice([0, 1, 2, 3, 4], p=[0.5, 0.3, 0.1, 0.07, 0.03]))

    branch_visits = np.array(branch_visits)
    service_calls = np.random.poisson(lam=1.5, size=num_records)
    service_calls = np.clip(service_calls, 0, 12)
    has_complaint = np.random.choice([True, False], size=num_records, p=[0.05, 0.95])
    credit_bands = np.random.choice(['Low', 'Medium', 'High'], size=num_records, p=[0.15, 0.60, 0.25])
    has_savings_goal = np.random.choice([True, False], size=num_records, p=[0.35, 0.65])

    return branch_visits, service_calls, has_complaint, credit_bands, has_savings_goal


def compute_churn_labels(df):
    """
    Assigns binary churn labels by scoring each customer with a logistic function,
    then thresholding at the percentile that locks in the target churn rate.
    Gaussian noise is added to prevent deterministic clustering.
    """
    score = np.full(len(df), BASELINE_BIAS)

    # strong positive signals — high weight because these are the clearest churn indicators
    score += np.where((df['was_kbc_ulster_customer']) & (df['months_since_switching'] < 12), 1.5, 0.0)
    score += np.where(df['experienced_switching_difficulty'], 1.5, 0.0)
    score += np.where(df['has_complaint_history'], 2.0, 0.0)
    score += np.where(df['customer_service_calls_6months'] > 6, 1.8, 0.0)
    score += np.where((df['uses_digital_bank_secondary']) & (df['num_products'] == 1), 1.4, 0.0)
    score += np.where(df['tenure_months'] < 6, 1.6, 0.0)

    # medium positive signals
    score += np.where(df['num_products'] == 1, 0.8, 0.0)
    score += np.where(df['monthly_transaction_count'] < 15, 0.6, 0.0)
    # no direct debits is a strong switching signal in the Irish market — nothing anchors the customer
    score += np.where(~df['has_direct_debits'], 0.7, 0.0)
    score += np.where((df['branch_visits_monthly'] == 0) & (df['age'] > 50), 0.8, 0.0)

    # negative signals — these act as retention anchors
    score += np.where(df['has_mortgage'], -2.5, 0.0)
    score += np.where(df['tenure_months'] > 60, -1.0, 0.0)
    score += np.where(df['num_products'] >= 3, -0.8, 0.0)
    score += np.where(df['has_savings_goal'], -0.6, 0.0)

    noise = np.random.normal(loc=0.0, scale=NOISE_STD_DEV, size=len(df))
    final_score = score + noise
    probabilities = 1.0 / (1.0 + np.exp(-final_score))

    # percentile threshold locks churn rate to TARGET_CHURN_RATE regardless of score distribution
    threshold = np.percentile(probabilities, 100.0 - TARGET_CHURN_RATE * 100.0)
    churn_labels = np.where(probabilities >= threshold, 1, 0)

    return churn_labels


def main():
    print("generating synthetic dataset...")

    customer_ids, ages = generate_demographics(NUM_RECORDS)
    account_types, balances, product_counts = generate_account_products(NUM_RECORDS)
    tx_counts, tx_amounts, has_dd, dd_counts = generate_transactions_direct_debits(NUM_RECORDS, account_types)

    uses_digital_sec = []
    for a in ages:
        if a < 35:
            uses_digital_sec.append(np.random.choice([True, False], p=[0.65, 0.35]))
        elif a < 55:
            uses_digital_sec.append(np.random.choice([True, False], p=[0.45, 0.55]))
        else:
            uses_digital_sec.append(np.random.choice([True, False], p=[0.20, 0.80]))
    uses_digital_sec = np.array(uses_digital_sec)

    was_kbc_ulster, months_since_switch, experienced_difficulty = generate_switching_status(NUM_RECORDS)
    branch_visits, service_calls, has_complaint, credit_bands, has_savings_goal = generate_service_complaints_credit(NUM_RECORDS, ages)

    tenure_months = np.random.randint(1, 181, size=NUM_RECORDS)
    has_mortgage = (account_types == 'Current + Mortgage')

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'tenure_months': tenure_months,
        'account_type': account_types,
        'monthly_balance_eur': balances,
        'num_products': product_counts,
        'monthly_transaction_count': tx_counts,
        'monthly_transaction_amount_eur': tx_amounts,
        'has_direct_debits': has_dd,
        'direct_debit_count': dd_counts,
        'uses_digital_bank_secondary': uses_digital_sec,
        'was_kbc_ulster_customer': was_kbc_ulster,
        'months_since_switching': months_since_switch,
        'experienced_switching_difficulty': experienced_difficulty,
        'branch_visits_monthly': branch_visits,
        'customer_service_calls_6months': service_calls,
        'has_complaint_history': has_complaint,
        'credit_score_band': credit_bands,
        'has_mortgage': has_mortgage,
        'has_savings_goal': has_savings_goal
    })

    df['churn'] = compute_churn_labels(df)

    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'irish_banking_churn.csv')
    df.to_csv(output_path, index=False)

    total_records = len(df)
    churn_count = int(df['churn'].sum())
    churn_pct = (churn_count / total_records) * 100.0
    kbc_ulster_count = int(df['was_kbc_ulster_customer'].sum())

    print(f"records: {total_records}")
    print(f"churn rate: {churn_pct:.2f}%")
    print(f"former kbc/ulster customers: {kbc_ulster_count} ({kbc_ulster_count/total_records*100.0:.2f}%)")
    print(df['churn'].value_counts().to_string())
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
