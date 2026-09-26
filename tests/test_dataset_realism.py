import csv
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "irish_banking_churn.csv"


def test_generated_records_respect_age_tenure_and_switch_constraints():
    with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 10_000
    assert sum(int(row["churn"]) for row in rows) == 2_100
    for row in rows:
        age = int(row["age"])
        tenure = int(row["tenure_months"])
        months_since_switch = int(row["months_since_switching"])
        assert 1 <= tenure <= max(1, (age - 18) * 12)
        assert months_since_switch <= tenure
        if row["was_kbc_ulster_customer"] == "False":
            assert months_since_switch == 0
