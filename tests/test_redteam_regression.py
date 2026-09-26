from redteam import run_redteam
from redteam.schema import ATTACKS_PATH, load_records


def test_closed_bypasses_stay_closed_offline():
    closed = [record for record in load_records(ATTACKS_PATH) if record.status == "closed_bypass"]
    assert {record.id for record in closed} >= {"a031", "a032", "a035", "a042"}
    outcomes = run_redteam.run_offline(closed, "on")
    reopened = [outcome["id"] for outcome in outcomes if outcome["success"]]
    assert reopened == []
    assert all(outcome["final_action"] is None for outcome in outcomes)
