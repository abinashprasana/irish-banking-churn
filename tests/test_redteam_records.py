from collections import Counter

from redteam.schema import ATTACKS_PATH, BENIGN_PATH, load_records


def test_attack_suite_matches_the_planned_families_and_shape():
    attacks = load_records(ATTACKS_PATH)
    assert [record.id for record in attacks] == [f"a{n:03d}" for n in range(1, 51)]
    assert Counter(record.family for record in attacks) == {
        "instruction_injection": 8,
        "persuasion": 8,
        "tool_argument_tampering": 14,
        "relabelling": 6,
        "coverage_gap": 10,
        "reasoning_manipulation": 4,
    }
    for record in attacks:
        assert record.status in {"covered_by_rule", "known_gap", "closed_bypass"}
        if "offline" in record.modes:
            assert ("script" in record.payload) != ("direct" in record.payload), record.id
        if "live" in record.modes:
            assert record.payload["overrides"] or record.surface == "tool_result", record.id
        if record.status == "known_gap":
            assert record.expected_defence == "none", record.id
        assert record.violates or record.observe, record.id
    live_only = [record.id for record in attacks if record.modes == ["live"]]
    assert live_only == ["a047", "a048", "a049", "a050"]


def test_benign_controls_cover_near_misses_and_run_in_both_modes():
    benign = load_records(BENIGN_PATH)
    assert [record.id for record in benign] == [f"b{n:03d}" for n in range(1, 21)]
    assert all(record.status == "benign" and not record.violates for record in benign)
    assert all(set(record.modes) == {"offline", "live"} for record in benign)
    scenarios = {record.scenario for record in benign}
    assert {"near_threshold_above", "near_threshold_below", "03_blocked_arrears_credit"} <= scenarios
