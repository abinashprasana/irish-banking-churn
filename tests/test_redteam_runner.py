from types import SimpleNamespace

import pytest

from agent import loop
from redteam import run_redteam
from redteam.live import (
    BudgetExhausted,
    PacedQuota,
    UsageLedger,
    _parse_reset,
    attach_usage_recorder,
    contains_key_material,
)
from redteam.report import proportion, summarise, wilson
from redteam.schema import ATTACKS_PATH, BENIGN_PATH, load_records
from redteam.ungated import gate_disabled


@pytest.fixture(scope="module")
def offline_runs():
    records = load_records(ATTACKS_PATH) + load_records(BENIGN_PATH)
    return {gate: run_redteam.run_offline(records, gate) for gate in ("on", "off")}


def test_gate_on_stops_every_covered_attack_and_blocks_no_benign_case(offline_runs):
    summary = summarise(offline_runs["on"])
    assert summary["covered_successes"] == []
    assert summary["headline"]["overall"]["successes"] == 0
    assert summary["benign"]["false_block"]["successes"] == 0
    assert summary["benign"]["oracle_flagged"]["successes"] == 0


def test_gate_off_lets_covered_attacks_through(offline_runs):
    summary = summarise(offline_runs["off"])
    assert summary["headline"]["overall"]["successes"] > 0
    assert summary["benign"]["false_block"]["successes"] == 0


def test_gate_off_leaves_the_production_loop_unchanged():
    execute, refusal = loop._execute_tool, loop._blocked_refusal
    with gate_disabled():
        assert loop._execute_tool is not execute
        assert loop._blocked_refusal is not refusal
    assert loop._execute_tool is execute
    assert loop._blocked_refusal is refusal


def test_live_mode_stops_without_a_key_and_makes_no_request():
    with pytest.raises(SystemExit, match="GROQ_API_KEY is not set"):
        run_redteam.run_live(load_records(BENIGN_PATH)[:1], "on", repeats=1, max_requests=1)


def test_paced_quota_waits_at_the_minute_limit_and_stops_at_the_budget(tmp_path):
    ticks = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        ticks[0] += seconds

    quota = PacedQuota(31, UsageLedger(), state_path=tmp_path / "quota.json", sleep=sleep)
    quota._monotonic = lambda: ticks[0]
    for _ in range(30):
        quota.reserve_request()
    assert slept == []
    quota.reserve_request()
    assert slept and slept[0] == pytest.approx(60.05)
    with pytest.raises(BudgetExhausted):
        quota.reserve_request()
    assert quota.used == 31
    assert '"requests": 31' in (tmp_path / "quota.json").read_text(encoding="utf-8")


def test_usage_recorder_keeps_usage_and_token_headers():
    usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=80, total_tokens=1280)
    response = SimpleNamespace(model="qwen/qwen3.6-27b", usage=usage)
    raw = SimpleNamespace(
        parse=lambda: response,
        headers={"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "7.5s"},
    )
    sdk = SimpleNamespace(with_raw_response=SimpleNamespace(create=lambda **kwargs: raw))
    guarded = SimpleNamespace(_completions=sdk)
    client = SimpleNamespace(chat=SimpleNamespace(completions=guarded))
    ledger = UsageLedger()
    attach_usage_recorder(client, ledger)
    assert client.chat.completions._completions.create(model="x") is response
    assert ledger.totals()["prompt_tokens"] == 1200
    assert ledger.totals()["models"] == ["qwen/qwen3.6-27b"]
    assert 0 < ledger.token_wait() <= 7.5
    assert _parse_reset("1m2.5s") == pytest.approx(62.5)
    assert contains_key_material('{"k": "gsk_abc"}') and not contains_key_material("{}")


def test_wilson_intervals_match_published_values():
    low, high = wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    assert wilson(0, 0) is None
    assert proportion(3, 50)["wilson_95"] == [pytest.approx(0.0206, abs=1e-4), pytest.approx(0.1622, abs=1e-4)]
