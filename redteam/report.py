"""Aggregate red team runs into proportions with Wilson 95 percent intervals.

Usage:
    python redteam/report.py redteam/results/offline_gate_on.json redteam/results/offline_gate_off.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


Z_95 = 1.959963984540054


def wilson(successes: int, total: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval (Wilson 1927; Brown, Cai and DasGupta 2001)."""

    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def proportion(successes: int, total: int) -> dict[str, Any]:
    interval = wilson(successes, total)
    return {
        "successes": successes,
        "total": total,
        "rate": round(successes / total, 4) if total else None,
        "wilson_95": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
    }


def fmt(entry: dict[str, Any] | None) -> str:
    if not entry or not entry["total"]:
        return "n/a"
    low, high = entry["wilson_95"]
    return f"{entry['successes']} of {entry['total']} ({low:.2f} to {high:.2f})"


def summarise(outcomes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Headline, operator surface, known gap and benign figures for one run.

    A run of an attack record counts as a success when the oracle finds at least
    one harm the record targets. Live runs repeat each record, and every repeat
    counts as one trial.
    """

    outcomes = list(outcomes)
    attacks = [o for o in outcomes if o["status"] != "benign" and not o.get("skipped")]
    benign = [o for o in outcomes if o["status"] == "benign" and not o.get("skipped")]

    def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        families: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            families.setdefault(row["family"], []).append(row)
        return {
            "overall": proportion(sum(r["success"] for r in rows), len(rows)),
            "by_family": {
                name: proportion(sum(r["success"] for r in items), len(items))
                for name, items in sorted(families.items())
            },
        }

    headline = [o for o in attacks if o["headline"]]
    operator = [o for o in attacks if o["surface"] == "scenario_metadata" and o["status"] != "known_gap"]
    gaps = [o for o in attacks if o["status"] == "known_gap"]
    return {
        "headline": group(headline),
        "operator_surface": group(operator),
        "known_gap": {
            "overall": proportion(sum(r["success"] for r in gaps), len(gaps)),
            "records": [
                {"id": r["id"], "family": r["family"], "violates": r["violates"], "success": r["success"]}
                for r in gaps
            ],
        },
        "benign": {
            "false_block": proportion(sum(r["blocked"] for r in benign), len(benign)),
            "oracle_flagged": proportion(sum(bool(r["findings"]) for r in benign), len(benign)),
        },
        "covered_successes": sorted(
            {o["id"] for o in attacks if o["success"] and o["status"] in {"covered_by_rule", "closed_bypass"}}
        ),
    }


def render_markdown(runs: list[dict[str, Any]]) -> str:
    labels = [f"{run['mode']}, gate {run['gate']}" for run in runs]
    lines = []
    if any(run["includes_drafts"] for run in runs):
        lines += ["**These figures include unreviewed draft records.**", ""]
    lines += ["| Measure | " + " | ".join(labels) + " |", "|---|" + "---|" * len(runs)]

    def row(name: str, getter: Any) -> None:
        lines.append(f"| {name} | " + " | ".join(fmt(getter(run["summary"])) for run in runs) + " |")

    row("Headline attack success", lambda s: s["headline"]["overall"])
    families = sorted({f for run in runs for f in run["summary"]["headline"]["by_family"]})
    for family in families:
        row(f"  {family}", lambda s, f=family: s["headline"]["by_family"].get(f))
    row("Operator metadata attack success", lambda s: s["operator_surface"]["overall"])
    row("Known gap attack success", lambda s: s["known_gap"]["overall"])
    row("Benign false block", lambda s: s["benign"]["false_block"])
    row("Benign flagged by oracle", lambda s: s["benign"]["oracle_flagged"])
    return "\n".join(lines) + "\n"


def main(paths: list[str]) -> int:
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    print(render_markdown(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
