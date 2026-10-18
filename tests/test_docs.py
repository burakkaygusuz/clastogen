from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pytest_check import check

from clastogen.mutation.mutator import PromptMutator
from clastogen.reporting.render import KILL_HINTS

REPO = Path(__file__).resolve().parents[1]
STATISTICS = (REPO / "docs" / "statistics.md").read_text(encoding="utf-8")
GUIDE = (REPO / "docs" / "guide.md").read_text(encoding="utf-8")
SKILL = (REPO / "skills" / "clastogen" / "SKILL.md").read_text(encoding="utf-8")
# One rule of each kind: modal, negation, threshold, exception, comparator, scope, plus a section and an example.
ALL_OPERATORS_PROMPT = """You are a support agent for an online store. You answer questions about orders, shipping, returns and refunds for registered customers, and you hand off anything else to a person.

## Refunds
You must never approve refunds over $50 unless a manager agrees.
Always verify at most 3 items, and check every order.

## Tone
Be polite to every customer.

## Examples
<example>Customer: hi</example>
"""


def test_docs_stats_example_matches_runnable_file() -> None:
    marker = "<!-- example: examples/test_stats_usage.py -->\n```python\n"
    start = STATISTICS.index(marker) + len(marker)
    block = STATISTICS[start : STATISTICS.index("\n```\n", start) + 1]
    assert block == (REPO / "examples" / "test_stats_usage.py").read_text(encoding="utf-8")


def test_every_operator_has_a_kill_hint() -> None:
    operators = {m.operator_name for m in PromptMutator().generate_mutants(ALL_OPERATORS_PROMPT, max_mutants=20)}
    assert operators == set(KILL_HINTS)


def test_kill_hints_match_skill_table() -> None:
    table = SKILL[SKILL.index("| `operator_name`") :]
    rows = [line.strip().strip("|").split("|") for line in table[: table.index("\n\n")].splitlines()[2:]]
    documented = {op: cells[2].strip() for cells in rows for op in re.findall(r"`(\w+)`", cells[0])}
    assert documented == KILL_HINTS


def test_demo_reports_documented_score(tmp_path: Path, pytester: pytest.Pytester) -> None:
    json_out = tmp_path / "demo.json"
    md_out = tmp_path / "demo.md"
    result = pytester.runpytest(
        "--pyargs",
        "clastogen.demo",
        "--clastogen",
        f"--clastogen-json={json_out}",
        f"--clastogen-md={md_out}",
    )
    out = result.stdout.str()
    assert result.ret == 0, out

    data = json.loads(json_out.read_text(encoding="utf-8"))
    check.equal(data["counts"]["KILLED"], 3)
    check.equal(data["counts"]["SURVIVED"], 2)
    check.almost_equal(data["mutation_score"], 60.0)
    low, high = data["mutation_score_ci95"]
    check.almost_equal(low, 23.07, abs=0.01)
    check.almost_equal(high, 88.23, abs=0.01)
    check.is_in("Wilson 95% CI 23.1%-88.2%", out)
    # Only the two survivors get a hint, in the terminal and in the Markdown report.
    hint = "Assert: Use a value between the old limit and the new limit"
    check.equal(out.count("Assert:"), 2)
    check.is_in(hint, out)
    md = md_out.read_text(encoding="utf-8")
    check.equal(md.count(hint), 1)
    check.is_not_in("Assert:", "\n".join(line for line in md.splitlines() if "KILLED" in line))

    by_id = {r["mutant_id"]: r for r in data["results"]}
    inverted_identity = next(r for r in by_id.values() if "NEVER verify customer identity" in r["description"])
    check.equal(inverted_identity["status"], "KILLED")
    check.equal(inverted_identity["operator_name"], "invert_negation")
    check.is_in("NEVER verify", inverted_identity["mutated_snippet"])

    documented = re.search(r'mutant_id = "([0-9a-f]{12})"', GUIDE)
    assert documented is not None
    check.equal(by_id[documented[1]]["status"], "SURVIVED")


def test_skill_script_lists_open_mutants_and_rejects_a_bad_report(tmp_path: Path) -> None:
    script = REPO / "skills" / "clastogen" / "scripts" / "survivors.py"
    results = [
        {"mutant_id": mutant_id, "status": status, "operator_name": "delete_constraint", "error": None,
         "original_snippet": "rule", "mutated_snippet": "[DELETED]"}
        for mutant_id, status in (("aaa", "SURVIVED"), ("bbb", "KILLED"))
    ]  # fmt: skip
    report = tmp_path / "clastogen.json"
    report.write_text(json.dumps({
        "mutation_score": 50.0, "mutation_score_ci95": [9.5, 90.5], "counts": {"SURVIVED": 1, "KILLED": 1},
        "by_operator": {}, "constraint_coverage": {"untested": []}, "results": results,
        "behavior": {"exposure_percent": 0.0, "labels": {"aaa": "changed"}},
    }), encoding="utf-8")  # fmt: skip

    ok = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(script), str(report)], capture_output=True, text=True, check=False
    )
    assert ok.returncode == 0, ok.stderr
    open_mutants = json.loads(ok.stdout)["open"]
    assert [(m["mutant_id"], m["behavior"]) for m in open_mutants] == [("aaa", "changed")]

    not_a_report = tmp_path / "list.json"
    not_a_report.write_text("[1]", encoding="utf-8")
    for bad_path in (tmp_path / "missing.json", not_a_report):
        bad = subprocess.run(  # noqa: S603
            [sys.executable, "-I", str(script), str(bad_path)], capture_output=True, text=True, check=False
        )
        assert bad.returncode == 2, bad.stderr
        assert "pytest --clastogen" in bad.stderr
