from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

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


def test_demo_reports_documented_score(tmp_path: Path, subprocess_env: None) -> None:
    json_out = tmp_path / "demo.json"
    md_out = tmp_path / "demo.md"
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--clastogen",
            f"--clastogen-json={json_out}",
            f"--clastogen-md={md_out}",
            "examples/test_banking_eval.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    data = json.loads(json_out.read_text(encoding="utf-8"))
    check.equal(data["counts"]["KILLED"], 3)
    check.equal(data["counts"]["SURVIVED"], 2)
    check.almost_equal(data["mutation_score"], 60.0)
    low, high = data["mutation_score_ci95"]
    check.almost_equal(low, 23.07, abs=0.01)
    check.almost_equal(high, 88.23, abs=0.01)
    check.is_in("Wilson 95% CI 23.1%-88.2%", proc.stdout)
    # Only the two survivors get a hint, in the terminal and in the Markdown report.
    hint = "Assert: Use a value between the old limit and the new limit"
    check.equal(proc.stdout.count("Assert:"), 2)
    check.is_in(hint, proc.stdout)
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
