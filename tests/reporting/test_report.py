from __future__ import annotations

import json

import pytest

from clastogen.models import BaselineRecord, MutantExecution, MutantStatus, OracleRecord
from clastogen.reporting.render import render_html, render_json, render_markdown
from clastogen.reporting.scoring import summarize


def _execution(test_id: str, mutant_id: str, status: MutantStatus) -> MutantExecution:
    return MutantExecution(
        test_id, "app:PROMPT", mutant_id, f"mutant {mutant_id}", "op", "rule", "RULE", status, 2, 1.0
    )


def test_report_lists_survivors_first_with_every_test_they_passed() -> None:
    executions = [
        _execution("t.py::test_a", "killed1", MutantStatus.KILLED),
        _execution("t.py::test_a", "survivor", MutantStatus.SURVIVED),
        _execution("t.py::test_b", "survivor", MutantStatus.SURVIVED),
    ]
    html = render_html(summarize(executions), executions, [])

    table = html[html.index("<h2>Mutants</h2>") :]
    assert table.index("mutant survivor") < table.index("mutant killed1")
    survivor_row = table[table.index("mutant survivor") : table.index("</tr>", table.index("mutant survivor"))]
    assert ">test_a</code><br><code" in survivor_row
    assert ">test_b</code>" in survivor_row


def test_markdown_lists_survivors_first_and_renders_prompt_text_literally() -> None:
    executions = [
        _execution("t.py::test_a", "killed1", MutantStatus.KILLED),
        _execution("t.py::test_a", "survivor", MutantStatus.SURVIVED),
        _execution("t.py::test_b", "survivor", MutantStatus.SURVIVED),
        MutantExecution(
            "t.py::test_a",
            "app:PROMPT",
            "pipe",
            "a | b <!--\n<details> @team `x`",
            "op",
            "rule",
            "RULE",
            MutantStatus.KILLED,
            2,
            1.0,
        ),
    ]
    flaky = BaselineRecord("t.py::test_flaky", "app:PROMPT", 5, 10, 0.5, stable=False)
    md = render_markdown(summarize(executions), executions, [flaky])

    assert md.startswith("## Clastogen mutation testing\n")
    assert "Mutation Score: 66.7% (2 of 3 killed)" in md
    assert md.index("mutant survivor") < md.index("mutant killed1")
    assert "| ✗ SURVIVED | `survivor` | ` mutant survivor ` | ` test_a, test_b ` |" in md
    assert "| `` a \\| b <!-- <details> @team `x` `` |" in md
    assert "Flaky tests (not mutated): ` test_flaky `" in md


def test_reports_show_operator_rates_and_untested_constraints() -> None:
    executions = [_execution("t.py::test_a", "m1", MutantStatus.KILLED)]
    summary = summarize(executions, [("app:PROMPT", "rule"), ("app:PROMPT", "never <b>shared</b>")])

    for out in (render_html(summary, executions, []), render_markdown(summary, executions, [])):
        assert "By operator: op 1/1 (100%)" in out
        assert "Constraint Coverage: 1 of 2 constraints killed (50.0%)" in out
    assert "<li><code>never &lt;b&gt;shared&lt;/b&gt;</code> untested: no measured mutant</li>" in render_html(
        summary, executions, []
    )
    assert "| ` never <b>shared</b> ` | ` untested: no measured mutant ` |" in render_markdown(summary, executions, [])


def test_reports_label_each_mutant_with_what_the_probe_found() -> None:
    executions = [
        _execution("t.py::test_a", "m1", MutantStatus.KILLED),
        _execution("t.py::test_a", "m2", MutantStatus.SURVIVED),
    ]
    summary = summarize(executions, oracle=[OracleRecord("m1", "changed"), OracleRecord("m2", "suspected")])

    md = render_markdown(summary, executions, [])
    html = render_html(summary, executions, [])
    assert "mutant m1 (behavior changed)" in md
    assert "mutant m2 (behavior suspected)" in md
    assert "<em>behavior changed</em>" in html
    assert "<em>behavior suspected</em>" in html


def test_killed_mutant_shows_its_killer_not_the_tests_it_survived() -> None:
    executions = [
        _execution("t.py::test_killer", "mixed", MutantStatus.KILLED),
        _execution("t.py::test_bystander", "mixed", MutantStatus.SURVIVED),
    ]
    summary = summarize(executions)
    html = render_html(summary, executions, [])
    md = render_markdown(summary, executions, [])

    table = html[html.index("<h2>Mutants</h2>") : html.index("</table>", html.index("<h2>Mutants</h2>"))]
    assert "test_killer" in table
    assert "test_bystander" not in table
    assert "` test_killer `" in md
    assert "test_bystander" not in md


_RULE_EXECUTIONS = [
    _execution("t.py::test_b", "m1", MutantStatus.KILLED),
    _execution("t.py::test_a", "m1", MutantStatus.KILLED),
    _execution("t.py::test_a", "m2", MutantStatus.SURVIVED),
]
_RULE_SUMMARY = summarize(
    _RULE_EXECUTIONS,
    [("app:PROMPT", "rule"), ("app:PROMPT", "never leak"), ("app:PROMPT", "always greet")],
    unmutated=[("app:PROMPT", "Be friendly.")],
)


def test_markdown_snapshot_of_per_rule_evidence() -> None:
    md = render_markdown(_RULE_SUMMARY, _RULE_EXECUTIONS, [])

    assert md == (
        "## Clastogen mutation testing\n\n"
        "**Mutation Score: 50.0% (1 of 2 killed); Wilson 95% CI 9.5%-90.5%**\n\n"
        "By operator: op 1/2 (50%)\n\n"
        "Constraint Coverage: 1 of 3 constraints killed (33.3%); 0 survived or inconclusive, 2 untested\n\n"
        "Rules (A mutant is killed if any test kills it; a rule's mutants are those that were measured; the test named is the lowest test ID among the killers.)\n\n"
        "| Rule | Evidence |\n"
        "| --- | --- |\n"
        "| ` always greet ` | ` untested: no measured mutant ` |\n"
        "| ` never leak ` | ` untested: no measured mutant ` |\n"
        "| ` rule ` | ` killed: 1 of 2 mutants killed by test_a ` |\n\n"
        "Unmutated sentences (no rule keyword or digit, so no rule-level mutant):\n\n"
        "- ` Be friendly. `\n\n"
        "| Status | Mutant | Change | Tests |\n"
        "| --- | --- | --- | --- |\n"
        "| ✗ SURVIVED | `m2` | ` mutant m2 ` | ` test_a ` |\n"
        "| ✓ KILLED | `m1` | ` mutant m1 ` | ` test_a ` |\n"
    )


def test_json_snapshot_of_per_rule_evidence() -> None:
    data = json.loads(render_json(_RULE_SUMMARY, _RULE_EXECUTIONS, []))

    assert data["constraint_coverage"] == {
        "total": 3,
        "killed": 1,
        "survived": 0,
        "percent": pytest.approx(100 / 3),
        "merge_rule": "A mutant is killed if any test kills it; a rule's mutants are those that were measured; the test named is the lowest test ID among the killers.",
        "rules": [
            {"target": "app:PROMPT", "rule": "always greet", "killed": 0, "mutants": 0, "killed_by": None,
             "status": "untested"},
            {"target": "app:PROMPT", "rule": "never leak", "killed": 0, "mutants": 0, "killed_by": None,
             "status": "untested"},
            {"target": "app:PROMPT", "rule": "rule", "killed": 1, "mutants": 2, "killed_by": "t.py::test_a",
             "status": "killed"},
        ],
    }  # fmt: skip
    assert data["unmutated_sentences"] == [{"target": "app:PROMPT", "sentence": "Be friendly."}]
