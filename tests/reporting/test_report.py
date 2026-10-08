from __future__ import annotations

from clastogen.models import BaselineRecord, MutantExecution
from clastogen.reporting.render import render_html, render_markdown
from clastogen.reporting.scoring import summarize
from clastogen.types import MutantStatus


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
