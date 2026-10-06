from __future__ import annotations

from clastogen.models import MutantExecution
from clastogen.report import render_html
from clastogen.scoring import summarize
from clastogen.types import MutantStatus


def _execution(test_id: str, mutant_id: str, status: MutantStatus) -> MutantExecution:
    return MutantExecution(test_id, "app:PROMPT", mutant_id, f"mutant {mutant_id}", status, 2, 1.0)


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
