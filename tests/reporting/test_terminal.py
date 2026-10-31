from __future__ import annotations

import pytest

from clastogen.models import ClastogenPluginState, MutantExecution, MutantStatus
from clastogen.reporting import terminal
from clastogen.reporting.scoring import summarize


class _Config:
    def __init__(self, **options: bool) -> None:
        self._options = options

    def getoption(self, name: str, default: bool | None = None) -> bool | None:
        return self._options.get(name, default)

    def get_verbosity(self) -> int:
        return 0


class _Reporter:
    def __init__(self, **options: bool) -> None:
        self.config = _Config(**options)
        self.lines: list[str] = []

    def write_line(self, line: str, **_: bool) -> None:
        self.lines.append(line)

    def write_sep(self, sep: str, title: str = "", **_: bool) -> None:
        self.lines.append(f"{sep} {title}".strip())


def _write(state: ClastogenPluginState, **options: bool) -> list[str]:
    reporter = _Reporter(**options)
    terminal.write(reporter, state, summarize(state.executions, tests=state.tests))  # ty: ignore[invalid-argument-type]
    return reporter.lines


def _execution(status: MutantStatus) -> MutantExecution:
    return MutantExecution(
        "t.py::test_a", "app:P", "m1", "Deleted constraint: 'x'", "delete_constraint", "x", "", status
    )


def test_inconclusive_hint_names_the_split_of_the_inconclusive_target() -> None:
    def execution(target: str, mutant_id: str, status: MutantStatus) -> MutantExecution:
        return MutantExecution("t.py::test_a", target, mutant_id, "d", "delete_constraint", "x", "", status)

    state = ClastogenPluginState(tests={"app:A": 12, "app:B": 1})
    state.executions = [
        execution("app:A", "a1", MutantStatus.KILLED),
        execution("app:B", "b1", MutantStatus.INCONCLUSIVE),
    ]
    assert not any("alpha is divided" in line for line in _write(state))

    state.executions.append(execution("app:A", "a2", MutantStatus.INCONCLUSIVE))
    assert any("With 12 tests on one target, alpha is divided by 12" in line for line in _write(state))


def test_no_marked_tests_is_reported() -> None:
    assert _write(ClastogenPluginState()) == ["Clastogen: No @pytest.mark.clastogen tests executed."]


@pytest.mark.parametrize(
    ("options", "mentions_probes"),
    [({}, True), ({"--clastogen-oracle": True}, False)],
)
def test_marked_tests_without_mutants_explain_why(options: dict[str, bool], mentions_probes: bool) -> None:
    (line,) = _write(ClastogenPluginState(marked_tests_count=1), **options)

    assert line.startswith("Clastogen: 0 mutants generated")
    assert ("run only with --clastogen-oracle" in line) is mentions_probes


def test_a_survivor_is_listed_with_the_test_that_passed_it() -> None:
    lines = _write(ClastogenPluginState(executions=[_execution(MutantStatus.SURVIVED)], marked_tests_count=1))

    assert any("[m1]" in line and "passed in test_a" in line for line in lines)
    assert any(line.startswith("1 survived") for line in lines)
    assert lines[-1] == "="


def test_terminal_snapshot_of_per_rule_evidence_caps_rows_and_sentences() -> None:
    def run(test_id: str, mutant_id: str, rule: str, status: MutantStatus) -> MutantExecution:
        return MutantExecution(test_id, "app:P", mutant_id, "d", "op", rule, "", status)

    executions = [
        run("t.py::test_b", "m1", "never leak", MutantStatus.KILLED),
        run("t.py::test_a", "m1", "never leak", MutantStatus.KILLED),
        run("t.py::test_a", "m2", "never leak", MutantStatus.SURVIVED),
        run("t.py::test_a", "m3", "max 5", MutantStatus.SURVIVED),
    ]
    constraints = [("app:P", "never leak"), ("app:P", "max 5"), *(("app:P", f"never {i}") for i in range(10))]
    sentences = [("app:P", f"Sentence {i:02}.") for i in range(12)]
    state = ClastogenPluginState(executions=executions, marked_tests_count=1)
    reporter = _Reporter()
    terminal.write(reporter, state, summarize(executions, constraints, unmutated=sentences))  # ty: ignore[invalid-argument-type]

    start = next(i for i, line in enumerate(reporter.lines) if line.startswith("Rules"))
    assert reporter.lines[start:-1] == [
        "Rules, weakest first (A mutant is killed if any test kills it; a rule's mutants are those that were measured; the test named is the lowest test ID among the killers.)",
        *(f"    untested: no measured mutant  never {i}" for i in range(10)),
        "    … and 2 more (-v shows all)",
        "    Untested: raise max_mutants, fix errors or add tests.",
        "Unmutated sentences (no rule keyword or digit, so no rule-level mutant):",
        *(f"    Sentence {i:02}." for i in range(10)),
        "    … and 2 more (-v shows all)",
    ]
