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
    terminal.write(reporter, state, summarize(state.executions))  # ty: ignore[invalid-argument-type]
    return reporter.lines


def _execution(status: MutantStatus) -> MutantExecution:
    return MutantExecution(
        "t.py::test_a", "app:P", "m1", "Deleted constraint: 'x'", "delete_constraint", "x", "", status
    )


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
