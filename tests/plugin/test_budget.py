from __future__ import annotations

import json

import pytest

from tests.plugin import AGENT

# Each run of the test body appends one byte to calls.log: the original call plus every trial.
_TEST = """
import pytest, agent
from pathlib import Path

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, max_steps=5)
def test_strong():
    Path("calls.log").open("a").write("x")
    assert "never approve refunds over $50." in agent.PROMPT
    assert "always verify identity" in agent.PROMPT
"""


def _trials(pytester: pytest.Pytester) -> int:
    log = pytester.path / "calls.log"
    return max(log.stat().st_size - 1, 0) if log.exists() else 0


@pytest.mark.parametrize(
    ("cap", "trials", "killed", "unmeasured", "score"),
    [(5, 0, 0, 2, None), (12, 0, 0, 2, None), (14, 11, 1, 1, 100.0), (30, 13, 2, 0, 100.0)],
)
def test_serial_run_never_exceeds_the_cap(
    pytester: pytest.Pytester, cap: int, trials: int, killed: int, unmeasured: int, score: float | None
) -> None:
    pytester.makepyfile(agent=AGENT, test_budget=_TEST)
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-max-calls={cap}", f"--clastogen-json={json_out}")
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert result.ret == pytest.ExitCode.OK
    assert _trials(pytester) == trials <= cap
    assert data["counts"]["KILLED"] == killed
    assert data["counts"]["UNMEASURED"] == unmeasured
    assert data["mutation_score"] == score


_PROBE = """
import pytest, agent
from pathlib import Path

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=5)
def test_probe():
    Path("calls.log").open("a").write("x")
    assert "never approve refunds over $50." in agent.PROMPT
    assert "always verify identity" in agent.PROMPT
"""


@pytest.mark.parametrize(
    ("cap", "trials", "labelled"),
    [(5, 0, False), (9, 0, False), (10, 10, True), (30, 15, True)],
)
def test_probe_needs_its_baseline_and_one_mutant_in_budget(
    pytester: pytest.Pytester, cap: int, trials: int, labelled: bool
) -> None:
    pytester.makepyfile(agent=AGENT, test_budget=_PROBE)
    result = pytester.runpytest("--clastogen", "--clastogen-oracle", f"--clastogen-max-calls={cap}")
    assert _trials(pytester) == trials <= cap
    assert ("Probes labelled mutants" in result.stdout.str()) is labelled


def test_unmeasured_fails_the_run_only_with_fail_under(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_budget=_TEST)
    lenient = pytester.runpytest("--clastogen", "--clastogen-max-calls=14")
    strict = pytester.runpytest("--clastogen", "--clastogen-max-calls=14", "--clastogen-fail-under=50")
    assert lenient.ret == pytest.ExitCode.OK
    assert strict.ret == pytest.ExitCode.TESTS_FAILED
    assert "1 mutant(s) unmeasured within --clastogen-max-calls" in strict.stdout.str()


def test_xdist_gives_each_worker_a_share_of_the_cap(pytester: pytest.Pytester, subprocess_env: None) -> None:
    pytester.makepyfile(agent=AGENT, test_budget=_TEST)
    json_out = pytester.path / "out.json"
    args = ("--clastogen", "--clastogen-max-calls=14", f"--clastogen-json={json_out}", "-n", "2", "--dist", "loadfile")
    result = pytester.runpytest_subprocess(*args)
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    # The serial run with 14 spends 11 trials; the one busy worker gets 7, which cannot fund the 9 baseline runs.
    assert _trials(pytester) == 0
    assert data["counts"]["UNMEASURED"] == 2


def test_max_calls_below_the_worker_count_is_a_usage_error(pytester: pytest.Pytester, subprocess_env: None) -> None:
    pytester.makepyfile(agent=AGENT, test_budget=_TEST)
    result = pytester.runpytest_subprocess("--clastogen", "--clastogen-max-calls=1", "-n", "2")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "--clastogen-max-calls 1 is below the 2 xdist workers" in result.stderr.str()


def test_max_calls_below_one_is_a_usage_error(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_budget=_TEST)
    result = pytester.runpytest("--clastogen", "--clastogen-max-calls=0")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "--clastogen-max-calls must be at least 1" in result.stderr.str()
