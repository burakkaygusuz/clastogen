from __future__ import annotations

import pytest

from tests.plugin import AGENT

# Each body appends to calls.log, so a line in it means the test ran (the original call or a mutant trial).
_TESTS = """
import pytest
from pathlib import Path
import agent

def _log():
    Path("calls.log").open("a").write("x")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    _log()
    assert "never approve refunds over $50." in agent.PROMPT
    assert "always verify identity" in agent.PROMPT

# A probe with oracle_samples runs only with --clastogen-oracle; the default run skips it.
@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=5)
def test_probe():
    _log()
    assert agent.PROMPT
"""


@pytest.mark.parametrize(("oracle", "mutant_runs", "calls"), [(False, 2, 49), (True, 4, 64)])
def test_collect_only_previews_worst_case_calls_without_running_anything(
    pytester: pytest.Pytester, oracle: bool, mutant_runs: int, calls: int
) -> None:
    pytester.makepyfile(agent=AGENT, test_preview=_TESTS)
    json_out = pytester.path / "out.json"
    args = ["--clastogen", "--collect-only", f"--clastogen-json={json_out}", "--clastogen-fail-under=80"]
    if oracle:
        args.append("--clastogen-oracle")
    result = pytester.runpytest(*args)
    stdout = result.stdout.str()
    assert result.ret == 0
    assert not (pytester.path / "calls.log").exists()
    assert not json_out.exists()
    # The suite test is planned with 9 baseline trials and 2 mutants at 20 steps each: 9 + 2 * 20 = 49.
    # The probe adds 5 * (1 + 2) = 15 trials and 2 mutant runs, but only with --clastogen-oracle.
    assert f"2 marked tests, 2 rules, {mutant_runs} mutant runs (0 suppressed)" in stdout
    assert f"At most {calls} test calls" in stdout


def test_collect_only_probe_alone_previews_nothing_without_oracle(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_probe="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", oracle_samples=5)
def test_probe():
    assert agent.PROMPT
""",
    )
    result = pytester.runpytest("--clastogen", "--collect-only")
    stdout = result.stdout.str()
    assert result.ret == 0
    assert "1 marked tests, 0 rules, 0 mutant runs (0 suppressed)" in stdout
    assert "At most 0 test calls" in stdout


def test_collect_only_without_marked_tests_says_so(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_plain="def test_plain():\n    assert True\n")
    result = pytester.runpytest("--clastogen", "--collect-only")
    assert result.ret == 0
    assert "Clastogen: No @pytest.mark.clastogen tests collected." in result.stdout.str()
