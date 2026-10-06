from __future__ import annotations

import json

import pytest

AGENT = '''
PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances."""
'''


def test_baseline_performs_ten_runs_and_reports_laplace_p0(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_laplace="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1)
def test_counted():
    assert len(agent.PROMPT) > 0
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}", "-v")
    assert result.ret == pytest.ExitCode.OK

    data = json.loads(json_out.read_text(encoding="utf-8"))
    (baseline,) = data["baselines"]
    assert baseline["runs"] == 10
    assert baseline["successes"] == 10
    assert baseline["p0"] == pytest.approx(11 / 12)
    assert baseline["stable"] is True
    assert "p0=0.917" in result.stdout.str()


_BREAKS_ON_RERUN = """
import pytest, agent

calls = {{"n": 0}}

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1)
def test_breaks():
    calls["n"] += 1
    if calls["n"] > 1:
        {action}
"""


@pytest.mark.parametrize(
    ("action", "error"),
    [
        ('raise RuntimeError("boom")', "unexpected execution error: RuntimeError: boom"),
        ('pytest.skip("flaky dependency")', "test skipped during baseline"),
    ],
)
def test_broken_baseline_is_rejected_with_reason(pytester: pytest.Pytester, action: str, error: str) -> None:
    pytester.makepyfile(agent=AGENT, test_breaks=_BREAKS_ON_RERUN.format(action=action))
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    result.assert_outcomes(passed=1)
    data = json.loads(json_out.read_text(encoding="utf-8"))
    (baseline,) = data["baselines"]
    assert (baseline["stable"], baseline["runs"], baseline["error"]) == (False, 1, error)
    assert data["results"] == []
    assert error in result.stdout.str()


def test_pytest_exit_during_baseline_stops_the_session(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_breaks=_BREAKS_ON_RERUN.format(action='pytest.exit("abort requested")'))
    result = pytester.runpytest("--clastogen")

    assert result.ret == pytest.ExitCode.INTERRUPTED
    assert "abort requested" in result.stdout.str()


def test_no_baseline_when_every_mutant_is_already_killed(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_order="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert "never approve refunds" in agent.PROMPT
    assert "verify identity" in agent.PROMPT

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_counted():
    with open("counted_calls.txt", "a") as fh:
        fh.write("x")
""",
    )
    json_out = pytester.path / "out.json"
    pytester.runpytest("--clastogen", f"--clastogen-json={json_out}").assert_outcomes(passed=2)

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert [b["test_id"].split("::")[-1] for b in data["baselines"]] == ["test_strong"]
    assert (pytester.path / "counted_calls.txt").read_text(encoding="utf-8") == "x"
