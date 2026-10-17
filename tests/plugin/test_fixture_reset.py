from __future__ import annotations

import json

import pytest
from pytest_check import check

from tests.plugin import AGENT


def test_failing_fixture_teardown_marks_mutants_error_without_stale_trials(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_teardown="""
import pytest, agent

setups = []

@pytest.fixture
def resource():
    setups.append(len(setups) + 1)
    yield setups[-1]
    raise ValueError("boom")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, p0=0.90)
def test_uses_resource(resource):
    with open("uses.txt", "a") as fh:
        fh.write(f"{resource}\\n")
    assert len(agent.PROMPT) > 0
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["results"], result.stdout.str()
    for r in data["results"]:
        check.equal(r["status"], "ERROR")
        check.equal("fixture reset failed: ValueError", r["error"])
    uses = (pytester.path / "uses.txt").read_text(encoding="utf-8").split()
    check.equal(len(uses), len(set(uses)), f"a trial ran against a stale fixture value: {uses}")
    check.is_in("fixture reset failed: ValueError", result.stdout.str())
    check.is_not_in("INTERNALERROR", result.stdout.str())


def test_dependent_function_fixtures_are_rebuilt_and_torn_down_per_trial(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_chain="""
import pytest, agent

events = []

@pytest.fixture
def base():
    events.append("setup base")
    yield "base"
    events.append("teardown base")

@pytest.fixture
def derived(base):
    events.append("setup derived")
    yield base + "+derived"
    events.append("teardown derived")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
def test_chain(derived):
    assert derived == "base+derived"

def test_zz_events_balanced():
    assert events.count("setup base") == events.count("teardown base")
    assert events.count("setup derived") == events.count("teardown derived")
    assert events.count("setup derived") > 2
""",
    )
    result = pytester.runpytest("--clastogen")
    result.assert_outcomes(passed=2)


def test_baseline_reset_failure_rejects_baseline_with_reason(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_baseline_teardown="""
import pytest, agent

@pytest.fixture
def resource():
    yield 1
    raise ValueError("boom")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_uses_resource(resource):
    assert len(agent.PROMPT) > 0
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    data = json.loads(json_out.read_text(encoding="utf-8"))
    (baseline,) = data["baselines"]
    check.equal(baseline["stable"], False)
    check.equal(baseline["runs"], 1)
    check.equal("fixture reset failed: ValueError", baseline["error"])
    check.equal({r["status"] for r in data["results"]}, {"ERROR"})
    assert result.ret == pytest.ExitCode.TESTS_FAILED, result.stdout.str()
    assert "fixture reset failed: ValueError" in result.stdout.str()


def test_tmp_path_and_function_fixtures_reset_into_real_decisions(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_captured="""
import pytest, agent

@pytest.fixture
def bot():
    return {"system": agent.PROMPT}

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_captured(bot, tmp_path):
    assert "never approve refunds over $50." in bot["system"]
    assert "always verify identity" in bot["system"]
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert [r["status"] for r in data["results"]] == ["KILLED", "KILLED"]
    check.is_not_in("INTERNALERROR", result.stdout.str())


def test_exception_text_stays_out_of_the_reports(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent='PROMPT = "You must never approve refunds over $50."\n',
        test_leak="""
import pytest, agent

CALLS = []

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
def test_leaks():
    CALLS.append(1)
    if len(CALLS) > 1:
        raise RuntimeError("api key sk-SECRET-123")
""",
    )
    paths = {"json": "out.json", "html": "out.html", "md": "out.md"}
    result = pytester.runpytest("-rA", "--clastogen", *(f"--clastogen-{kind}={path}" for kind, path in paths.items()))
    check.is_not_in("sk-SECRET-123", result.stdout.str())

    for path in paths.values():
        text = (pytester.path / path).read_text(encoding="utf-8")
        check.is_in("unexpected execution error: RuntimeError", text)
        check.is_not_in("sk-SECRET-123", text)
