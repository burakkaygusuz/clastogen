from __future__ import annotations

import json

import pytest

AGENT = '''
PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances."""
'''


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
    assert data["results"]
    for r in data["results"]:
        assert r["status"] == "ERROR"
        assert "fixture reset failed: ValueError: boom" in r["error"]
    uses = (pytester.path / "uses.txt").read_text(encoding="utf-8").split()
    assert len(uses) == len(set(uses)), f"a trial ran against a stale fixture value: {uses}"
    assert "fixture reset failed: ValueError: boom" in result.stdout.str()
    assert "INTERNALERROR" not in result.stdout.str()


def test_other_trial_exceptions_record_error_reason(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_crash="""
import pytest, agent

calls = {"n": 0}

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
def test_crashes_under_mutation():
    calls["n"] += 1
    if calls["n"] > 1:
        raise KeyError("missing")
""",
    )
    json_out = pytester.path / "out.json"
    pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")
    (r,) = json.loads(json_out.read_text(encoding="utf-8"))["results"]
    assert r["status"] == "ERROR"
    assert "KeyError" in r["error"]


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


def test_tmp_path_fixture_resets_into_real_decisions(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_tmp="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_uses_tmp(tmp_path):
    assert "never approve refunds over $50." in agent.PROMPT
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert [r["status"] for r in data["results"]] == ["KILLED", "KILLED"]
    (baseline,) = data["baselines"]
    assert (baseline["successes"], baseline["runs"], baseline["stable"]) == (10, 10, True)
    result.assert_outcomes(passed=1)
    assert "INTERNALERROR" not in result.stdout.str()


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
    assert baseline["stable"] is False
    assert baseline["runs"] == 1
    assert "fixture reset failed: ValueError: boom" in baseline["error"]
    assert {r["status"] for r in data["results"]} == {"ERROR"}
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "fixture reset failed: ValueError: boom" in result.stdout.str()


def test_function_fixture_is_rebuilt_under_the_mutated_prompt(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_captured="""
import pytest, agent

@pytest.fixture
def bot():
    return {"system": agent.PROMPT}

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_captured(bot):
    assert "never approve refunds over $50." in bot["system"]
    assert "always verify identity" in bot["system"]
""",
    )
    json_out = pytester.path / "out.json"
    pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert [r["status"] for r in data["results"]] == ["KILLED", "KILLED"]
