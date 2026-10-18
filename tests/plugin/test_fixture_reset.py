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
    result = pytester.runpytest("--clastogen", *(f"--clastogen-{kind}={path}" for kind, path in paths.items()))
    check.is_not_in("sk-SECRET-123", result.stdout.str())

    for path in paths.values():
        text = (pytester.path / path).read_text(encoding="utf-8")
        check.is_in("unexpected execution error: RuntimeError", text)
        check.is_not_in("sk-SECRET-123", text)


@pytest.mark.parametrize(
    ("action", "status"),
    [("pytest.skip('no resource')", "SKIPPED"), ("pytest.fail('no resource')", "ERROR")],
)
def test_pytest_outcome_in_fixture_reset_is_contained_per_mutant(
    pytester: pytest.Pytester, action: str, status: str
) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_outcome=f"""
import pytest, agent

setups = []

@pytest.fixture
def resource():
    setups.append(1)
    if len(setups) > 1:
        {action}
    return 1

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, p0=0.90)
def test_uses_resource(resource):
    assert len(agent.PROMPT) > 0
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    result.assert_outcomes(passed=1)
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert {r["status"] for r in data["results"]} == {status}, result.stdout.str()


@pytest.mark.parametrize("arg", ["resource", "client"])
def test_failed_reset_does_not_leave_later_trials_on_a_torn_down_fixture(pytester: pytest.Pytester, arg: str) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_retry="""
import pytest, agent

setups = []

@pytest.fixture
def resource():
    setups.append(len(setups) + 1)
    if len(setups) == 2:
        raise ValueError("boom")
    state = {"id": setups[-1], "open": True}
    yield state
    state["open"] = False

@pytest.fixture
def client(resource):
    return resource

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, p0=0.90)
def test_uses_resource(ARG):
    with open("uses.txt", "a") as fh:
        fh.write(f"{ARG['id']}:{ARG['open']}\\n")
    assert len(agent.PROMPT) > 0
""".replace("ARG", arg),
    )
    pytester.runpytest("--clastogen")

    uses = (pytester.path / "uses.txt").read_text(encoding="utf-8").split()
    assert all(use.endswith(":True") for use in uses), f"a trial ran against a torn-down fixture: {uses}"
    assert len({use.split(":")[0] for use in uses}) == len(uses)


def test_caplog_does_not_carry_records_between_trials(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_caplog="""
import logging, pytest, agent

@pytest.fixture
def noisy():
    logging.getLogger().warning("setup")
    yield
    logging.getLogger().warning("teardown")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
def test_logs_once(noisy, caplog):
    assert agent.PROMPT
    assert "leaked" not in caplog.text
    logging.getLogger().warning("leaked")
    assert [r.getMessage() for r in caplog.records] == ["leaked"]
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    result.assert_outcomes(passed=1)
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert {r["status"] for r in data["results"]} == {"SURVIVED"}, result.stdout.str()


def test_unittest_testcase_is_an_error_not_a_survivor(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_case="""
import unittest, agent
import pytest

class TestPrompt(unittest.TestCase):
    @pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
    def test_prompt(self):
        self.assertIn("never approve", agent.PROMPT)
""",
    )
    json_out = pytester.path / "out.json"
    pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert {r["status"] for r in data["results"]} == {"ERROR"}


def test_pytest_exit_in_fixture_reset_stops_the_session(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_exit="""
import pytest, agent

setups = []

@pytest.fixture
def resource():
    setups.append(1)
    if len(setups) > 1:
        pytest.exit("abort requested")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
def test_uses_resource(resource):
    assert len(agent.PROMPT) > 0
""",
    )
    result = pytester.runpytest("--clastogen")

    assert result.ret == pytest.ExitCode.INTERRUPTED, result.stdout.str()
    assert "abort requested" in result.stdout.str()


@pytest.mark.parametrize(
    ("where", "error"),
    [("body", "unexpected execution error: SystemExit"), ("fixture", "fixture reset failed: SystemExit")],
)
def test_system_exit_in_a_trial_is_an_error_not_a_test_failure(
    pytester: pytest.Pytester, where: str, error: str
) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_exit="""
import sys, pytest, agent

calls = {"fixture": 0, "body": 0}

def maybe_exit(where):
    calls[where] += 1
    if calls[where] > 1 and where == WHERE:
        sys.exit(2)

@pytest.fixture
def resource():
    maybe_exit("fixture")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
def test_exits(resource):
    maybe_exit("body")
""".replace("WHERE", repr(where)),
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    result.assert_outcomes(passed=1)
    data = json.loads(json_out.read_text(encoding="utf-8"))
    (mutant,) = data["results"]
    assert (mutant["status"], mutant["error"]) == ("ERROR", error), result.stdout.str()
