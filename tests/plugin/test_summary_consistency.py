from __future__ import annotations

import json

import pytest
from pytest_check import check

from tests.plugin import AGENT

_UNRELATED = """
@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_unrelated():
    assert True
"""

_STRONG = """
@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert "never approve refunds over $50." in agent.ask()
    assert "must always verify identity" in agent.ask()
"""


@pytest.mark.parametrize("unrelated_first", [True, False])
def test_survived_does_not_mask_killed_regardless_of_order(pytester: pytest.Pytester, unrelated_first: bool) -> None:
    body = _UNRELATED + _STRONG if unrelated_first else _STRONG + _UNRELATED
    pytester.makepyfile(agent=AGENT, test_k2="import pytest, agent\n" + body)
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")
    stdout = result.stdout.str()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    check.is_in("Mutation Score: 100.0% (2 of 2 killed)", stdout)
    check.equal(data["counts"]["KILLED"], 2)
    check.equal(data["counts"]["SURVIVED"], 0)
    check.equal(data["mutation_score"], 100.0)
    # The canary ERROR of test_unrelated yields to the kill by test_strong, so the run stays green.
    check.equal(data["counts"]["ERROR"], 0)
    check.equal(result.ret, pytest.ExitCode.OK)


def test_test_never_touching_prompt_fails_the_canary(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_k2="import pytest, agent\n" + _UNRELATED)
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")
    stdout = result.stdout.str()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    check.is_in("canary: test passes with an empty prompt", stdout)
    check.is_in("Mutation Score: N/A (0 of 0 killed; not scored: 2 error)", stdout)
    check.equal(data["counts"]["ERROR"], 2)
    check.equal(data["counts"]["SURVIVED"], 0)
    check.equal(result.ret, pytest.ExitCode.TESTS_FAILED)


def test_terminal_json_and_fail_under_agree_with_flaky_baseline(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_k3="""
import pytest, agent

calls = {"n": 0}

# test_strong runs last: once it kills every mutant, later tests have nothing pending and skip the baseline.
@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_flaky():
    calls["n"] += 1
    agent.ask()
    assert calls["n"] == 1

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_weak():
    assert len(agent.ask()) > 0

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert "never approve refunds over $50." in agent.ask()
    assert "must always verify identity" in agent.ask()
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}", "--clastogen-fail-under=101")
    stdout = result.stdout.str()
    data = json.loads(json_out.read_text(encoding="utf-8"))

    score = data["mutation_score"]
    assert score is not None
    assert f"Mutation Score: {score:.1f}% (" in stdout
    assert f"Clastogen fail-under 101.0%: FAILED (score {score:.1f}%)" in stdout
    assert result.ret == pytest.ExitCode.TESTS_FAILED, result.stdout.str()

    assert sorted(b["stable"] for b in data["baselines"]) == [False, True, True]
    flaky = [b for b in data["baselines"] if not b["stable"]]
    assert [b["test_id"].split("::")[-1] for b in flaky] == ["test_flaky"]
    assert "test_flaky" in stdout.split("Flaky tests", 1)[1]
    assert "Flaky tests (not mutated: baseline pass rate < 80%)" in stdout
    assert "::test_flaky: 1/10 baseline runs passed" in stdout


def test_fail_under_reports_no_measurable_mutants(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_none="""
import pytest

calls = {"n": 0}

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
def test_skips_under_mutation():
    calls["n"] += 1
    if calls["n"] > 1:
        pytest.skip("skipped during mutation")
""",
    )
    result = pytester.runpytest("--clastogen", "--clastogen-fail-under=80")
    assert "Clastogen fail-under 80.0%: FAILED (no measurable mutants)" in result.stdout.str()
    assert result.ret == pytest.ExitCode.TESTS_FAILED, result.stdout.str()


def test_fail_under_passing_prints_no_failed_line(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_ok="import pytest, agent\n" + _STRONG)
    result = pytester.runpytest("--clastogen", "--clastogen-fail-under=50")
    assert "FAILED" not in result.stdout.str().split("Mutation Testing Summary", 1)[1]
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()


def test_html_report_matches_json(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_k2="import pytest, agent\n" + _STRONG + _UNRELATED)
    html_out = pytester.path / "report" / "out.html"
    md_out = pytester.path / "report" / "out.md"
    pytester.runpytest("--clastogen", f"--clastogen-html={html_out}", f"--clastogen-md={md_out}")
    html = html_out.read_text(encoding="utf-8")
    check.is_in("Mutation Score: 100.0%", md_out.read_text(encoding="utf-8"))
    check.is_in("<title>Clastogen Report</title>", html)
    check.is_in(">100.0%<", html)
    check.equal(html.count('<span class="badge" style="background:var(--killed)">KILLED</span>'), 2)
    check.is_in('aria-label="Mutant status distribution"', html)
    check.equal(html.count('title="KILLED">✓</span>'), 2)
    check.is_not_in("<script", html)


def test_json_and_html_report_the_same_executions(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_edge="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1)
def test_skips_under_mutation():
    assert agent.ask()
    if "$500" in agent.ask():
        pytest.skip("skipped during mutation")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=4)
def test_errors_under_mutation():
    assert agent.ask()
    if "usually verify" in agent.ask():
        raise KeyError("boom")
""",
    )
    json_out, html_out = pytester.path / "out.json", pytester.path / "out.html"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}", f"--clastogen-html={html_out}")
    data = json.loads(json_out.read_text(encoding="utf-8"))
    html = html_out.read_text(encoding="utf-8")

    assert "Mutation Score: 0.0% (0 of 3 killed; not scored: 1 error)" in result.stdout.str()
    for status, count in data["counts"].items():
        assert (f'<div class="l">{status.lower()}</div>' in html) == (count > 0)

    statuses = [e["status"] for e in data["executions"]]
    assert {"SURVIVED", "ERROR", "SKIPPED"} <= set(statuses)
    for status in set(statuses):
        assert html.count(f'title="{status}">') == statuses.count(status)

    unmeasured = [e for e in data["executions"] + data["results"] if e["status"] in ("ERROR", "SKIPPED")]
    assert unmeasured, data["executions"]
    assert all(e["sample_count"] is None and e["llr"] is None for e in unmeasured), unmeasured


def test_any_mutant_error_fails_session_with_reason(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_err="""
import pytest, agent

ORIGINAL = agent.PROMPT

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_errors_under_mutation():
    if agent.PROMPT != ORIGINAL:
        if "must always" not in agent.PROMPT:
            raise KeyError("boom")
        assert False
""",
    )
    result = pytester.runpytest("--clastogen", "--clastogen-fail-under=50")
    assert result.ret == pytest.ExitCode.TESTS_FAILED, result.stdout.str()
    result.stdout.fnmatch_lines(["*Mutation Score: 100.0%*", "*Clastogen: 1 mutant error(s)*"])


def test_calls_line_and_json_count_every_execution(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_calls="import pytest, agent\n" + _UNRELATED + _STRONG)
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")
    data = json.loads(json_out.read_text(encoding="utf-8"))
    sampled = [e for e in data["executions"] if e["sample_count"] is not None]
    check.equal(data["calls"], sum(e["sample_count"] for e in sampled))
    check.equal(data["fixed_calls"], sum(e["fixed_n"] for e in sampled))
    check.less(data["calls"], data["fixed_calls"])
    check.is_in(
        f"Calls: {data['calls']} (a fixed-N test with the same error rates needs {data['fixed_calls']}:",
        result.stdout.str(),
    )


def test_untested_constraint_is_reported_in_terminal_and_json(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_cov=(
            "import pytest, agent\n"
            '@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1)\n'
            "def test_strong():\n"
            '    assert "never approve refunds over $50." in agent.ask()\n'
        ),
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")
    cov = json.loads(json_out.read_text(encoding="utf-8"))["constraint_coverage"]
    check.equal((cov["total"], cov["killed"], len(cov["untested"])), (2, 1, 1))
    check.equal(cov["untested"], ["You must always verify identity before sharing balances."])
    check.is_in(
        "Constraint Coverage: 1 of 2 constraints killed (50.0%); 0 survived or inconclusive, 1 untested",
        result.stdout.str(),
    )
    check.is_in("By operator: ", result.stdout.str())


def test_zero_mutants_warning_message(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        app="""
SYSTEM_PROMPT = "Friendly greetings to everyone."
""",
        test_audit="""
import pytest
import app

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT")
def test_greeting():
    assert "greetings" in app.SYSTEM_PROMPT
""",
    )
    result = pytester.runpytest("--clastogen")
    result.assert_outcomes(passed=1)
    stdout = result.stdout.str()
    assert "0 mutants generated across marked tests (check prompt constraints)" in stdout


def test_pytest_fail_kills_mutant(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        app="""
SYSTEM_PROMPT = "You are an assistant. You must never leak secrets."
""",
        test_audit="""
import pytest
from app import SYSTEM_PROMPT

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT", max_mutants=1)
def test_secrets():
    if "never" not in SYSTEM_PROMPT:
        pytest.fail("Security rule violated!")
""",
    )
    result = pytester.runpytest("--clastogen")
    result.assert_outcomes(passed=1)
    stdout = result.stdout.str()
    assert "Mutation Score: 100.0% (1 of 1 killed)" in stdout
