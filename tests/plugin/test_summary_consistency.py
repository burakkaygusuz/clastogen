from __future__ import annotations

import json

import pytest

AGENT = '''
PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances."""

def ask():
    return PROMPT.lower()
'''


_UNRELATED = """
@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_unrelated():
    assert True
"""

_STRONG = """
@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert "never approve refunds" in agent.ask()
    assert "verify identity" in agent.ask()
"""


@pytest.mark.parametrize("unrelated_first", [True, False])
def test_survived_does_not_mask_killed_regardless_of_order(pytester: pytest.Pytester, unrelated_first: bool) -> None:
    body = _UNRELATED + _STRONG if unrelated_first else _STRONG + _UNRELATED
    pytester.makepyfile(agent=AGENT, test_k2="import pytest, agent\n" + body)
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")
    stdout = result.stdout.str()
    assert "Killed (Caught by Suite): 2" in stdout
    assert "Survived (Blind Spots)  : 0" in stdout
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["counts"]["KILLED"] == 2
    assert data["counts"]["SURVIVED"] == 0
    assert data["mutation_score"] == 100.0


def test_test_never_touching_prompt_survives(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_k2="import pytest, agent\n" + _UNRELATED)
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")
    assert "Survived (Blind Spots)  : 2" in result.stdout.str()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["counts"]["SURVIVED"] == 2
    assert data["mutation_score"] == 0.0


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
    assert "never approve refunds" in agent.ask()
    assert "verify identity" in agent.ask()
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}", "--clastogen-fail-under=101")
    stdout = result.stdout.str()
    data = json.loads(json_out.read_text(encoding="utf-8"))

    score = data["mutation_score"]
    assert score is not None
    assert f"Mutation Score          : {score:.1f}%" in stdout
    assert "Adjusted" not in stdout
    assert f"Clastogen fail-under 101.0%: FAILED (score {score:.1f}%)" in stdout
    assert result.ret == pytest.ExitCode.TESTS_FAILED

    assert sorted(b["stable"] for b in data["baselines"]) == [False, True, True]
    flaky = [b for b in data["baselines"] if not b["stable"]]
    assert [b["test_id"].split("::")[-1] for b in flaky] == ["test_flaky"]
    assert "test_flaky" in stdout.split("Flaky baselines", 1)[1]


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
    assert result.ret == pytest.ExitCode.TESTS_FAILED


def test_fail_under_passing_prints_no_failed_line(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_ok="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert "never approve refunds" in agent.ask()
    assert "verify identity" in agent.ask()
""",
    )
    result = pytester.runpytest("--clastogen", "--clastogen-fail-under=50")
    assert "FAILED" not in result.stdout.str().split("Mutation Testing Summary", 1)[1]
    assert result.ret == pytest.ExitCode.OK


def test_html_report_matches_json(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_k2="import pytest, agent\n" + _STRONG + _UNRELATED)
    html_out = pytester.path / "report" / "out.html"
    pytester.runpytest("--clastogen", f"--clastogen-html={html_out}")
    html = html_out.read_text(encoding="utf-8")
    assert "<title>Clastogen Report</title>" in html
    assert ">100.0%<" in html
    assert html.count('<span class="badge" style="background:var(--killed)">KILLED</span>') == 2
    assert 'aria-label="Mutant status distribution"' in html
    assert html.count('title="KILLED">✓</span>') == 2
    assert "<script" not in html


def test_json_and_html_report_the_same_executions(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_edge="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1)
def test_skips_under_mutation():
    if "never approve" not in agent.ask():
        pytest.skip("skipped during mutation")

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=4)
def test_errors_under_mutation():
    if "verify identity" not in agent.ask():
        raise KeyError("boom")
""",
    )
    json_out, html_out = pytester.path / "out.json", pytester.path / "out.html"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}", f"--clastogen-html={html_out}")
    data = json.loads(json_out.read_text(encoding="utf-8"))
    html = html_out.read_text(encoding="utf-8")

    assert "Skipped (Excl.)         : 0" in result.stdout.str()
    for status in data["counts"]:
        assert f'<div class="l">{status.lower()}</div>' in html

    statuses = [e["status"] for e in data["executions"]]
    assert {"SURVIVED", "ERROR", "SKIPPED"} <= set(statuses)
    for status in set(statuses):
        assert html.count(f'title="{status}">') == statuses.count(status)

    unmeasured = [e for e in data["executions"] + data["results"] if e["status"] in ("ERROR", "SKIPPED")]
    assert unmeasured
    assert all(e["sample_count"] is None and e["llr"] is None for e in unmeasured)
