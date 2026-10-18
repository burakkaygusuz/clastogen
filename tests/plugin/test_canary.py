from __future__ import annotations

import json

import pytest

from clastogen.models import ClastogenParams
from clastogen.runner import plan_test
from tests.plugin import AGENT

_CANARY = "canary: test passes with an empty prompt; read the prompt at call time, or assert on behavior that the prompt causes"


def test_test_with_a_prompt_copied_at_import_fails_the_canary_in_every_report(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_copy="""
import pytest, agent

COPY = agent.PROMPT

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_copy():
    assert "never approve refunds over $50." in COPY
""",
    )
    paths = {"json": "out.json", "html": "out.html", "md": "out.md"}
    result = pytester.runpytest("--clastogen", *(f"--clastogen-{kind}={path}" for kind, path in paths.items()))

    assert result.ret == pytest.ExitCode.TESTS_FAILED, result.stdout.str()
    assert _CANARY in result.stdout.str()
    data = json.loads((pytester.path / "out.json").read_text(encoding="utf-8"))
    assert {(r["status"], r["error"]) for r in data["results"]} == {("ERROR", _CANARY)}
    assert data["counts"]["KILLED"] == 0
    for path in ("out.html", "out.md"):
        assert _CANARY in (pytester.path / path).read_text(encoding="utf-8")


def test_prompt_dependent_test_is_measured_without_a_canary(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_strong="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert "never approve refunds over $50." in agent.PROMPT
    assert "always verify identity" in agent.PROMPT
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    assert _CANARY not in result.stdout.str()
    assert json.loads(json_out.read_text(encoding="utf-8"))["counts"]["KILLED"] == 2


def test_test_that_killed_a_mutant_skips_the_canary(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_default="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_default():
    assert "NEVER" not in agent.PROMPT
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    assert _CANARY not in result.stdout.str()
    counts = json.loads(json_out.read_text(encoding="utf-8"))["counts"]
    assert (counts["KILLED"], counts["SURVIVED"], counts["ERROR"]) == (1, 1, 0)


def test_crash_on_the_empty_prompt_is_neutral(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_crash="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1, p0=0.90)
def test_crash():
    assert agent.PROMPT[0] == "Y"
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    assert _CANARY not in result.stdout.str()
    (mutant,) = json.loads(json_out.read_text(encoding="utf-8"))["results"]
    assert mutant["status"] == "SURVIVED"


_P0_TEST = """
import pytest, agent
from pathlib import Path

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, max_steps=5, p0=0.90)
def test_strong():
    with Path("calls.log").open("a") as f:
        f.write("x")
    assert agent.PROMPT is not None
"""


@pytest.mark.parametrize(
    ("cap", "trials", "unproven", "unmeasured"),
    [(4, 0, 0, 2), (5, 5, 1, 1), (8, 8, 0, 1), (12, 10, 2, 0), (13, 13, 0, 0)],
)
def test_canary_trials_count_against_the_budget_with_p0_given(
    pytester: pytest.Pytester, cap: int, trials: int, unproven: int, unmeasured: int
) -> None:
    pytester.makepyfile(agent=AGENT, test_budget=_P0_TEST)
    json_out = pytester.path / "out.json"
    pytester.runpytest("--clastogen", f"--clastogen-max-calls={cap}", f"--clastogen-json={json_out}")

    log = pytester.path / "calls.log"
    # The log holds the original call plus every trial, and the canary runs only when its 3 trials still fit after the mutants.
    assert (max(log.stat().st_size - 1, 0) if log.exists() else 0) == trials <= cap
    counts = json.loads(json_out.read_text(encoding="utf-8"))["counts"]
    assert (counts["SURVIVED"] + counts["INCONCLUSIVE"], counts["UNMEASURED"]) == (unproven, unmeasured)


@pytest.mark.parametrize(("p0", "calls"), [(None, 9 + 3 + 2 * 5), (0.90, 3 + 2 * 5)])
def test_plan_counts_the_canary_trials(p0: float | None, calls: int) -> None:
    params = ClastogenParams("agent:PROMPT", max_mutants=2, max_steps=5, p0=p0)
    prompt = "You must never approve refunds over $50.\nYou must always verify identity."

    assert plan_test("t::a", params, prompt, set(), with_oracle=False).calls == calls
