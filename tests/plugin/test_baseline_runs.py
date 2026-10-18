from __future__ import annotations

import json

import pytest
from pytest_check import check

from tests.plugin import AGENT


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
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    (baseline,) = data["baselines"]
    check.equal(baseline["runs"], 10)
    check.equal(baseline["successes"], 10)
    check.almost_equal(baseline["p0"], 11 / 12)
    check.equal(baseline["stable"], True)
    check.is_in("p0=0.917", result.stdout.str())


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
    ("action", "error", "status", "mutant_error"),
    [
        (
            'raise RuntimeError("boom")',
            "unexpected execution error: RuntimeError",
            "ERROR",
            "baseline: unexpected execution error: RuntimeError",
        ),
        ('pytest.skip("flaky dependency")', "test skipped during baseline", "SKIPPED", None),
    ],
)
def test_broken_baseline_is_rejected_with_reason(
    pytester: pytest.Pytester, action: str, error: str, status: str, mutant_error: str | None
) -> None:
    pytester.makepyfile(agent=AGENT, test_breaks=_BREAKS_ON_RERUN.format(action=action))
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    result.assert_outcomes(passed=1)
    # The baseline error lands on the unmeasured mutants: ERROR fails the run, SKIPPED does not.
    assert result.ret == (pytest.ExitCode.TESTS_FAILED if mutant_error else pytest.ExitCode.OK), result.stdout.str()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    (baseline,) = data["baselines"]
    assert (baseline["stable"], baseline["runs"], baseline["error"]) == (False, 1, error)
    (mutant,) = data["results"]
    assert (mutant["status"], mutant["error"]) == (status, mutant_error)
    assert "Flaky tests" not in result.stdout.str()
    if mutant_error:
        assert mutant_error in result.stdout.str()


def test_flaky_baseline_records_its_mutants_as_skipped(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_breaks=_BREAKS_ON_RERUN.format(action="assert False"))
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    (mutant,) = data["results"]
    assert (mutant["status"], mutant["error"]) == ("SKIPPED", "baseline: unstable")
    assert "Flaky tests" in result.stdout.str()


@pytest.mark.parametrize("files", [("test_a.py", "test_b.py"), ("test_b.py", "test_a.py")])
def test_baseline_error_outcome_does_not_depend_on_test_order(
    pytester: pytest.Pytester, files: tuple[str, ...]
) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_a="""
import pytest, agent
ORIGINAL = agent.PROMPT

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=1)
def test_kills():
    assert agent.PROMPT == ORIGINAL
""",
        test_b=_BREAKS_ON_RERUN.format(action='raise RuntimeError("boom")'),
    )
    result = pytester.runpytest("--clastogen", *files)

    # test_a kills every mutant whose baseline broke in test_b, whichever runs first.
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    result.stdout.fnmatch_lines(["*Mutation Score: 100.0% (1 of 1 killed)*"])


def test_pytest_exit_during_baseline_stops_the_session(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_breaks=_BREAKS_ON_RERUN.format(action='pytest.exit("abort requested")'))
    result = pytester.runpytest("--clastogen")

    assert result.ret == pytest.ExitCode.INTERRUPTED, result.stdout.str()
    assert "abort requested" in result.stdout.str()


def test_no_baseline_when_every_mutant_is_already_killed(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_order="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert "never approve refunds over $50." in agent.PROMPT
    assert "must always verify identity" in agent.PROMPT

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


@pytest.mark.parametrize(("failing_calls", "warned"), [((2, 3), True), ((2,), False)])
def test_noisy_baseline_is_flagged_below_ninety_percent(
    pytester: pytest.Pytester, failing_calls: tuple[int, ...], warned: bool
) -> None:
    pytester.makepyfile(
        app='SYSTEM_PROMPT = "You are an assistant. You must never leak secrets."',
        test_noisy=f"""
import pytest

call_count = 0

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT", max_mutants=1)
def test_noisy():
    global call_count
    call_count += 1
    assert call_count not in {failing_calls}
""",
    )
    json_out, md_out = pytester.path / "out.json", pytester.path / "out.md"
    stdout = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}", f"--clastogen-md={md_out}").stdout.str()
    assert (
        "Noisy baselines (pass rate < 90%): false KILLs can exceed alpha for test_noisy.py::test_noisy" in stdout
    ) is warned
    assert [b["noisy"] for b in json.loads(json_out.read_text(encoding="utf-8"))["baselines"]] == [warned]
    assert (
        "Noisy baselines (false KILLs can exceed alpha): ` test_noisy `" in md_out.read_text(encoding="utf-8")
    ) is warned
