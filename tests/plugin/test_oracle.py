from __future__ import annotations

import json

import pytest
from pytest_check import check

from tests.plugin import AGENT, STRONG, WEAK

PROBE = """
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=20)
def test_refund_probe():
    assert "never approve refunds over $50." in agent.PROMPT
"""

NOOP_PROBE = """
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=20)
def test_noop_probe():
    assert agent.PROMPT
"""


def _run(pytester: pytest.Pytester, *tests: str, oracle: bool = True) -> tuple[str, dict]:
    pytester.makepyfile(agent=AGENT, test_o="import pytest, agent\n" + "\n".join(tests))
    json_out = pytester.path / "out.json"
    args = ["--clastogen", f"--clastogen-json={json_out}"] + (["--clastogen-oracle"] if oracle else [])
    result = pytester.runpytest(*args)
    return result.stdout.str(), json.loads(json_out.read_text(encoding="utf-8"))


def test_killed_mutants_split_into_caught_and_killed_without_a_behavior_change(pytester: pytest.Pytester) -> None:
    stdout, data = _run(pytester, STRONG, PROBE)
    check.equal(data["mutation_score"], 100.0)
    check.equal(data["total_mutants"], 2)
    behavior = data["behavior"]
    check.equal(
        {
            k: behavior[k]
            for k in ("caught", "false_kills", "blind_spots", "no_change_found", "suspected", "unlabelled")
        },
        {
            "caught": 1,
            "false_kills": 1,
            "blind_spots": 0,
            "no_change_found": 0,
            "suspected": 0,
            "unlabelled": 0,
        },
    )
    check.equal(behavior["exposure_percent"], 100.0)
    check.is_in("Behavior-changing mutants killed: 1 of 1 (100.0%)", stdout)
    check.is_in("behavior changed", stdout)
    check.is_in("no behavior change found", stdout)


def test_surviving_mutants_split_into_blind_spots_and_no_change_found(pytester: pytest.Pytester) -> None:
    stdout, data = _run(pytester, WEAK, PROBE)
    check.equal(data["mutation_score"], 0.0)
    check.equal(data["behavior"]["blind_spots"], 1)
    check.equal(data["behavior"]["no_change_found"], 1)
    check.equal(data["behavior"]["exposure_percent"], 0.0)
    check.is_in("Behavior-changing mutants killed: 0 of 1 (0.0%); defects: 1 blind spots", stdout)


@pytest.mark.parametrize(
    ("samples", "label", "phrase"),
    [(6, "suspected", "behavior suspected"), (7, "no_change_found", "no behavior change found")],
)
def test_a_probe_rules_out_a_drop_only_with_enough_samples(
    pytester: pytest.Pytester, samples: int, label: str, phrase: str
) -> None:
    # All passes on both sides: the upper bound of the drop is 0.311 at 6 samples (>= delta) and 0.279 at 7.
    probe = NOOP_PROBE.replace("oracle_samples=20", f"oracle_samples={samples}")
    stdout, data = _run(pytester, WEAK, probe)
    check.equal(set(data["behavior"]["labels"].values()), {label})
    check.is_in(phrase, stdout)


def test_probe_never_adds_an_sprt_verdict_of_its_own(pytester: pytest.Pytester) -> None:
    _, data = _run(pytester, PROBE)
    check.equal(data["total_mutants"], 0)
    check.equal(data["baselines"], [])


def test_adding_a_probe_never_removes_a_behavior_changed_label(pytester: pytest.Pytester) -> None:
    _, one = _run(pytester, WEAK, PROBE)
    _, two = _run(pytester, WEAK, PROBE, NOOP_PROBE)
    changed_one = {m for m, label in one["behavior"]["labels"].items() if label == "changed"}
    changed_two = {m for m, label in two["behavior"]["labels"].items() if label == "changed"}
    check.is_true(changed_one)
    check.is_true(changed_one <= changed_two)


def test_without_the_flag_probes_run_as_plain_tests_and_nothing_is_labelled(pytester: pytest.Pytester) -> None:
    stdout, data = _run(pytester, STRONG, PROBE, oracle=False)
    check.equal(data["mutation_score"], 100.0)
    check.is_not_in("behavior", data)
    check.is_not_in("Behavior-changing", stdout)


def test_a_failing_probe_on_the_original_prompt_is_not_measured(pytester: pytest.Pytester) -> None:
    broken = PROBE.replace("never approve refunds over $50.", "text that is not in the prompt")
    pytester.makepyfile(agent=AGENT, test_o="import pytest, agent\n" + STRONG + broken)
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", "--clastogen-oracle", f"--clastogen-json={json_out}")
    result.assert_outcomes(passed=1, failed=1)
    check.is_not_in("behavior", json.loads(json_out.read_text(encoding="utf-8")))


def test_probe_records_survive_parallel_workers(pytester: pytest.Pytester, subprocess_env: None) -> None:
    pytester.makepyfile(agent=AGENT, test_a="import pytest, agent\n" + WEAK, test_b="import pytest, agent\n" + PROBE)
    json_out = pytester.path / "out.json"
    result = pytester.runpytest_subprocess(
        "--clastogen", "--clastogen-oracle", "-n", "2", "--dist", "loadfile", f"--clastogen-json={json_out}"
    )
    assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()
    behavior = json.loads(json_out.read_text(encoding="utf-8"))["behavior"]
    check.equal((behavior["blind_spots"], behavior["no_change_found"]), (1, 1))


def test_probes_alone_without_the_flag_say_why_nothing_was_measured(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=AGENT, test_o="import pytest, agent\n" + PROBE)
    stdout = pytester.runpytest("--clastogen").stdout.str()
    check.is_in("run only with --clastogen-oracle", stdout)


def test_a_flaky_probe_is_not_measured(pytester: pytest.Pytester) -> None:
    flaky = """
import itertools
_n = itertools.count(1)

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=20)
def test_flaky_probe():
    assert next(_n) % 2
"""
    _, data = _run(pytester, WEAK, flaky)
    check.is_not_in("behavior", data)


TROUBLES = ["raise KeyError('boom')", "pytest.skip('cannot run')"]


@pytest.mark.parametrize("trouble", TROUBLES)
def test_a_probe_that_cannot_rerun_on_the_original_prompt_is_not_measured(
    pytester: pytest.Pytester, trouble: str
) -> None:
    # The first call is the probe's own plain run; every rerun after it hits the trouble.
    probe = f"""
import itertools
_n = itertools.count(1)

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=20)
def test_troubled_probe():
    if next(_n) > 1:
        {trouble}
    assert agent.PROMPT
"""
    stdout, data = _run(pytester, WEAK, probe)
    check.is_in("2 passed", stdout)  # the trouble is a warning, never a failure of the probe itself
    check.is_not_in("behavior", data)


@pytest.mark.parametrize("trouble", TROUBLES)
def test_a_mutant_the_probe_cannot_run_on_is_left_unlabelled(pytester: pytest.Pytester, trouble: str) -> None:
    probe = f"""
@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=20)
def test_troubled_probe():
    if "never approve refunds over $50." not in agent.PROMPT:
        {trouble}
    assert agent.PROMPT
"""
    stdout, data = _run(pytester, WEAK, probe)
    check.is_in("2 passed", stdout)
    # Only the mutant that leaves the refund rule alone is measured; the other one is unlabelled.
    check.equal(len(data["behavior"]["labels"]), 1)
    check.equal(data["behavior"]["unlabelled"], 1)
