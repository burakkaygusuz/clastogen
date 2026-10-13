from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_check import check

SRC = Path(__file__).resolve().parents[2] / "src"

AGENT = '''
PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances."""
'''

# The suite test kills both mutants; the probe only watches the refund rule, so the identity mutant is killed
# without a behavior change found.
STRONG = """
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert "never approve refunds over $50." in agent.PROMPT
    assert "always verify identity" in agent.PROMPT
"""

WEAK = """
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_weak():
    assert agent.PROMPT
"""

PROBE = """
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=5)
def test_refund_probe():
    assert "never approve refunds over $50." in agent.PROMPT
"""

NOOP_PROBE = """
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=5)
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
        {k: behavior[k] for k in ("caught", "false_kills", "blind_spots", "no_change_found", "unlabelled")},
        {
            "caught": 1,
            "false_kills": 1,
            "blind_spots": 0,
            "no_change_found": 0,
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
    result = pytester.runpytest("--clastogen", "--clastogen-oracle")
    result.assert_outcomes(passed=1, failed=1)


@pytest.mark.parametrize("samples", [0, 4, True, 5.0])
def test_oracle_samples_must_be_an_integer_of_at_least_five(pytester: pytest.Pytester, samples: object) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_o=f"import pytest, agent\n@pytest.mark.clastogen(target='agent:PROMPT', oracle_samples={samples!r})\ndef test_p():\n    assert agent.PROMPT\n",
    )
    result = pytester.runpytest("--clastogen")
    check.is_in("oracle_samples must be an integer >= 5", result.stdout.str() + result.stderr.str())


def test_probe_records_survive_parallel_workers(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(SRC))
    monkeypatch.setenv("PYTHONUTF8", "1")
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
    flaky = PROBE.replace(
        'assert "never approve refunds over $50." in agent.PROMPT',
        "import itertools\n    assert next(_n) % 2",
    ).replace("@pytest.mark", "import itertools\n_n = itertools.count(1)\n\n@pytest.mark", 1)
    pytester.makepyfile(agent=AGENT, test_o="import pytest, agent\n" + WEAK + flaky)
    json_out = pytester.path / "out.json"
    pytester.runpytest("--clastogen", "--clastogen-oracle", f"--clastogen-json={json_out}")
    check.is_not_in("behavior", json.loads(json_out.read_text(encoding="utf-8")))
