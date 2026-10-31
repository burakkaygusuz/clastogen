from __future__ import annotations

import json

import pytest

from tests.plugin import AGENT, STRONG, WEAK

PROBE = """
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2, oracle_samples=20)
def test_probe():
    assert agent.PROMPT
"""

FAILING_WEAK = WEAK.replace("assert agent.PROMPT", "assert not agent.PROMPT")


def _tests_per_target(pytester: pytest.Pytester, *args: str) -> dict[str, int]:
    json_out = pytester.path / "out.json"
    pytester.runpytest_subprocess("--clastogen", f"--clastogen-json={json_out}", *args)
    return json.loads(json_out.read_text(encoding="utf-8"))["tests_per_target"]


@pytest.fixture
def suite(pytester: pytest.Pytester, subprocess_env: None) -> pytest.Pytester:
    pytester.makepyfile(agent=AGENT, test_a_strong=STRONG, test_b_weak=WEAK, test_c_probe=PROBE)
    return pytester


def test_t_counts_the_marked_tests_of_a_target_and_not_probes(suite: pytest.Pytester) -> None:
    assert _tests_per_target(suite) == {"agent:PROMPT": 2}


def test_t_counts_only_tests_that_survive_deselection(suite: pytest.Pytester) -> None:
    assert _tests_per_target(suite, "-k", "strong") == {"agent:PROMPT": 1}


def test_t_counts_only_the_last_failed_tests_with_lf(pytester: pytest.Pytester, subprocess_env: None) -> None:
    pytester.makepyfile(agent=AGENT, test_a_strong=STRONG, test_b_weak=FAILING_WEAK)
    pytester.runpytest_subprocess()
    assert _tests_per_target(pytester, "--lf") == {"agent:PROMPT": 1}


def test_t_is_the_same_under_xdist_loadfile(suite: pytest.Pytester) -> None:
    assert _tests_per_target(suite, "-n", "2", "--dist", "loadfile") == {"agent:PROMPT": 2}


def test_t_is_multiplied_by_the_worker_count_under_dist_each(suite: pytest.Pytester) -> None:
    assert _tests_per_target(suite, "-n", "2", "--dist", "each") == {"agent:PROMPT": 4}


def test_a_bad_target_fails_its_test_and_not_the_session_under_xdist(
    pytester: pytest.Pytester, subprocess_env: None
) -> None:
    pytester.makepyfile(test_bad=STRONG.replace("agent:PROMPT", "agent:MISSING"), test_good=WEAK, agent=AGENT)
    result = pytester.runpytest_subprocess("--clastogen", "-n", "2")
    result.stdout.fnmatch_lines(["*Failed to resolve target*"])
    assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()
