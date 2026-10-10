from __future__ import annotations

import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

AGENT = '''
PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances."""
'''


def test_parallel_workers_merge_results_and_baselines(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(SRC))
    pytester.makepyfile(
        agent=AGENT,
        test_refunds="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_refunds():
    assert "never approve refunds over $50." in agent.PROMPT
""",
        test_identity="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_identity():
    assert "always verify identity" in agent.PROMPT
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest_subprocess(
        "--clastogen", "-n", "2", "--dist", "loadfile", f"--clastogen-json={json_out}"
    )
    assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert {b["test_id"].split("::")[-1] for b in data["baselines"]} == {"test_refunds", "test_identity"}
    assert data["total_mutants"] == 2
    assert data["counts"]["KILLED"] == 2
    assert data["mutation_score"] == 100.0


def test_parallel_workers_reach_the_same_verdicts_as_a_serial_run(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(SRC))
    pytester.makepyfile(
        agent=AGENT,
        test_a_strong="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=3)
def test_strong():
    assert "never approve refunds over $50." in agent.PROMPT
    assert "always verify identity" in agent.PROMPT
""",
        test_b_weak="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=3)
def test_weak():
    assert agent.PROMPT
""",
    )
    runs = {}
    for name, args in (("serial", ()), ("parallel", ("-n", "2", "--dist", "loadfile"))):
        json_out = pytester.path / f"{name}.json"
        pytester.runpytest_subprocess("--clastogen", f"--clastogen-json={json_out}", *args)
        runs[name] = json.loads(json_out.read_text(encoding="utf-8"))

    def verdicts(data: dict) -> dict[str, str]:
        return {r["mutant_id"]: r["status"] for r in data["results"]}

    assert verdicts(runs["parallel"]) == verdicts(runs["serial"])
    assert runs["parallel"]["mutation_score"] == runs["serial"]["mutation_score"] == 100.0
    # The serial run skips mutants that the strong test killed; each worker only knows its own kills.
    assert runs["serial"]["calls"] < runs["parallel"]["calls"]
