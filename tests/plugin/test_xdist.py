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
    assert result.ret == pytest.ExitCode.OK

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert {b["test_id"].split("::")[-1] for b in data["baselines"]} == {"test_refunds", "test_identity"}
    assert data["total_mutants"] == 2
    assert data["counts"]["KILLED"] == 2
    assert data["mutation_score"] == 100.0
