from __future__ import annotations

import json

import pytest

AGENT = '''
import json

PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances."""


def ask() -> dict[str, bool]:
    """Simulates an SDK: the prompt is JSON-encoded into a request body and decoded on the 'server'."""
    payload = json.loads(json.dumps({"messages": [{"role": "system", "content": PROMPT}]}))
    system = payload["messages"][0]["content"]
    return {
        "refuses_refunds": "never approve refunds over $50." in system,
        "verifies_identity": "always verify identity" in system,
    }
'''


def test_prompt_serialized_by_sdk_is_observed(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent=AGENT,
        test_sdk="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_strong():
    assert agent.ask() == {"refuses_refunds": True, "verifies_identity": True}

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
def test_weak():
    assert len(agent.ask()) == 2
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["counts"]["KILLED"] == 2
    assert data["counts"]["SURVIVED"] == 0
    assert data["mutation_score"] == 100.0
    assert {r["test_id"].split("::")[-1] for r in data["results"]} == {"test_strong"}
    assert result.ret == pytest.ExitCode.OK
