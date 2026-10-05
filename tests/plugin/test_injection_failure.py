from __future__ import annotations

import json

import pytest


def test_injection_failure_is_a_per_mutant_error_and_the_test_still_passes(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        agent="""
from dataclasses import dataclass

@dataclass(frozen=True)
class Cfg:
    system: str = "You must never approve refunds over $50. You must always verify identity."

CFG = Cfg()
""",
        test_frozen="""
import pytest, agent

@pytest.mark.clastogen(target="agent:CFG.system", max_mutants=2, p0=0.90)
def test_frozen():
    assert "never approve" in agent.CFG.system
""",
    )
    json_out = pytester.path / "out.json"
    result = pytester.runpytest("--clastogen", f"--clastogen-json={json_out}")

    result.assert_outcomes(passed=1)
    results = json.loads(json_out.read_text(encoding="utf-8"))["results"]
    assert [r["status"] for r in results] == ["ERROR", "ERROR"]
    assert all("prompt injection failed: FrozenInstanceError" in r["error"] for r in results)
