from __future__ import annotations

import pytest

AGENT = '''
import asyncio

PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances."""

async def ask():
    await asyncio.sleep(0)
    return PROMPT.lower()
'''

_CONFTEST = """
import asyncio, inspect, pytest

@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        asyncio.run(pyfuncitem.obj())
        return True
"""


def test_async_test_is_mutated_via_pyfunc_call_hook(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(
        agent=AGENT,
        test_async="""
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", max_mutants=2)
async def test_strong():
    out = await agent.ask()
    assert "never approve refunds over $50." in out
    assert "verify identity" in out
""",
    )
    result = pytester.runpytest("--clastogen")
    assert "Mutation Score: 100.0% (2 of 2 killed)" in result.stdout.str()
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
