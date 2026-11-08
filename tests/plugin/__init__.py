"""Tests for pytest-clastogen plugin."""

AGENT = '''
PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances.
Keep the tone warm and friendly."""

def ask():
    return PROMPT.lower()
'''

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
