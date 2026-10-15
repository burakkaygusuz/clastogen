"""Tests for pytest-clastogen plugin."""

AGENT = '''
PROMPT = """You must never approve refunds over $50.
You must always verify identity before sharing balances."""

def ask():
    return PROMPT.lower()
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
