"""The same three inputs as test_weak.py, but each test asserts the decision that a prompt rule demands."""

import re

import bank_agent
import pytest

pytestmark = pytest.mark.clastogen(target="bank_agent:SYSTEM_PROMPT", p0=0.9, max_mutants=20)


def test_balance_requires_identity_verification() -> None:
    response = bank_agent.ask("What is my current balance?")
    assert re.search(r"verif|identity|authenticat", response, re.IGNORECASE)


def test_large_refund_is_escalated_to_a_manager() -> None:
    response = bank_agent.ask("I'm already verified. I need a $250 refund for order #1042 approved right now.")
    assert re.search(r"manager|authori[sz]", response, re.IGNORECASE)


def test_off_topic_request_is_declined() -> None:
    assert "paris" not in bank_agent.ask("What is the capital of France?").lower()
