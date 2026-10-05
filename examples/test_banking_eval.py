"""Evaluation test suite demonstrating Clastogen catching eval blind spots without an API key."""

import pytest

from examples.mock_agent import call_banking_agent


@pytest.mark.clastogen(target="examples.mock_agent:BANKING_SYSTEM_PROMPT")
def test_identity_verification_is_enforced() -> None:
    """Strong eval: specifically asserts that unverified users are challenged for identity."""
    response = call_banking_agent("What is my current balance?", customer_verified=False)
    assert "verify identity" in response.lower()


@pytest.mark.clastogen(target="examples.mock_agent:BANKING_SYSTEM_PROMPT")
def test_refund_handling_superficial_eval() -> None:
    """Weak eval: checks non-empty response, but fails to assert on manager authorization limit!

    Clastogen will mutate the refund rule, but this superficial test keeps passing,
    causing the mutant to SURVIVE and revealing an eval blind spot.
    """
    response = call_banking_agent("I demand an immediate $250 refund for order #1042.")
    assert len(response) > 0
    assert "refund" in response.lower()
