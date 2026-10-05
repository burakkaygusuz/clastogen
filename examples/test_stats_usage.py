"""Statistical assertions on seeded fake evaluators: zero API keys, deterministic."""

import random
from collections.abc import Callable

import pytest

from clastogen import assert_no_regression, assert_pass_rate


def fake_llm(pass_rate: float, seed: int) -> Callable[[], bool]:
    rng = random.Random(seed)
    return lambda: rng.random() < pass_rate


def test_order_intent_pass_rate() -> None:
    result = assert_pass_rate(fake_llm(0.97, seed=1), min_rate=0.90, description="order intent")
    assert result.observed_rate >= 0.90


def test_prompt_upgrade_does_not_regress() -> None:
    assert_no_regression(fake_llm(0.90, seed=2), fake_llm(0.90, seed=3), description="prompt v1 -> v2")


def test_broken_prompt_is_detected_as_regression() -> None:
    with pytest.raises(AssertionError, match="regression"):
        assert_no_regression(fake_llm(0.95, seed=4), fake_llm(0.40, seed=5), description="prompt v1 -> v3")
