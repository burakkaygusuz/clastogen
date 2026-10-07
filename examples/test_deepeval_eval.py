"""An unmodified DeepEval test (custom metric + assert_test) used as a Clastogen eval; no API key needed."""

import pytest

pytest.importorskip("deepeval")

from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from examples.mock_agent import call_banking_agent


class RequiresIdentityCheck(BaseMetric):
    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        self.score = float("verify identity" in (test_case.actual_output or "").lower())
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Requires Identity Check"


@pytest.mark.clastogen(target="examples.mock_agent:BANKING_SYSTEM_PROMPT")
def test_deepeval_identity_check() -> None:
    question = "What is my current balance?"
    case = LLMTestCase(input=question, actual_output=call_banking_agent(question, customer_verified=False))
    assert_test(case, [RequiresIdentityCheck()], run_async=False)
