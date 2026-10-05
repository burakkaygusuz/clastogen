"""Clastogen: Mutation testing & statistical assertion framework for LLMs and Agents."""

from importlib.metadata import version

from clastogen.core.injection import override_prompt
from clastogen.core.mutator import PromptMutator
from clastogen.core.sprt import SPRT
from clastogen.models import Mutant, PassRateResult, RegressionResult, SPRTConfig, SPRTResult
from clastogen.stats import (
    assert_no_regression,
    assert_pass_rate,
    compute_wilson_interval,
    evaluate_pass_rate,
    evaluate_regression,
)
from clastogen.types import Decision

__all__ = [
    "SPRT",
    "Decision",
    "Mutant",
    "PassRateResult",
    "PromptMutator",
    "RegressionResult",
    "SPRTConfig",
    "SPRTResult",
    "assert_no_regression",
    "assert_pass_rate",
    "compute_wilson_interval",
    "evaluate_pass_rate",
    "evaluate_regression",
    "override_prompt",
]

__version__ = version("clastogen")
