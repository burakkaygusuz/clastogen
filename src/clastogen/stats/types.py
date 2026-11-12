"""Public types of the statistical tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    """Statistical outcome of a hypothesis test."""

    KILLED = "KILLED"
    SURVIVED = "SURVIVED"
    INCONCLUSIVE = "INCONCLUSIVE"


def check_int(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def check_unit(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be a number in (0, 1), got {value!r}")


def check_error_rates(alpha: float, beta: float) -> None:
    if alpha + beta >= 1.0:
        raise ValueError(f"alpha + beta must be < 1, got alpha={alpha}, beta={beta}")


@dataclass(frozen=True)
class SPRTConfig:
    """Configuration for Wald's Sequential Probability Ratio Test."""

    alpha: float = 0.05
    beta: float = 0.10
    p0: float = 0.90
    p1: float = 0.60
    max_steps: int = 20

    def __post_init__(self) -> None:
        check_unit("alpha", self.alpha)
        check_unit("beta", self.beta)
        check_error_rates(self.alpha, self.beta)
        if not (0.0 < self.p1 < self.p0 < 1.0):
            raise ValueError(f"Requirement 0 < p1 < p0 < 1 violated: p0={self.p0}, p1={self.p1}")
        check_int("max_steps", self.max_steps, 1)

    @classmethod
    def from_absolute_drop(
        cls,
        p0: float,
        delta: float,
        alpha: float = 0.05,
        beta: float = 0.10,
        max_steps: int = 20,
    ) -> SPRTConfig:
        """Derives p1 by subtracting delta from p0."""
        return cls(alpha=alpha, beta=beta, p0=p0, p1=p0 - delta, max_steps=max_steps)


@dataclass(frozen=True)
class SPRTResult:
    decision: Decision
    sample_count: int
    cumulative_llr: float
    passes: int


@dataclass(frozen=True, slots=True)
class PassRateResult:
    """Outcome of a statistical pass rate assertion."""

    passed: bool
    observed_rate: float
    sample_count: int
    success_count: int
    ci_lower: float
    ci_upper: float
    confidence: float
    decided: bool
    description: str | None = None


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """Outcome of a one-sided test that a candidate evaluator regressed relative to a baseline."""

    regressed: bool
    p_value: float
    test_name: str
    baseline_rate: float
    candidate_rate: float
    sample_count: int
    description: str | None = None
