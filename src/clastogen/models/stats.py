from dataclasses import dataclass


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
