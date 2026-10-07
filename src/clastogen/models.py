from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from clastogen.types import Decision, MutantStatus


@dataclass(frozen=True)
class Mutant:
    """Stable, content-addressed mutant specification."""

    id: str
    target_symbol: str
    operator_name: str
    original_snippet: str
    mutated_snippet: str
    mutated_prompt: str
    description: str

    @classmethod
    def create(
        cls,
        target_symbol: str,
        operator_name: str,
        original_snippet: str,
        mutated_snippet: str,
        mutated_prompt: str,
        description: str,
    ) -> Mutant:
        """Constructs a mutant with a stable 12-character SHA-256 hash ID."""
        content_key = f"{operator_name}:{target_symbol}:{original_snippet}->{mutated_snippet}"
        stable_id = hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:12]
        return cls(
            id=stable_id,
            target_symbol=target_symbol,
            operator_name=operator_name,
            original_snippet=original_snippet,
            mutated_snippet=mutated_snippet,
            mutated_prompt=mutated_prompt,
            description=description,
        )


@dataclass(frozen=True)
class MutantExecution:
    """Record of a single mutant evaluation by a test; sample_count and llr are None unless the SPRT ran."""

    test_id: str
    target: str
    mutant_id: str
    description: str
    status: MutantStatus
    sample_count: int | None = None
    llr: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class BaselineRecord:
    """Outcome of the unmutated baseline runs for a marked test; p0 is the Laplace estimate fed to the SPRT; error is set when a trial could not run cleanly."""

    test_id: str
    target: str
    successes: int
    runs: int
    p0: float
    stable: bool
    error: str | None = None


@dataclass(frozen=True)
class MutationSummary:
    """Order-independent aggregate of all mutant evaluations in a session."""

    results: tuple[MutantExecution, ...]
    counts: dict[MutantStatus, int]
    score: float | None

    @property
    def total(self) -> int:
        return len(self.results)


@dataclass
class ClastogenPluginState:
    """Type-safe state stored in pytest config.stash."""

    results: list[MutantExecution] = field(default_factory=list)
    baselines: list[BaselineRecord] = field(default_factory=list)
    summary: MutationSummary | None = None
    killed_mutants: set[str] = field(default_factory=set)
    suppressed_mutants: set[str] = field(default_factory=set)
    marked_tests_count: int = 0


@dataclass(frozen=True)
class ClastogenParams:
    """Arguments of @pytest.mark.clastogen; p0=None measures the baseline instead of assuming it."""

    target: str
    max_mutants: int = 5
    delta: float = 0.30
    p0: float | None = None
    max_steps: int = 20
    alpha: float = 0.05
    beta: float = 0.10

    def __post_init__(self) -> None:
        for name in ("max_mutants", "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        for name in ("delta", "p0", "alpha", "beta"):
            value = getattr(self, name)
            if name == "p0" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be a number in (0, 1), got {value!r}")
        if self.alpha + self.beta >= 1.0:
            raise ValueError(f"alpha + beta must be < 1, got alpha={self.alpha}, beta={self.beta}")


@dataclass(frozen=True)
class SPRTConfig:
    """Configuration for Wald's Sequential Probability Ratio Test."""

    alpha: float = 0.05
    beta: float = 0.10
    p0: float = 0.90
    p1: float = 0.60
    max_steps: int = 20

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        if self.alpha + self.beta >= 1.0:
            raise ValueError(f"alpha + beta must be < 1, got alpha={self.alpha}, beta={self.beta}")
        if not (0.0 < self.p1 < self.p0 < 1.0):
            raise ValueError(f"Requirement 0 < p1 < p0 < 1 violated: p0={self.p0}, p1={self.p1}")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError(f"max_steps must be an integer >= 1, got {self.max_steps}")

    @classmethod
    def from_absolute_drop(
        cls,
        p0: float,
        delta: float,
        alpha: float = 0.05,
        beta: float = 0.10,
        max_steps: int = 20,
    ) -> SPRTConfig:
        """Derives p1 by subtracting delta from p0, clamping p0 to [0.02, 0.99] to prevent division by zero."""
        clamped_p0 = min(max(p0, 0.02), 0.99)
        p1 = max(clamped_p0 - delta, 0.01)
        if p1 >= clamped_p0:
            raise ValueError(f"Drop delta={delta} leaves p1={p1} >= p0={clamped_p0}")
        return cls(alpha=alpha, beta=beta, p0=clamped_p0, p1=p1, max_steps=max_steps)


@dataclass(frozen=True)
class SPRTResult:
    decision: Decision
    sample_count: int
    cumulative_llr: float


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
