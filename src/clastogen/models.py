from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypedDict, get_args


class Decision(StrEnum):
    """Statistical outcome of a hypothesis test."""

    KILLED = "KILLED"
    SURVIVED = "SURVIVED"
    INCONCLUSIVE = "INCONCLUSIVE"


# Declaration order is the merge precedence used by clastogen.reporting.scoring.summarize.
class MutantStatus(StrEnum):
    """Final disposition of a mutant after evaluation."""

    KILLED = "KILLED"
    SURVIVED = "SURVIVED"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    UNMEASURED = "UNMEASURED"
    SUPPRESSED = "SUPPRESSED"


class RecordsPayload(TypedDict):
    """Records of one marked test; execnet serializes only exact builtin types, so statuses travel as plain str."""

    baselines: list[dict[str, Any]]
    executions: list[dict[str, Any]]
    constraints: list[list[str]]
    oracle: list[dict[str, Any]]


# Unmutated runs used to estimate p0; the Laplace estimate with n=10 keeps false KILLs on unchanged mutants low.
BASELINE_RUNS = 10
MIN_BASELINE_RATE = 0.80
# Laplace estimate at the fewest passes an accepted baseline can have.
MIN_BASELINE_P0 = (math.ceil(MIN_BASELINE_RATE * BASELINE_RUNS) + 1) / (BASELINE_RUNS + 2)
# Below this raw rate the selection by the 80% rule inflates p0, so false KILLs can exceed alpha (docs/statistics.md).
NOISY_BASELINE_RATE = 0.90
# Trials on an empty prompt after the mutants of a test that killed none: passing all of them means its survivors say nothing.
CANARY_RUNS = 3


def _check_int(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _check_unit(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be a number in (0, 1), got {value!r}")


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
    """Record of a single mutant evaluation by a test; sample_count, llr and fixed_n are None unless the SPRT ran.

    fixed_n is the sample size of a fixed-N test with the same error rates (see fixed_sample_size).
    """

    test_id: str
    target: str
    mutant_id: str
    description: str
    operator_name: str
    original_snippet: str
    mutated_snippet: str
    status: MutantStatus
    sample_count: int | None = None
    llr: float | None = None
    error: str | None = None
    fixed_n: int | None = None


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

    @property
    def noisy(self) -> bool:
        """True for a stable baseline below NOISY_BASELINE_RATE, where false KILLs can exceed alpha."""
        return self.stable and self.successes / self.runs < NOISY_BASELINE_RATE


# Weakest to strongest; probes merge to the strongest label. Plain strings, because execnet sends only builtins.
Behavior = Literal["no_change_found", "suspected", "changed"]
BEHAVIOR_ORDER: tuple[Behavior, ...] = get_args(Behavior)


@dataclass(frozen=True)
class OracleRecord:
    """One probe test's verdict on one mutant.

    changed is a significant pass rate drop of at least delta; suspected is no such drop, but a drop of delta is not
    ruled out either (the upper bound of the drop is at least delta); no_change_found rules it out.
    """

    mutant_id: str
    label: Behavior


@dataclass(frozen=True)
class MutationSummary:
    """Order-independent aggregate of all mutant evaluations in a session.

    calls and fixed_calls sum sample_count and fixed_n over every sampled execution, not only the merged results.
    behavior maps a mutant ID to the strongest label of any probe; mutants no probe measured are absent.
    """

    results: tuple[MutantExecution, ...]
    counts: dict[MutantStatus, int]
    score: float | None
    calls: int = 0
    fixed_calls: int = 0
    constraints_killed: int = 0
    constraints_survived: int = 0
    untested: tuple[str, ...] = ()
    behavior: dict[str, Behavior] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def constraints_total(self) -> int:
        return self.constraints_killed + self.constraints_survived + len(self.untested)

    @property
    def coverage(self) -> float | None:
        """Share of extracted constraints with a killed mutant, None when the prompts had no constraints."""
        return self.constraints_killed / self.constraints_total * 100.0 if self.constraints_total else None


@dataclass(frozen=True)
class RunPlan:
    """Worst-case work of one marked test, computed at collection; rules are empty for a probe, which records none."""

    test_id: str
    target: str
    rules: tuple[str, ...]
    mutants: int
    suppressed: int
    calls: int


class TrialBudget:
    """Counts the trials one process runs against --clastogen-max-calls; limit None means no cap.

    The original test call is not counted. A unit (the baseline, one mutant or one probe sample set) starts only when
    its worst case fits in what is left, so the cap is never exceeded.
    """

    def __init__(self, limit: int | None = None) -> None:
        self.remaining = limit

    def fits(self, units: int) -> bool:
        return self.remaining is None or self.remaining >= units

    def wrap(self, trial: Callable[[], bool]) -> Callable[[], bool]:
        def counted() -> bool:
            if self.remaining is not None:
                self.remaining -= 1
            return trial()

        return counted


@dataclass
class ClastogenPluginState:
    """Type-safe state stored in pytest config.stash."""

    executions: list[MutantExecution] = field(default_factory=list)
    baselines: list[BaselineRecord] = field(default_factory=list)
    oracle: list[OracleRecord] = field(default_factory=list)
    summary: MutationSummary | None = None
    constraints: set[tuple[str, str]] = field(default_factory=set)
    killed_mutants: set[str] = field(default_factory=set)
    suppressed_mutants: set[str] = field(default_factory=set)
    marked_tests_count: int = 0
    preview: list[RunPlan] = field(default_factory=list)
    budget: TrialBudget = field(default_factory=TrialBudget)


@dataclass(frozen=True)
class ClastogenParams:
    """Arguments of @pytest.mark.clastogen; p0=None measures the baseline instead of assuming it.

    oracle_samples makes the test a probe: with --clastogen-oracle it runs that many times on the original prompt and
    on every mutant to find the mutants that change behavior (delta is the minimum drop, alpha the significance and
    the one-sided confidence of the bound that rules a drop out), and it gets no SPRT verdict of its own.
    """

    target: str
    max_mutants: int = 5
    delta: float = 0.30
    p0: float | None = None
    max_steps: int = 20
    alpha: float = 0.05
    beta: float = 0.10
    oracle_samples: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_mutants", "max_steps"):
            _check_int(name, getattr(self, name), 1)
        if self.oracle_samples is not None:
            _check_int("oracle_samples", self.oracle_samples, 5)
        for name in ("delta", "alpha", "beta"):
            _check_unit(name, getattr(self, name))
        if self.p0 is not None:
            _check_unit("p0", self.p0)
        if self.oracle_samples is not None and self.alpha >= 0.5:
            raise ValueError(f"alpha must be < 0.5 for a probe, got {self.alpha}")
        if self.p0 is not None and self.delta >= self.p0:
            raise ValueError(f"delta must be < p0, got delta={self.delta}, p0={self.p0}")
        if self.p0 is None and self.oracle_samples is None and self.delta >= MIN_BASELINE_P0:
            raise ValueError(
                f"delta must be < {MIN_BASELINE_P0:.2f}, the lowest p0 an accepted baseline can measure, got delta={self.delta}"
            )
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
        _check_unit("alpha", self.alpha)
        _check_unit("beta", self.beta)
        if self.alpha + self.beta >= 1.0:
            raise ValueError(f"alpha + beta must be < 1, got alpha={self.alpha}, beta={self.beta}")
        if not (0.0 < self.p1 < self.p0 < 1.0):
            raise ValueError(f"Requirement 0 < p1 < p0 < 1 violated: p0={self.p0}, p1={self.p1}")
        _check_int("max_steps", self.max_steps, 1)

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
