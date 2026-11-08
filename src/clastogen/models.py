from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypedDict, get_args

from clastogen.stats.types import check_error_rates, check_int, check_unit
from clastogen.trial import TrialBudget


class MutantStatus(StrEnum):
    """Final disposition of a mutant after evaluation."""

    KILLED = "KILLED"
    SURVIVED = "SURVIVED"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    UNMEASURED = "UNMEASURED"
    SUPPRESSED = "SUPPRESSED"


class BaselineState(StrEnum):
    """Outcome of the unmutated baseline runs; only STABLE yields an SPRT config."""

    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    SKIPPED = "SKIPPED"
    ERRORED = "ERRORED"


class RecordsPayload(TypedDict):
    """Records of one marked test; execnet serializes only exact builtin types, so statuses travel as plain str."""

    baselines: list[dict[str, Any]]
    executions: list[dict[str, Any]]
    constraints: list[list[str]]
    unmutated: list[list[str]]
    oracle: list[dict[str, Any]]
    tests: dict[str, int]
    targets: list[str]


BASELINE_RUNS = 10
MIN_BASELINE_RATE = 0.80
MIN_BASELINE_P0 = (math.ceil(MIN_BASELINE_RATE * BASELINE_RUNS) + 1) / (BASELINE_RUNS + 2)
NOISY_BASELINE_RATE = 0.90
CANARY_RUNS = 3


CUSTOM_OPERATOR = "custom"


def snippet_covers(original: str, sentence: str) -> bool:
    """True when a custom mutant's original snippet, minus surrounding whitespace and trailing .!?;:, sits inside the sentence.

    Sentences are stored without their trailing punctuation, so an original copied with its period must still match.
    """
    core = original.strip().rstrip(".!?;:").rstrip()
    return bool(core) and core in sentence


@dataclass(frozen=True)
class CustomMutant:
    """A hand- or agent-written mutant from .clastogen/mutants.toml; mutated "" deletes the snippet."""

    target: str
    original: str
    mutated: str
    description: str


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
        """Constructs a mutant with a stable 12-character SHA-256 hash ID over the JSON-encoded fields."""
        content_key = json.dumps([operator_name, target_symbol, original_snippet, mutated_snippet], ensure_ascii=False)
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
class RuleCoverage:
    """Evidence for one extracted rule: how many of its measured mutants were killed, and the lowest test ID among the tests that killed one.

    A mutant counts as killed when any test killed it; killed_by is the lowest test ID among those killers.
    """

    target: str
    rule: str
    killed: int
    mutants: int
    killed_by: str | None

    @property
    def status(self) -> Literal["killed", "survived", "untested"]:
        if self.killed:
            return "killed"
        return "survived" if self.mutants else "untested"


@dataclass(frozen=True)
class MutationSummary:
    """Order-independent aggregate of all mutant evaluations in a session.

    calls and fixed_calls sum sample_count and fixed_n over every sampled execution, not only the merged results.
    behavior maps a mutant ID to the strongest label of any probe; mutants no probe measured are absent.
    tests maps a target to T, the number of tests the SPRT alpha of its mutants is divided by.
    """

    results: tuple[MutantExecution, ...]
    counts: dict[MutantStatus, int]
    score: float | None
    calls: int = 0
    fixed_calls: int = 0
    rules: tuple[RuleCoverage, ...] = ()
    unmutated: tuple[tuple[str, str], ...] = ()
    tests: dict[str, int] = field(default_factory=dict)
    behavior: dict[str, Behavior] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def constraints_killed(self) -> int:
        return sum(r.status == "killed" for r in self.rules)

    @property
    def constraints_survived(self) -> int:
        return sum(r.status == "survived" for r in self.rules)

    @property
    def untested(self) -> tuple[str, ...]:
        return tuple(r.rule for r in self.rules if r.status == "untested")

    @property
    def constraints_total(self) -> int:
        return len(self.rules)

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


@dataclass
class ClastogenPluginState:
    """Type-safe state stored in pytest config.stash."""

    executions: list[MutantExecution] = field(default_factory=list)
    baselines: list[BaselineRecord] = field(default_factory=list)
    oracle: list[OracleRecord] = field(default_factory=list)
    summary: MutationSummary | None = None
    constraints: set[tuple[str, str]] = field(default_factory=set)
    unmutated: set[tuple[str, str]] = field(default_factory=set)
    tests: dict[str, int] = field(default_factory=dict)
    killed_mutants: set[str] = field(default_factory=set)
    suppressed_mutants: set[str] = field(default_factory=set)
    custom_mutants: tuple[CustomMutant, ...] = ()
    measured_targets: set[str] = field(default_factory=set)
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
            check_int(name, getattr(self, name), 1)
        if self.oracle_samples is not None:
            check_int("oracle_samples", self.oracle_samples, 5)
        for name in ("delta", "alpha", "beta"):
            check_unit(name, getattr(self, name))
        if self.p0 is not None:
            check_unit("p0", self.p0)
        if self.oracle_samples is not None and self.alpha >= 0.5:
            raise ValueError(f"alpha must be < 0.5 for a probe, got {self.alpha}")
        if self.p0 is not None and self.delta >= self.p0:
            raise ValueError(f"delta must be < p0, got delta={self.delta}, p0={self.p0}")
        if self.p0 is None and self.oracle_samples is None and self.delta >= MIN_BASELINE_P0:
            raise ValueError(
                f"delta must be < {MIN_BASELINE_P0:.2f}, the lowest p0 an accepted baseline can measure, got delta={self.delta}"
            )
        check_error_rates(self.alpha, self.beta)
