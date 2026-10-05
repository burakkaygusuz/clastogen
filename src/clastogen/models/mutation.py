from __future__ import annotations

import hashlib
from dataclasses import dataclass

from clastogen.types import MutantStatus


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
