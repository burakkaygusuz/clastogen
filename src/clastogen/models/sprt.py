from __future__ import annotations

from dataclasses import dataclass

from clastogen.types import Decision


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
