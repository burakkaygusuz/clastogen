from dataclasses import dataclass, field

from clastogen.models.mutation import BaselineRecord, MutantExecution, MutationSummary


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

    def __post_init__(self) -> None:
        for name in ("max_mutants", "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        for name in ("delta", "p0"):
            value = getattr(self, name)
            if name == "p0" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be a number in (0, 1), got {value!r}")
