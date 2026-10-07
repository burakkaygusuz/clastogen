from enum import StrEnum
from typing import Any, TypedDict


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
    SUPPRESSED = "SUPPRESSED"


class RecordsPayload(TypedDict):
    """Records of one marked test; execnet serializes only exact builtin types, so statuses travel as plain str."""

    baselines: list[dict[str, Any]]
    results: list[dict[str, Any]]
