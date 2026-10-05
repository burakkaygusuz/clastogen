from enum import StrEnum


class Decision(StrEnum):
    """Statistical outcome of a hypothesis test."""

    KILLED = "KILLED"
    SURVIVED = "SURVIVED"
    INCONCLUSIVE = "INCONCLUSIVE"


# Declaration order is the merge precedence used by clastogen.scoring.summarize.
class MutantStatus(StrEnum):
    """Final disposition of a mutant after evaluation."""

    KILLED = "KILLED"
    SURVIVED = "SURVIVED"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    SUPPRESSED = "SUPPRESSED"
