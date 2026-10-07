from collections.abc import Iterable

from clastogen.models import MutantExecution, MutationSummary
from clastogen.types import MutantStatus

_RANK = {status: rank for rank, status in enumerate(MutantStatus)}
MEASURED = frozenset({MutantStatus.KILLED, MutantStatus.SURVIVED, MutantStatus.INCONCLUSIVE})


def summarize(results: Iterable[MutantExecution]) -> MutationSummary:
    """Merges records per mutant by status precedence and scores killed / (killed + survived + inconclusive)."""
    merged: dict[str, MutantExecution] = {}
    for r in results:
        current = merged.get(r.mutant_id)
        if current is None or (_RANK[r.status], r.test_id) < (_RANK[current.status], current.test_id):
            merged[r.mutant_id] = r

    unique = tuple(merged[m_id] for m_id in sorted(merged))
    counts = dict.fromkeys(MutantStatus, 0)
    for r in unique:
        counts[r.status] += 1

    measurable = sum(counts[status] for status in MEASURED)
    score = counts[MutantStatus.KILLED] / measurable * 100.0 if measurable else None
    return MutationSummary(results=unique, counts=counts, score=score)


def score_line(summary: MutationSummary) -> str:
    """Returns the one-line score text shared by the terminal summary and the Markdown report."""
    score = "N/A" if summary.score is None else f"{summary.score:.1f}%"
    counts = summary.counts
    killed = counts[MutantStatus.KILLED]
    detail = f"{killed} of {sum(counts[s] for s in MEASURED)} killed"
    unscored = (MutantStatus.ERROR, MutantStatus.SKIPPED, MutantStatus.SUPPRESSED)
    if excluded := [f"{n} {s.value.lower()}" for s in unscored if (n := counts[s])]:
        detail += f"; not scored: {', '.join(excluded)}"
    return f"Mutation Score: {score} ({detail})"
