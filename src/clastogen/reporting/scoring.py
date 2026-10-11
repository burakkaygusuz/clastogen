from collections.abc import Iterable

from clastogen.models import MutantExecution, MutationSummary
from clastogen.types import MutantStatus

_RANK = {status: rank for rank, status in enumerate(MutantStatus)}
MEASURED = frozenset({MutantStatus.KILLED, MutantStatus.SURVIVED, MutantStatus.INCONCLUSIVE})


def summarize(results: Iterable[MutantExecution], constraints: Iterable[tuple[str, str]] = ()) -> MutationSummary:
    """Merges records per mutant by status precedence and scores killed / (killed + survived + inconclusive).

    constraints are the (target, rule) pairs extracted from the prompts; each is killed, survived (measured, never
    killed) or untested (no mutant measured), whatever the score says.
    """
    merged: dict[str, MutantExecution] = {}
    calls = fixed_calls = 0
    for r in results:
        if r.sample_count is not None and r.fixed_n is not None:
            calls += r.sample_count
            fixed_calls += r.fixed_n
        current = merged.get(r.mutant_id)
        if current is None or (_RANK[r.status], r.test_id) < (_RANK[current.status], current.test_id):
            merged[r.mutant_id] = r

    unique = tuple(merged[m_id] for m_id in sorted(merged))
    counts = dict.fromkeys(MutantStatus, 0)
    for r in unique:
        counts[r.status] += 1

    measurable = sum(counts[status] for status in MEASURED)
    score = counts[MutantStatus.KILLED] / measurable * 100.0 if measurable else None

    killed_keys = {(r.target, r.original_snippet) for r in unique if r.status == MutantStatus.KILLED}
    measured_keys = {(r.target, r.original_snippet) for r in unique if r.status in MEASURED}
    killed = survived = 0
    untested: list[str] = []
    for key in sorted(set(constraints)):
        if key in killed_keys:
            killed += 1
        elif key in measured_keys:
            survived += 1
        else:
            untested.append(key[1])
    return MutationSummary(
        results=unique,
        counts=counts,
        score=score,
        calls=calls,
        fixed_calls=fixed_calls,
        constraints_killed=killed,
        constraints_survived=survived,
        untested=tuple(untested),
    )


def operator_rates(summary: MutationSummary) -> dict[str, tuple[int, int]]:
    """Maps each operator to (killed, measured) over the merged results, with the score's denominator."""
    rates: dict[str, tuple[int, int]] = {}
    for r in summary.results:
        if r.status in MEASURED:
            killed, measured = rates.get(r.operator_name, (0, 0))
            rates[r.operator_name] = (killed + (r.status == MutantStatus.KILLED), measured + 1)
    return dict(sorted(rates.items()))


def operator_line(summary: MutationSummary) -> str | None:
    """Returns the per-operator kill counts, or None when no mutant was measured."""
    rates = operator_rates(summary)
    if not rates:
        return None
    return "By operator: " + ", ".join(f"{op} {k}/{n} ({k / n:.0%})" for op, (k, n) in rates.items())


def coverage_line(summary: MutationSummary) -> str | None:
    """Returns the constraint coverage text, or None when no constraint was extracted."""
    if summary.coverage is None:
        return None
    untested = len(summary.untested)
    return (
        f"Constraint Coverage: {summary.constraints_killed} of {summary.constraints_total} constraints killed "
        f"({summary.coverage:.1f}%); {summary.constraints_survived} survived or inconclusive, {untested} untested"
    )


def calls_line(summary: MutationSummary) -> str | None:
    """Returns the SPRT call count against a fixed-N test with the same error rates, or None when nothing was sampled."""
    if not summary.fixed_calls:
        return None
    saved = (1 - summary.calls / summary.fixed_calls) * 100.0
    outcome = f"{saved:.0f}% fewer" if saved > 0 else "no saving"
    return f"Calls: {summary.calls} (a fixed-N test with the same error rates needs {summary.fixed_calls}: {outcome})"


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
