from collections.abc import Iterable
from typing import Any

from clastogen.models import (
    BEHAVIOR_ORDER,
    Behavior,
    MutantExecution,
    MutantStatus,
    MutationSummary,
    OracleRecord,
    RuleCoverage,
)
from clastogen.stats.assertions import compute_wilson_interval

_RANK = {status: rank for rank, status in enumerate(MutantStatus)}
MEASURED = frozenset({MutantStatus.KILLED, MutantStatus.SURVIVED, MutantStatus.INCONCLUSIVE})


def _rule_coverage(
    unique: tuple[MutantExecution, ...], constraints: Iterable[tuple[str, str]]
) -> tuple[RuleCoverage, ...]:
    """Tallies the merged measured mutants of each rule; the rules sort by (target, text)."""
    by_rule: dict[tuple[str, str], list[MutantExecution]] = {}
    for r in unique:
        if r.status in MEASURED:
            by_rule.setdefault((r.target, r.original_snippet), []).append(r)
    rows = []
    for target, rule in sorted(set(constraints)):
        mutants = by_rule.get((target, rule), [])
        killers = [r.test_id for r in mutants if r.status == MutantStatus.KILLED]
        rows.append(RuleCoverage(target, rule, len(killers), len(mutants), min(killers, default=None)))
    return tuple(rows)


def summarize(
    results: Iterable[MutantExecution],
    constraints: Iterable[tuple[str, str]] = (),
    oracle: Iterable[OracleRecord] = (),
    unmutated: Iterable[tuple[str, str]] = (),
) -> MutationSummary:
    """Merges records per mutant by status precedence and scores killed / (killed + survived + inconclusive).

    constraints are the (target, rule) pairs extracted from the prompts; each is killed, survived (measured, never
    killed) or untested (no mutant measured), whatever the score says. A mutant gets the strongest label of any probe
    in oracle (changed, then suspected, then no_change_found), so a probe added later can only raise a label.
    unmutated are the (target, sentence) pairs that rule discovery skipped.
    """
    behavior: dict[str, Behavior] = {}
    for o in oracle:
        behavior[o.mutant_id] = max(behavior.get(o.mutant_id, o.label), o.label, key=BEHAVIOR_ORDER.index)
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
    behavior = {m_id: v for m_id, v in behavior.items() if m_id in merged and merged[m_id].status in MEASURED}
    counts = dict.fromkeys(MutantStatus, 0)
    for r in unique:
        counts[r.status] += 1

    measurable = sum(counts[status] for status in MEASURED)
    score = counts[MutantStatus.KILLED] / measurable * 100.0 if measurable else None

    return MutationSummary(
        results=unique,
        counts=counts,
        score=score,
        calls=calls,
        fixed_calls=fixed_calls,
        rules=_rule_coverage(unique, constraints),
        unmutated=tuple(sorted(set(unmutated))),
        behavior=behavior,
    )


def rule_line(r: RuleCoverage) -> str:
    """Returns the evidence of one rule, shared by the terminal and the Markdown report."""
    if r.status == "untested":
        return "untested: no measured mutant"
    line = f"{r.status}: {r.killed} of {r.mutants} mutants killed"
    return f"{line} by {r.killed_by.split('::')[-1]}" if r.killed_by else line


def weakest_first(rules: Iterable[RuleCoverage]) -> list[RuleCoverage]:
    """Orders rules untested, survived, killed, so a cut list hides the best protected ones."""
    return sorted(rules, key=lambda r: ("untested", "survived", "killed").index(r.status))


MERGE_NOTE = (
    "A mutant is killed if any test kills it; a rule's mutants are those that were measured; "
    "the test named is the lowest test ID among the killers."
)


def survivor_tests(executions: Iterable[MutantExecution]) -> dict[str, list[str]]:
    """Maps each mutant to every test that still passed with it."""
    passed_in: dict[str, list[str]] = {}
    for e in executions:
        if e.status == MutantStatus.SURVIVED:
            passed_in.setdefault(e.mutant_id, []).append(e.test_id)
    return passed_in


def behavior_label(summary: MutationSummary, mutant_id: str) -> str | None:
    """Returns the oracle's label of a mutant, or None when no probe measured it.

    The oracle can fail to find a drop but cannot prove there is none, so it never says "equivalent".
    """
    if (label := summary.behavior.get(mutant_id)) is None:
        return None
    return {"changed": "behavior changed", "suspected": "behavior suspected"}.get(label, "no behavior change found")


# (probe label, killed) -> exposure bucket; suspected ignores the kill status, and a mutant no probe labelled is absent.
_QUADRANT = {
    ("changed", True): "caught",
    ("changed", False): "blind_spots",
    ("no_change_found", True): "false_kills",
    ("no_change_found", False): "no_change_found",
    ("suspected", True): "suspected",
    ("suspected", False): "suspected",
}


def exposure(summary: MutationSummary) -> dict[str, Any] | None:
    """Crosses the measured mutants' kill status with the oracle's label (not killed includes INCONCLUSIVE).

    Returns None when no probe ran. suspected counts killed and surviving mutants alike and stays out of the other
    quadrants. exposure_percent is the share of the behavior-changing mutants that were killed, None when no mutant
    changed behavior.
    """
    if not summary.behavior:
        return None
    counts = dict.fromkeys(("caught", "blind_spots", "no_change_found", "false_kills", "suspected", "unlabelled"), 0)
    for r in summary.results:
        if r.status in MEASURED:
            key = (summary.behavior.get(r.mutant_id), r.status == MutantStatus.KILLED)
            counts[_QUADRANT.get(key, "unlabelled")] += 1
    changed_total = counts["caught"] + counts["blind_spots"]
    return {**counts, "exposure_percent": counts["caught"] / changed_total * 100.0 if changed_total else None}


def exposure_line(summary: MutationSummary) -> str | None:
    """Returns the second axis next to the score, or None when no probe ran."""
    if (e := exposure(summary)) is None:
        return None
    percent = e["exposure_percent"]
    head = (
        "Behavior-changing mutants killed: N/A (no mutant changed behavior)"
        if percent is None
        else f"Behavior-changing mutants killed: {e['caught']} of {e['caught'] + e['blind_spots']} ({percent:.1f}%)"
    )
    return (
        f"{head}; defects: {e['blind_spots']} blind spots, {e['false_kills']} killed without a behavior change; "
        f"not proven: {e['no_change_found']} survived with no behavior change found (no proof of equivalence), "
        f"{e['unlabelled']} unlabelled (no probe measured them); "
        f"suspected: {e['suspected']} (a drop of delta is not ruled out)"
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


def score_interval(summary: MutationSummary) -> tuple[float, float] | None:
    """Returns the Wilson 95% interval of the score in percent, or None when no mutant was measured.

    It treats the measured mutants as independent trials, which they are not, and it leaves out the SPRT error rate.
    """
    measurable = sum(summary.counts[s] for s in MEASURED)
    if not measurable:
        return None
    low, high = compute_wilson_interval(summary.counts[MutantStatus.KILLED], measurable)
    return low * 100.0, high * 100.0


def score_line(summary: MutationSummary) -> str:
    """Returns the one-line score text shared by the terminal summary and the Markdown report."""
    score = "N/A" if summary.score is None else f"{summary.score:.1f}%"
    counts = summary.counts
    killed = counts[MutantStatus.KILLED]
    detail = f"{killed} of {sum(counts[s] for s in MEASURED)} killed"
    unscored = (MutantStatus.ERROR, MutantStatus.SKIPPED, MutantStatus.UNMEASURED, MutantStatus.SUPPRESSED)
    if excluded := [f"{n} {s.value.lower()}" for s in unscored if (n := counts[s])]:
        detail += f"; not scored: {', '.join(excluded)}"
    line = f"Mutation Score: {score} ({detail})"
    if interval := score_interval(summary):
        line += f"; Wilson 95% CI {interval[0]:.1f}%-{interval[1]:.1f}%"
    return line


def below_threshold(summary: MutationSummary, threshold: float) -> bool:
    """A session without measurable mutants fails any positive threshold."""
    if summary.score is None:
        return threshold > 0.0
    return summary.score < threshold
