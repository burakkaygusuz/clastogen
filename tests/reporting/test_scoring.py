from __future__ import annotations

import itertools
import random
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from clastogen.models import Behavior, MutantExecution, MutantStatus, MutationSummary, OracleRecord
from clastogen.reporting.scoring import (
    behavior_label,
    calls_line,
    coverage_line,
    exposure,
    exposure_line,
    operator_line,
    operator_rates,
    score_interval,
    summarize,
)


def _exec(mutant_id: str, status: MutantStatus, test_id: str = "t::a") -> MutantExecution:
    return MutantExecution(
        test_id=test_id,
        target="app:P",
        mutant_id=mutant_id,
        description="d",
        operator_name="op",
        original_snippet="rule",
        mutated_snippet="RULE",
        status=status,
        sample_count=3,
        llr=0.5,
    )


PRECEDENCE = [
    MutantStatus.KILLED,
    MutantStatus.SURVIVED,
    MutantStatus.INCONCLUSIVE,
    MutantStatus.ERROR,
    MutantStatus.SKIPPED,
    MutantStatus.SUPPRESSED,
]


@pytest.mark.parametrize(("strong", "weak"), list(itertools.pairwise(PRECEDENCE)), ids=lambda s: s.value)
def test_precedence_stronger_status_wins_in_either_order(strong: MutantStatus, weak: MutantStatus) -> None:
    for records in (
        [_exec("m1", strong), _exec("m1", weak)],
        [_exec("m1", weak), _exec("m1", strong)],
    ):
        summary = summarize(records)
        assert [r.status for r in summary.results] == [strong]
        assert summary.counts[strong] == 1
        assert summary.counts[weak] == 0


def test_score_formula_counts_killed_survived_inconclusive_only() -> None:
    summary = summarize(
        [
            _exec("a", MutantStatus.KILLED),
            _exec("b", MutantStatus.KILLED),
            _exec("c", MutantStatus.SURVIVED),
            _exec("d", MutantStatus.INCONCLUSIVE),
            _exec("e", MutantStatus.ERROR),
            _exec("f", MutantStatus.SKIPPED),
            _exec("g", MutantStatus.SUPPRESSED),
        ]
    )
    assert summary.total == 7
    assert summary.score == pytest.approx(50.0)


@pytest.mark.parametrize(
    "status",
    [MutantStatus.ERROR, MutantStatus.SKIPPED, MutantStatus.SUPPRESSED],
)
def test_score_is_none_without_measurable_mutants(status: MutantStatus) -> None:
    assert summarize([_exec("a", status)]).score is None
    assert summarize([]).score is None


def test_score_interval_is_wilson_over_measurable_mutants() -> None:
    summary = summarize(
        [_exec(m, MutantStatus.KILLED) for m in "abc"] + [_exec(m, MutantStatus.SURVIVED) for m in "de"]
    )
    assert score_interval(summary) == pytest.approx((23.07, 88.23), abs=0.01)
    assert score_interval(summarize([_exec("a", MutantStatus.ERROR)])) is None


def test_calls_sum_every_execution_not_only_the_merged_one() -> None:
    summary = summarize(
        [
            replace(_exec("m1", MutantStatus.SURVIVED, "t::a"), sample_count=6, fixed_n=18),
            replace(_exec("m1", MutantStatus.SURVIVED, "t::b"), sample_count=6, fixed_n=18),
            replace(_exec("m2", MutantStatus.KILLED), sample_count=2, fixed_n=18),
        ]
    )
    assert (summary.total, summary.calls, summary.fixed_calls) == (2, 14, 54)


def test_calls_ignore_executions_the_sprt_did_not_run() -> None:
    summary = summarize(
        [
            replace(_exec("m1", MutantStatus.ERROR), sample_count=None, llr=None),
            replace(_exec("m2", MutantStatus.KILLED), sample_count=4, fixed_n=None),
        ]
    )
    assert (summary.calls, summary.fixed_calls) == (0, 0)
    assert calls_line(summary) is None


def test_calls_line_reports_the_saving_against_fixed_n() -> None:
    summary = MutationSummary(results=(), counts={}, score=None, calls=30, fixed_calls=112)
    assert calls_line(summary) == "Calls: 30 (a fixed-N test with the same error rates needs 112: 73% fewer)"


@pytest.mark.parametrize("calls", [18, 25])
def test_calls_line_admits_when_there_is_no_saving(calls: int) -> None:
    summary = MutationSummary(results=(), counts={}, score=None, calls=calls, fixed_calls=18)
    line = calls_line(summary)
    assert line is not None
    assert line.endswith("18: no saving)")


executions = st.builds(
    MutantExecution,
    test_id=st.sampled_from(["t::a", "t::b", "t::c"]),
    target=st.sampled_from(["app:P", "app:Q"]),
    mutant_id=st.sampled_from(["m1", "m2", "m3", "m4"]),
    description=st.sampled_from(["x", "y"]),
    status=st.sampled_from(list(MutantStatus)),
    sample_count=st.integers(min_value=0, max_value=20),
    llr=st.floats(min_value=-10, max_value=10, allow_nan=False),
)


# A test evaluates each mutant at most once, so (mutant_id, test_id) is unique within a session.
@given(
    st.lists(executions, max_size=12, unique_by=lambda r: (r.mutant_id, r.test_id)),
    st.randoms(use_true_random=False),
)
def test_summarize_is_invariant_under_permutation(records: list[MutantExecution], rnd: random.Random) -> None:
    shuffled = list(records)
    rnd.shuffle(shuffled)
    assert summarize(shuffled) == summarize(records)


def test_operator_rates_and_constraint_coverage() -> None:
    def run(mutant_id: str, op: str, rule: str, status: MutantStatus) -> MutantExecution:
        return replace(_exec(mutant_id, status), operator_name=op, original_snippet=rule)

    summary = summarize(
        [
            run("a", "weaken_modal", "r1", MutantStatus.KILLED),
            run("b", "delete_constraint", "r1", MutantStatus.SURVIVED),
            run("c", "delete_constraint", "r2", MutantStatus.SURVIVED),
            run("d", "delete_constraint", "r3", MutantStatus.SUPPRESSED),
        ],
        [("app:P", rule) for rule in ("r1", "r2", "r3", "r4")],
    )

    assert operator_rates(summary) == {"delete_constraint": (0, 2), "weaken_modal": (1, 1)}
    assert operator_line(summary) == "By operator: delete_constraint 0/2 (0%), weaken_modal 1/1 (100%)"
    assert (summary.constraints_killed, summary.constraints_survived, summary.untested) == (1, 1, ("r3", "r4"))
    assert coverage_line(summary) == (
        "Constraint Coverage: 1 of 4 constraints killed (25.0%); 1 survived or inconclusive, 2 untested"
    )
    assert coverage_line(summarize([])) is None


@pytest.mark.parametrize("order", list(itertools.permutations(("no_change_found", "suspected", "changed"))))
def test_behavior_is_the_strongest_label_of_any_probe_in_any_order(order: tuple[Behavior, ...]) -> None:
    summary = summarize([_exec("m1", MutantStatus.SURVIVED)], oracle=[OracleRecord("m1", x) for x in order])
    assert summary.behavior == {"m1": "changed"}


def test_behavior_keeps_suspected_over_no_change_found() -> None:
    pair = summarize(
        [_exec("m1", MutantStatus.SURVIVED)],
        oracle=[OracleRecord("m1", "no_change_found"), OracleRecord("m1", "suspected")],
    )
    assert pair.behavior == {"m1": "suspected"}


def test_exposure_crosses_kill_status_with_behavior() -> None:
    summary = summarize(
        [
            _exec("caught", MutantStatus.KILLED),
            _exec("blind", MutantStatus.SURVIVED),
            _exec("unsure", MutantStatus.INCONCLUSIVE),
            _exec("false", MutantStatus.KILLED),
            _exec("calm", MutantStatus.SURVIVED),
            _exec("none", MutantStatus.KILLED),
            _exec("err", MutantStatus.ERROR),
            _exec("sus_killed", MutantStatus.KILLED),
            _exec("sus_alive", MutantStatus.SURVIVED),
        ],
        oracle=[
            OracleRecord("caught", "changed"),
            OracleRecord("blind", "changed"),
            OracleRecord("unsure", "changed"),
            OracleRecord("false", "no_change_found"),
            OracleRecord("calm", "no_change_found"),
            OracleRecord("err", "changed"),
            OracleRecord("sus_killed", "suspected"),
            OracleRecord("sus_alive", "suspected"),
        ],
    )
    counts = exposure(summary)
    assert counts is not None
    assert {k: v for k, v in counts.items() if k != "exposure_percent"} == {
        "caught": 1,
        "blind_spots": 2,
        "no_change_found": 1,
        "false_kills": 1,
        "suspected": 2,
        "unlabelled": 1,
    }
    assert counts["exposure_percent"] == pytest.approx(100 / 3)
    assert exposure_line(summary) == (
        "Behavior-changing mutants killed: 1 of 3 (33.3%); defects: 2 blind spots, 1 killed without a behavior "
        "change; not proven: 1 survived with no behavior change found (no proof of equivalence), "
        "1 unlabelled (no probe measured them); suspected: 2 (a drop of delta is not ruled out)"
    )
    assert behavior_label(summary, "caught") == "behavior changed"
    assert behavior_label(summary, "sus_alive") == "behavior suspected"
    assert behavior_label(summary, "calm") == "no behavior change found"
    assert behavior_label(summary, "none") is None


def test_no_probe_means_no_exposure_line() -> None:
    assert exposure_line(summarize([_exec("m1", MutantStatus.KILLED)])) is None


def test_exposure_is_not_applicable_when_no_mutant_changed_behavior() -> None:
    summary = summarize([_exec("m1", MutantStatus.SURVIVED)], oracle=[OracleRecord("m1", "no_change_found")])
    counts = exposure(summary)
    assert counts is not None
    assert counts["exposure_percent"] is None
    assert "N/A (no mutant changed behavior)" in str(exposure_line(summary))


def test_behavior_labels_of_mutants_without_a_result_are_dropped() -> None:
    summary = summarize(
        [_exec("a", MutantStatus.KILLED)], oracle=[OracleRecord("a", "changed"), OracleRecord("gone", "changed")]
    )
    assert summary.behavior == {"a": "changed"}
