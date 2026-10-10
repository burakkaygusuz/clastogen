from __future__ import annotations

import random
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from clastogen.models import MutantExecution, MutationSummary
from clastogen.reporting.scoring import calls_line, summarize
from clastogen.types import MutantStatus


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


@pytest.mark.parametrize("index", range(len(PRECEDENCE) - 1))
def test_precedence_stronger_status_wins_in_either_order(index: int) -> None:
    strong, weak = PRECEDENCE[index], PRECEDENCE[index + 1]
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
