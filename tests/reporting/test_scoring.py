from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from clastogen.models import MutantExecution
from clastogen.reporting.scoring import summarize
from clastogen.types import MutantStatus


def _exec(mutant_id: str, status: MutantStatus, test_id: str = "t::a") -> MutantExecution:
    return MutantExecution(
        test_id=test_id,
        target="app:P",
        mutant_id=mutant_id,
        description="d",
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
