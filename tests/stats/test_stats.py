import itertools
import math
from collections.abc import Callable
from statistics import NormalDist

import pytest
from hypothesis import given
from hypothesis import strategies as st

from clastogen.stats.assertions import (
    assert_no_regression,
    assert_pass_rate,
    compare_pass_counts,
    compute_wilson_interval,
    drop_upper_bound,
    evaluate_pass_rate,
    evaluate_regression,
)


@pytest.mark.parametrize("toward", [-math.inf, math.inf])
def test_wilson_interval_edges_are_exact_for_any_z_rounding(monkeypatch: pytest.MonkeyPatch, toward: float) -> None:
    z = math.nextafter(NormalDist().inv_cdf(0.975), toward)

    def fake_inv_cdf(self: NormalDist, q: float) -> float:
        return z

    monkeypatch.setattr(NormalDist, "inv_cdf", fake_inv_cdf)
    assert compute_wilson_interval(0, 10)[0] == 0.0
    assert compute_wilson_interval(10, 10)[1] == 1.0


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((0, 0), "Total trials must be at least 1"),
        ((5, 4), "Successes"),
        ((1, 2, 1.5), "Confidence must be strictly between 0 and 1"),
    ],
)
def test_wilson_interval_validations(args: tuple[float, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compute_wilson_interval(*args)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(("n", "alpha"), [(5, 0.05), (20, 0.05), (30, 0.05), (20, 0.1)])
def test_drop_upper_bound_of_two_perfect_groups_is_the_wilson_gap(n: int, alpha: float) -> None:
    z2 = NormalDist().inv_cdf(1 - alpha) ** 2
    assert drop_upper_bound(n, n, n, alpha) == pytest.approx(z2 / (n + z2))


@given(st.integers(0, 20), st.integers(0, 20))
def test_drop_upper_bound_covers_the_observed_drop_and_grows_with_failures(base: int, cand: int) -> None:
    bound = drop_upper_bound(base, cand, 20)
    assert bound >= (base - cand) / 20 - 1e-12
    if cand > 0:
        assert drop_upper_bound(base, cand - 1, 20) >= bound


def _counting(outcome: bool) -> tuple[Callable[[], bool], list[int]]:
    calls = [0]

    def evaluator() -> bool:
        calls[0] += 1
        return outcome

    return evaluator, calls


def test_evaluate_pass_rate_perfect_evaluator_passes_after_wald_bound() -> None:
    evaluator, calls = _counting(True)

    res = evaluate_pass_rate(evaluator)

    # Defaults: H0 0.90 vs H1 0.70, alpha = beta = 0.05 -> ceil(ln 19 / ln(9/7)) = 12 passes.
    assert res.passed
    assert res.decided
    assert calls[0] == res.sample_count == res.success_count == 12
    assert res.observed_rate == 1.0


def test_assert_pass_rate_failing_evaluator_stops_early() -> None:
    evaluator, calls = _counting(False)

    with pytest.raises(AssertionError) as exc_info:
        assert_pass_rate(evaluator, description="payment_authorization")

    # ceil(ln 19 / ln 3) = 3 failures.
    assert calls[0] == 3
    message = str(exc_info.value)
    assert "Statistical pass rate assertion failed: 'payment_authorization'" in message
    assert "Observed Rate   : 0.0%" in message
    assert "indifference zone down to 70.0%" in message


def test_evaluate_pass_rate_truncation_decides_by_llr_sign() -> None:
    passing, _ = _counting(True)
    # T, T, F, T leaves LLR = 3 ln(7/9) + ln 3 > 0, short of the kill threshold ln 19.
    flaky = itertools.cycle([True, True, False]).__next__

    good = evaluate_pass_rate(passing, max_samples=5)
    bad = evaluate_pass_rate(flaky, max_samples=4)

    assert (good.passed, good.decided, good.sample_count) == (True, False, 5)
    assert (bad.passed, bad.decided, bad.sample_count) == (False, False, 4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_rate": 1.1}, "min_rate"),
        ({"tolerance": 0.0}, "tolerance"),
        ({"confidence": 1.0}, "confidence"),
        ({"max_samples": 0}, "max_samples"),
    ],
)
def test_evaluate_pass_rate_rejects_invalid_arguments(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_pass_rate(lambda: True, **kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"sample_count": 4}, "sample_count"), ({"confidence": 0.5}, "confidence"), ({"confidence": 1.0}, "confidence")],
)
def test_evaluate_regression_rejects_invalid_arguments(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_regression(lambda: True, lambda: True, **kwargs)  # ty: ignore[invalid-argument-type]


def test_the_assertions_return_their_result_when_nothing_fails() -> None:
    assert assert_pass_rate(lambda: True).passed
    assert not assert_no_regression(lambda: True, lambda: True, sample_count=5, paired=False).regressed


def test_paired_mcnemar_p_value_is_exact_when_candidate_is_not_worse() -> None:
    baseline = iter([True, False, True, True, True]).__next__
    candidate = iter([False, True, True, True, True]).__next__

    res = evaluate_regression(baseline, candidate, sample_count=5, paired=True)

    # One baseline-only win and one candidate-only win: P(X >= 1 | n=2, p=0.5) = 0.75.
    assert res.test_name == "mcnemar"
    assert res.p_value == pytest.approx(0.75)
    assert not res.regressed


def test_assert_no_regression_paired_mcnemar() -> None:
    baseline = itertools.cycle([True, True, False, True, True, False, True, True]).__next__

    with pytest.raises(AssertionError) as exc_info:
        assert_no_regression(
            baseline,
            itertools.cycle([False, False, False, False, True, False, False, False]).__next__,
            confidence=0.95,
            sample_count=20,
            paired=True,
            description="prompt_v2_regression",
        )

    assert "Significant performance regression detected: 'prompt_v2_regression'" in str(exc_info.value)


def test_assert_no_regression_independent_fisher() -> None:
    with pytest.raises(AssertionError) as exc_info:
        assert_no_regression(
            lambda: True,
            lambda: False,
            confidence=0.95,
            sample_count=10,
            paired=False,
            description="independent_trial",
        )

    assert "Significant performance regression detected: 'independent_trial'" in str(exc_info.value)
    assert "fisher_exact" in str(exc_info.value)


@given(
    k=st.integers(min_value=0, max_value=100),
    n=st.integers(min_value=1, max_value=100),
)
def test_wilson_interval_hypothesis_property(k: int, n: int) -> None:
    k = min(k, n)
    low, high = compute_wilson_interval(k, n, confidence=0.95)
    assert 0.0 <= low <= high <= 1.0


def test_unpaired_regression_rejects_non_bool_evaluator_output() -> None:
    def bad() -> bool:
        return 1  # ty: ignore[invalid-return-type]

    with pytest.raises(TypeError, match="bool"):
        evaluate_regression(bad, lambda: True, sample_count=5, paired=False)
    with pytest.raises(TypeError, match="bool"):
        evaluate_regression(bad, lambda: True, sample_count=5, paired=True)


def test_unpaired_regression_is_one_sided() -> None:
    worse = evaluate_regression(
        lambda: True, itertools.cycle([False, False, False, True]).__next__, sample_count=40, paired=False
    )
    assert worse.regressed
    assert worse.p_value < 0.001

    better = evaluate_regression(
        itertools.cycle([False, False, False, True]).__next__, lambda: True, sample_count=40, paired=False
    )
    assert not better.regressed
    assert better.p_value == pytest.approx(1.0)


@given(
    sample_count=st.integers(min_value=5, max_value=40),
    baseline=st.data(),
    confidence=st.floats(min_value=0.6, max_value=0.99),
)
def test_compare_pass_counts_matches_the_unpaired_evaluation(
    sample_count: int, baseline: st.DataObject, confidence: float
) -> None:
    b = baseline.draw(st.lists(st.booleans(), min_size=sample_count, max_size=sample_count))
    c = baseline.draw(st.lists(st.booleans(), min_size=sample_count, max_size=sample_count))

    expected = evaluate_regression(
        iter(b).__next__, iter(c).__next__, confidence=confidence, sample_count=sample_count, paired=False
    )

    assert compare_pass_counts(sum(b), sum(c), sample_count, confidence) == expected


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"sample_count": 4}, "sample_count"), ({"confidence": 0.5}, "confidence")]
)
def test_compare_pass_counts_rejects_invalid_arguments(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compare_pass_counts(3, 3, **{"sample_count": 10, **kwargs})  # ty: ignore[invalid-argument-type]
