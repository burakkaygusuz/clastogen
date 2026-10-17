import math
from dataclasses import replace

import pytest
from pytest_check import check

from clastogen import SPRT, Decision, SPRTConfig
from clastogen.stats.sprt import binom_cdf, fixed_sample_size

CONFIG = SPRTConfig(alpha=0.05, beta=0.10, p0=0.90, p1=0.60, max_steps=20)


def test_wald_thresholds_and_llr_increments_exact() -> None:
    sprt = SPRT(CONFIG)

    check.almost_equal(sprt.threshold_a, math.log(0.90 / 0.05))
    check.almost_equal(sprt.threshold_b, math.log(0.10 / 0.95))
    check.almost_equal(sprt.llr_pass, math.log(2.0 / 3.0))
    check.almost_equal(sprt.llr_fail, math.log(4.0))


def test_sprt_three_consecutive_fails_kills_at_step_three() -> None:
    sprt = SPRT(CONFIG)

    observations = [False, False, False]
    result = sprt.run_evaluator(iter(observations).__next__)

    check.equal(result.decision, Decision.KILLED)
    check.equal(result.sample_count, 3)
    check.greater_equal(result.cumulative_llr, sprt.threshold_a)


def test_sprt_clean_passes_survives_at_step_six() -> None:
    sprt = SPRT(CONFIG)

    clean_passes = [True] * 10
    result = sprt.run_evaluator(iter(clean_passes).__next__)

    check.equal(result.decision, Decision.SURVIVED)
    check.equal(result.sample_count, 6)
    check.less_equal(result.cumulative_llr, sprt.threshold_b)


def test_sprt_inconclusive_when_truncated() -> None:
    sprt = SPRT(replace(CONFIG, max_steps=6))

    oscillating = [False, True, True, True, False, True]
    result = sprt.run_evaluator(iter(oscillating).__next__)

    assert result.decision == Decision.INCONCLUSIVE
    assert result.sample_count == 6


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SPRTConfig(alpha=0.05, beta=0.10, p0=0.90, p1=0.60), (18, 13)),
        (SPRTConfig(alpha=0.05, beta=0.10, p0=0.75, p1=0.50), (33, 20)),
    ],
)
def test_fixed_sample_size_is_the_smallest_test_with_the_sprt_error_rates(
    config: SPRTConfig, expected: tuple[int, int]
) -> None:
    found = fixed_sample_size(config)
    assert found is not None
    n, c = found

    check.equal((n, c), expected)
    check.less_equal(binom_cdf(n, c, config.p0), config.alpha)
    check.less_equal(1 - binom_cdf(n, c, config.p1), config.beta)
    # One call fewer has no cutoff that satisfies both error rates.
    check.is_false(
        any(
            binom_cdf(n - 1, k, config.p0) <= config.alpha and 1 - binom_cdf(n - 1, k, config.p1) <= config.beta
            for k in range(n - 1)
        )
    )


@pytest.mark.parametrize(("p0", "delta"), [(1.0, 0.25), (0.5, 0.6)])
def test_sprt_from_absolute_drop_rejects_hypotheses_outside_unit_interval(p0: float, delta: float) -> None:
    with pytest.raises(ValueError, match="0 < p1 < p0 < 1"):
        SPRTConfig.from_absolute_drop(p0=p0, delta=delta)


def test_fixed_sample_size_survives_a_small_delta_that_overflowed_binomial_coefficients() -> None:
    config = SPRTConfig.from_absolute_drop(p0=11 / 12, delta=0.02, alpha=0.05, beta=0.10, max_steps=20)

    found = fixed_sample_size(config)
    assert found is not None
    n, c = found
    assert n > 1030  # math.comb(n, c) * float overflows from here
    assert binom_cdf(n, c, config.p0) <= config.alpha
    assert 1 - binom_cdf(n, c, config.p1) <= config.beta
    assert binom_cdf(n, c - 1, config.p0) <= config.alpha
    assert 1 - binom_cdf(n, c - 1, config.p1) > config.beta


def test_fixed_sample_size_is_none_beyond_the_sample_cap() -> None:
    config = SPRTConfig.from_absolute_drop(p0=11 / 12, delta=0.01, alpha=0.05, beta=0.10, max_steps=20)

    assert fixed_sample_size(config) is None
