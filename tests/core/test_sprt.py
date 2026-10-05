import math

import pytest

from clastogen import SPRT, Decision, SPRTConfig


def test_wald_thresholds_and_llr_increments_exact() -> None:
    sprt = SPRT(SPRTConfig(alpha=0.05, beta=0.10, p0=0.90, p1=0.60))

    assert sprt.threshold_a == pytest.approx(math.log(0.90 / 0.05))
    assert sprt.threshold_b == pytest.approx(math.log(0.10 / 0.95))
    assert sprt.llr_pass == pytest.approx(math.log(2.0 / 3.0))
    assert sprt.llr_fail == pytest.approx(math.log(4.0))
    assert sprt.threshold_a > 0
    assert sprt.threshold_b < 0


def test_sprt_three_consecutive_fails_kills_at_step_three() -> None:
    config = SPRTConfig(alpha=0.05, beta=0.10, p0=0.90, p1=0.60, max_steps=20)
    sprt = SPRT(config)

    observations = [False, False, False]
    result = sprt.run_evaluator(iter(observations).__next__)

    assert result.decision == Decision.KILLED
    assert result.sample_count == 3
    assert result.cumulative_llr >= sprt.threshold_a


def test_sprt_clean_passes_survives_at_step_six() -> None:
    config = SPRTConfig(alpha=0.05, beta=0.10, p0=0.90, p1=0.60, max_steps=20)
    sprt = SPRT(config)

    clean_passes = [True] * 10
    result = sprt.run_evaluator(iter(clean_passes).__next__)

    assert result.decision == Decision.SURVIVED
    assert result.sample_count == 6
    assert result.cumulative_llr <= sprt.threshold_b


def test_sprt_inconclusive_when_truncated() -> None:
    config = SPRTConfig(alpha=0.05, beta=0.10, p0=0.90, p1=0.60, max_steps=6)
    sprt = SPRT(config)

    oscillating = [False, True, True, True, False, True]
    result = sprt.run_evaluator(iter(oscillating).__next__)

    assert result.decision == Decision.INCONCLUSIVE
    assert result.sample_count == 6


def test_sprt_from_absolute_drop_handles_100_percent_pass_without_zerodivision() -> None:
    config = SPRTConfig.from_absolute_drop(p0=1.0, delta=0.25)
    assert config.p0 <= 0.99
    assert config.p1 < config.p0

    res = SPRT(config).run_evaluator(iter([False]).__next__)
    assert res.decision == Decision.KILLED
