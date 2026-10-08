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


@pytest.mark.parametrize(("p0", "delta"), [(1.0, 0.25), (0.5, 0.6)])
def test_sprt_from_absolute_drop_rejects_hypotheses_outside_unit_interval(p0: float, delta: float) -> None:
    with pytest.raises(ValueError, match="0 < p1 < p0 < 1"):
        SPRTConfig.from_absolute_drop(p0=p0, delta=delta)
