from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from clastogen.models import BASELINE_RUNS, MIN_BASELINE_P0, ClastogenParams, Decision
from clastogen.stats.sprt import SPRT, config_from_baseline

REPS = 2000


def _pct_killed(baseline_p: float, mutant_p: float, seed: int = 7) -> float:
    """Drives the baseline -> config_from_baseline -> SPRT decision path against accepted baselines only."""
    rng = random.Random(seed)  # noqa: S311
    killed = accepted = 0
    for _ in range(REPS):
        baseline = [True] + [rng.random() < baseline_p for _ in range(BASELINE_RUNS - 1)]
        config = config_from_baseline(baseline, delta=0.30, max_steps=20)
        if config is None:
            continue
        accepted += 1
        result = SPRT(config).run_evaluator(lambda: rng.random() < mutant_p)
        killed += result.decision is Decision.KILLED
    return killed / accepted * 100.0


def test_flaky_baseline_is_rejected() -> None:
    assert config_from_baseline([True] * 7 + [False] * 3, delta=0.30, max_steps=20) is None


def test_stable_baseline_yields_laplace_p0_config() -> None:
    config = config_from_baseline([True] * 10, delta=0.30, max_steps=15, alpha=0.01, beta=0.02)

    assert config is not None
    assert config.p0 == pytest.approx(11 / 12)
    assert config.p1 == pytest.approx(11 / 12 - 0.30)
    assert (config.max_steps, config.alpha, config.beta) == (15, 0.01, 0.02)


@given(
    delta=st.floats(min_value=0.01, max_value=MIN_BASELINE_P0, exclude_max=True),
    successes=st.integers(min_value=8, max_value=BASELINE_RUNS - 1),
)
def test_any_accepted_baseline_supports_a_valid_delta(delta: float, successes: int) -> None:
    ClastogenParams(target="app:P", delta=delta)
    baseline = [True] * successes + [False] * (BASELINE_RUNS - successes)

    assert config_from_baseline(baseline, delta=delta, max_steps=20) is not None


@pytest.mark.parametrize("delta", [MIN_BASELINE_P0, 0.95])
def test_delta_no_baseline_can_support_is_rejected(delta: float) -> None:
    with pytest.raises(ValueError, match="lowest p0 an accepted baseline can measure"):
        ClastogenParams(target="app:P", delta=delta)


def test_probe_delta_is_a_minimum_drop_not_bounded_by_the_baseline() -> None:
    assert ClastogenParams(target="app:P", delta=0.8, oracle_samples=10).delta == 0.8


@pytest.mark.parametrize("true_p", [0.90, 0.95])
def test_unchanged_mutant_is_rarely_falsely_killed(true_p: float) -> None:
    assert _pct_killed(true_p, true_p) < 5.0


def test_strong_mutant_is_detected() -> None:
    assert _pct_killed(0.95, 0.30) > 95.0
