from __future__ import annotations

import random

import pytest

from clastogen.stats.sprt import BASELINE_RUNS, SPRT, config_from_baseline
from clastogen.types import Decision

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
    config = config_from_baseline([True] * 10, delta=0.30, max_steps=15)

    assert config is not None
    assert config.p0 == pytest.approx(11 / 12)
    assert config.p1 == pytest.approx(11 / 12 - 0.30)
    assert config.max_steps == 15


def test_alpha_beta_reach_config() -> None:
    config = config_from_baseline([True] * 10, delta=0.30, max_steps=15, alpha=0.01, beta=0.02)

    assert config is not None
    assert (config.alpha, config.beta) == (0.01, 0.02)


@pytest.mark.parametrize("true_p", [0.90, 0.95])
def test_unchanged_mutant_is_rarely_falsely_killed(true_p: float) -> None:
    assert _pct_killed(true_p, true_p) < 5.0


def test_strong_mutant_is_detected() -> None:
    assert _pct_killed(0.95, 0.30) > 95.0
