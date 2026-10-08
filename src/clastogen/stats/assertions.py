from __future__ import annotations

import math
from collections.abc import Callable
from statistics import NormalDist

from clastogen.models import PassRateResult, RegressionResult, SPRTConfig
from clastogen.stats.sprt import SPRT, check_outcome
from clastogen.types import Decision


def compute_wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Computes the Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successful Bernoulli trials (0 <= successes <= total).
        total: Total number of trials (total >= 1).
        confidence: Desired two-sided confidence level (0.0 < confidence < 1.0).

    Returns:
        A tuple of (lower_bound, upper_bound) strictly clamped within [0.0, 1.0].
    """
    if total < 1:
        raise ValueError(f"Total trials must be at least 1, got {total}")
    if not (0 <= successes <= total):
        raise ValueError(f"Successes ({successes}) must be between 0 and total ({total})")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"Confidence must be strictly between 0 and 1, got {confidence}")

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == total else min(1.0, centre + half)
    return low, high


def _binom_upper_tail(wins: int, n: int) -> float:
    """Exact P(X >= wins) for X ~ Binomial(n, 0.5)."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (1 << n)


def _fisher_upper_tail(a: int, c: int, n: int) -> float:
    """Exact one-sided Fisher p-value P(X >= a) for two groups of n with a and c passes (fixed margins)."""
    total_passes = a + c
    return sum(
        math.comb(n, x) * math.comb(n, total_passes - x) for x in range(a, min(n, total_passes) + 1)
    ) / math.comb(2 * n, total_passes)


def _validate_pass_rate_inputs(min_rate: float, tolerance: float, confidence: float, max_samples: int) -> None:
    if not (0.0 < min_rate < 1.0):
        raise ValueError(f"min_rate must be in (0, 1), got {min_rate}")
    if not (0.0 < tolerance < 1.0):
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    if not (0.5 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence}")
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")


def evaluate_pass_rate(
    evaluator: Callable[[], bool],
    min_rate: float = 0.90,
    tolerance: float = 0.20,
    confidence: float = 0.95,
    max_samples: int = 50,
    description: str | None = None,
) -> PassRateResult:
    """Decides with Wald's SPRT whether evaluator meets min_rate, sampling until a decision or max_samples.

    The test is H0: rate = min_rate against H1: rate = min_rate - tolerance, with both error rates set to
    1 - confidence: a true rate of min_rate or higher passes with probability >= confidence, and a true rate of
    min_rate - tolerance or lower fails with probability >= confidence. Rates in between form the indifference
    zone where either outcome is acceptable. If max_samples runs out first, the sign of the log-likelihood ratio
    decides, which keeps both error rates close to nominal when max_samples is large enough (see docs/statistics.md).

    Args:
        evaluator: Zero-argument callable returning True (pass) or False (fail).
        min_rate: Pass probability that must be accepted, in (0.0, 1.0).
        tolerance: Width of the indifference zone below min_rate, in (0.0, 1.0).
        confidence: Probability of the correct decision at both ends of the indifference zone, in (0.5, 1.0).
        max_samples: Maximum number of evaluator calls.
        description: Optional label for diagnostics.
    """
    _validate_pass_rate_inputs(min_rate, tolerance, confidence, max_samples)
    error_rate = 1.0 - confidence
    sprt = SPRT(
        SPRTConfig.from_absolute_drop(
            p0=min_rate, delta=tolerance, alpha=error_rate, beta=error_rate, max_steps=max_samples
        )
    )

    samples: list[bool] = []

    def collect_sample() -> bool:
        outcome = check_outcome(evaluator())
        samples.append(outcome)
        return outcome

    res = sprt.run_evaluator(collect_sample)
    # alpha == beta puts the midpoint between the Wald thresholds at 0.
    passed = res.decision == Decision.SURVIVED or (res.decision == Decision.INCONCLUSIVE and res.cumulative_llr < 0.0)

    n = len(samples)
    success_count = sum(samples)
    ci_low, ci_high = compute_wilson_interval(success_count, n, confidence)
    return PassRateResult(
        passed=passed,
        observed_rate=success_count / n,
        sample_count=n,
        success_count=success_count,
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence=confidence,
        decided=res.decision != Decision.INCONCLUSIVE,
        description=description,
    )


def assert_pass_rate(
    evaluator: Callable[[], bool],
    min_rate: float = 0.90,
    tolerance: float = 0.20,
    confidence: float = 0.95,
    max_samples: int = 50,
    description: str | None = None,
) -> PassRateResult:
    """Asserts evaluate_pass_rate's decision; raises AssertionError with statistical diagnostics on failure."""
    res = evaluate_pass_rate(
        evaluator=evaluator,
        min_rate=min_rate,
        tolerance=tolerance,
        confidence=confidence,
        max_samples=max_samples,
        description=description,
    )

    if not res.passed:
        desc_str = f": '{res.description}'" if res.description else ""
        msg = (
            f"Statistical pass rate assertion failed{desc_str}\n"
            f"  Observed Rate   : {res.observed_rate * 100:.1f}% ({res.success_count} passes / {res.sample_count} samples)\n"
            f"  Required Min    : {min_rate * 100:.1f}% (indifference zone down to {max(min_rate - tolerance, 0.0) * 100:.1f}%)\n"
            f"  Wilson {res.confidence * 100:.0f}% CI   : [{res.ci_lower * 100:.1f}%, {res.ci_upper * 100:.1f}%]\n"
            f"  SPRT Decided    : {res.decided}"
        )
        raise AssertionError(msg)

    return res


def evaluate_regression(
    baseline_fn: Callable[[], bool],
    candidate_fn: Callable[[], bool],
    confidence: float = 0.95,
    sample_count: int = 30,
    paired: bool = True,
    description: str | None = None,
) -> RegressionResult:
    """Tests, one-sided, whether candidate_fn has a lower pass rate than baseline_fn.

    Paired mode runs both callables once per trial and applies a one-sided exact McNemar test to the discordant
    pairs; it is only valid when both callables evaluate the same input within a trial (e.g. the same prompt,
    case or seed). Unpaired mode applies a one-sided Fisher exact test to the two pass/fail tables.
    """
    if sample_count < 5:
        raise ValueError(f"sample_count must be >= 5, got {sample_count}")
    if not (0.5 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence}")

    alpha = 1.0 - confidence

    if paired:
        b_wins = 0
        c_wins = 0
        b_passes = 0
        c_passes = 0

        for _ in range(sample_count):
            b_out = check_outcome(baseline_fn())
            c_out = check_outcome(candidate_fn())
            b_passes += b_out
            c_passes += c_out

            if b_out and not c_out:
                b_wins += 1
            elif not b_out and c_out:
                c_wins += 1

        discordant = b_wins + c_wins
        p_val = _binom_upper_tail(b_wins, discordant) if discordant else 1.0
        test_name = "mcnemar"
    else:
        b_passes = sum(check_outcome(baseline_fn()) for _ in range(sample_count))
        c_passes = sum(check_outcome(candidate_fn()) for _ in range(sample_count))

        # Candidate worse than baseline <=> baseline passes more often.
        p_val = _fisher_upper_tail(b_passes, c_passes, sample_count)
        test_name = "fisher_exact"

    return RegressionResult(
        regressed=p_val < alpha,
        p_value=p_val,
        test_name=test_name,
        baseline_rate=b_passes / sample_count,
        candidate_rate=c_passes / sample_count,
        sample_count=sample_count,
        description=description,
    )


def assert_no_regression(
    baseline_fn: Callable[[], bool],
    candidate_fn: Callable[[], bool],
    confidence: float = 0.95,
    sample_count: int = 30,
    paired: bool = True,
    description: str | None = None,
) -> RegressionResult:
    """Asserts that candidate_fn does not have a significantly lower pass rate than baseline_fn.

    Raises AssertionError if the one-sided regression test is significant at the given confidence.
    Improvements and statistically insignificant drops pass.
    """
    res = evaluate_regression(
        baseline_fn=baseline_fn,
        candidate_fn=candidate_fn,
        confidence=confidence,
        sample_count=sample_count,
        paired=paired,
        description=description,
    )

    if res.regressed:
        desc_str = f": '{res.description}'" if res.description else ""
        msg = (
            f"Significant performance regression detected{desc_str}\n"
            f"  Baseline Rate   : {res.baseline_rate * 100:.1f}%\n"
            f"  Candidate Rate  : {res.candidate_rate * 100:.1f}%\n"
            f"  Test Method     : {res.test_name}\n"
            f"  P-Value         : p = {res.p_value:.4f} (< alpha={1.0 - confidence:.4f})\n"
            f"  Sample Count    : {res.sample_count}"
        )
        raise AssertionError(msg)

    return res
