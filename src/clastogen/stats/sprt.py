import inspect
import logging
import math
from collections.abc import Callable, Sequence
from functools import cache

from clastogen.stats.types import Decision, SPRTConfig, SPRTResult

logger = logging.getLogger(__name__)

MAX_FIXED_N = 2000


def check_outcome(raw: object) -> bool:
    """Returns raw if it is a bool; rejects awaitables and other types with TypeError."""
    if inspect.isawaitable(raw):
        raise TypeError("evaluator returned an awaitable/coroutine; expected synchronous callable returning bool")
    if not isinstance(raw, bool):
        raise TypeError(f"evaluator must return bool, got {type(raw).__name__}")
    return raw


def estimate_p0(outcomes: Sequence[bool]) -> float:
    """Estimates the baseline pass probability with Laplace's rule of succession, (s + 1) / (n + 2)."""
    return (sum(outcomes) + 1) / (len(outcomes) + 2)


def _pmf(n: int, k: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p), computed in log space so large n cannot overflow."""
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(log_comb + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_cdf(n: int, c: int, p: float) -> float:
    """P(X <= c) for X ~ Binomial(n, p)."""
    return sum(_pmf(n, k, p) for k in range(c + 1))


@cache
def fixed_sample_size(config: SPRTConfig) -> tuple[int, int] | None:
    """Smallest (n, c) of a one-sided binomial test with the SPRT's error rates: KILLED if passes <= c.

    Requires P(passes <= c | p0) <= alpha and P(passes > c | p1) <= beta. Only c <= the largest alpha-feasible
    cutoff is allowed and a larger c only helps beta, so each n costs one pass; c is then the smallest feasible.
    Returns None past MAX_FIXED_N, which bounds the O(n^2) scan.
    """
    for n in range(1, MAX_FIXED_N + 1):
        cutoff, cum = None, 0.0
        for c in range(n):
            cum += _pmf(n, c, config.p0)
            if cum > config.alpha:
                break
            cutoff = c
        if cutoff is None:
            continue
        cum = 0.0
        for c in range(cutoff + 1):
            cum += _pmf(n, c, config.p1)
            if 1 - cum <= config.beta:
                return n, c
    return None


class SPRT:
    """Wald's Sequential Probability Ratio Test engine for Bernoulli trials with early stopping."""

    def __init__(self, config: SPRTConfig) -> None:
        self.config = config
        self.threshold_a = math.log((1.0 - config.beta) / config.alpha)
        self.threshold_b = math.log(config.beta / (1.0 - config.alpha))
        self.llr_pass = math.log(config.p1 / config.p0)
        self.llr_fail = math.log((1.0 - config.p1) / (1.0 - config.p0))

    def run_evaluator(self, evaluator: Callable[[], bool]) -> SPRTResult:
        """Executes evaluator dynamically until a decision is reached or max_steps is exhausted."""
        cumulative_llr = 0.0
        decision = Decision.INCONCLUSIVE
        step = 0
        passes = 0
        while step < self.config.max_steps:
            step += 1
            outcome = check_outcome(evaluator())
            passes += outcome
            cumulative_llr += self.llr_pass if outcome else self.llr_fail
            if cumulative_llr >= self.threshold_a:
                decision = Decision.KILLED
                break
            if cumulative_llr <= self.threshold_b:
                decision = Decision.SURVIVED
                break

        logger.debug("SPRT %s after %d step(s) (LLR=%.3f)", decision.value, step, cumulative_llr)
        return SPRTResult(decision=decision, sample_count=step, cumulative_llr=cumulative_llr, passes=passes)
