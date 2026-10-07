import inspect
import logging
import math
from collections.abc import Callable, Sequence

from clastogen.models import SPRTConfig, SPRTResult
from clastogen.types import Decision

logger = logging.getLogger(__name__)

# Unmutated runs used to estimate p0; the Laplace estimate with n=10 keeps false KILLs on unchanged mutants low.
BASELINE_RUNS = 10
MIN_BASELINE_RATE = 0.80


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


def config_from_baseline(
    outcomes: Sequence[bool], *, delta: float, max_steps: int, alpha: float = 0.05, beta: float = 0.10
) -> SPRTConfig | None:
    """Builds the SPRT config from baseline outcomes, or None when the raw pass rate is below MIN_BASELINE_RATE."""
    if sum(outcomes) / len(outcomes) < MIN_BASELINE_RATE:
        return None
    return SPRTConfig.from_absolute_drop(
        p0=estimate_p0(outcomes), delta=delta, alpha=alpha, beta=beta, max_steps=max_steps
    )


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
        while step < self.config.max_steps:
            step += 1
            cumulative_llr += self.llr_pass if check_outcome(evaluator()) else self.llr_fail
            if cumulative_llr >= self.threshold_a:
                decision = Decision.KILLED
                break
            if cumulative_llr <= self.threshold_b:
                decision = Decision.SURVIVED
                break

        logger.debug("SPRT %s after %d step(s) (LLR=%.3f)", decision.value, step, cumulative_llr)
        return SPRTResult(decision=decision, sample_count=step, cumulative_llr=cumulative_llr)
