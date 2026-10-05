from __future__ import annotations

import math
from statistics import NormalDist


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
