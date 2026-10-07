from __future__ import annotations

import math
import random  # Seeded PRNG: the simulations must be reproducible, not unpredictable.
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clastogen import SPRT, Decision, SPRTConfig
from clastogen.core.sprt import BASELINE_RUNS, MIN_BASELINE_RATE, config_from_baseline
from clastogen.stats import evaluate_pass_rate


@dataclass
class SimulationStats:
    true_p: float
    trials: int
    pct_killed: float
    pct_survived: float
    pct_inconclusive: float
    asn: float
    min_steps: int
    max_steps: int
    pct_rejected: float = 0.0


def _tally(decisions: list[Decision], steps: list[int], true_p: float, rejected: float) -> SimulationStats:
    n = len(decisions)
    return SimulationStats(
        true_p=true_p,
        trials=n,
        pct_killed=decisions.count(Decision.KILLED) / n * 100.0,
        pct_survived=decisions.count(Decision.SURVIVED) / n * 100.0,
        pct_inconclusive=decisions.count(Decision.INCONCLUSIVE) / n * 100.0,
        asn=sum(steps) / n,
        min_steps=min(steps),
        max_steps=max(steps),
        pct_rejected=rejected,
    )


def run_simulation(
    config: SPRTConfig,
    true_p: float,
    trials: int = 10_000,
    seed: int = 42,
) -> SimulationStats:
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be an integer >= 1")
    if not 0.0 <= true_p <= 1.0:
        raise ValueError("true_p must be in [0, 1]")

    rng = random.Random(seed)
    sprt = SPRT(config)

    def evaluator() -> bool:
        return rng.random() < true_p  # NOSONAR

    results = [sprt.run_evaluator(evaluator) for _ in range(trials)]
    return _tally([r.decision for r in results], [r.sample_count for r in results], true_p, 0.0)


def run_full_flow_simulation(
    baseline_p: float,
    mutant_p: float,
    delta: float = 0.30,
    max_steps: int = 20,
    trials: int = 10_000,
    seed: int = 42,
) -> SimulationStats:
    """Simulates the plugin decision path: baseline runs -> Laplace p0 -> SPRT on the mutant.

    Rates are over baselines that were accepted; rejected (flaky) baselines are reported in pct_rejected.
    """
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be an integer >= 1")
    if not (0.0 <= baseline_p <= 1.0 and 0.0 <= mutant_p <= 1.0):
        raise ValueError("baseline_p and mutant_p must be in [0, 1]")

    rng = random.Random(seed)
    decisions: list[Decision] = []
    steps: list[int] = []

    def mutant_evaluator() -> bool:
        return rng.random() < mutant_p  # NOSONAR

    for _ in range(trials):
        baseline = [True] + [rng.random() < baseline_p for _ in range(BASELINE_RUNS - 1)]  # NOSONAR
        config = config_from_baseline(baseline, delta=delta, max_steps=max_steps)
        if config is None:
            continue
        result = SPRT(config).run_evaluator(mutant_evaluator)
        decisions.append(result.decision)
        steps.append(result.sample_count)

    rejected = (trials - len(decisions)) / trials * 100.0
    if not decisions:
        return SimulationStats(mutant_p, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, rejected)
    return _tally(decisions, steps, mutant_p, rejected)


def run_pass_rate_simulation(true_p: float, trials: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Returns (pass %, mean samples) of evaluate_pass_rate with its default arguments."""
    rng = random.Random(seed)
    results = [evaluate_pass_rate(lambda: rng.random() < true_p) for _ in range(trials)]  # NOSONAR
    return sum(r.passed for r in results) / trials * 100.0, sum(r.sample_count for r in results) / trials


def fixed_sample_size(config: SPRTConfig) -> tuple[int, int]:
    """Smallest (n, c) of a one-sided binomial test with the SPRT's error rates: KILLED if passes <= c.

    Requires P(passes <= c | p0) <= alpha and P(passes > c | p1) <= beta.
    """

    def cdf(n: int, c: int, p: float) -> float:
        return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(c + 1))

    n = 1
    while True:
        for c in range(n + 1):
            if cdf(n, c, config.p0) <= config.alpha and 1 - cdf(n, c, config.p1) <= config.beta:
                return n, c
        n += 1


def format_simulation_report(title: str, config: SPRTConfig, stats_list: list[SimulationStats]) -> str:
    sprt = SPRT(config)
    lines: list[str] = [
        "=" * 80,
        f" {title} ",
        f" Config: alpha={config.alpha}, beta={config.beta}, p0={config.p0:.2f}, p1={config.p1:.2f}, max_steps={config.max_steps}",
        f" Wald Thresholds: A (Killed) = +{sprt.threshold_a:.3f}, B (Survived) = {sprt.threshold_b:.3f}",
        "-" * 80,
        f"{'True P':<8} | {'KILLED %':<10} | {'SURVIVED %':<11} | {'INCONCLUSIVE %':<14} | {'ASN (Mean)':<10} | {'[Min, Max]'}",
        "-" * 80,
    ]
    for s in stats_list:
        note = ""
        if abs(s.true_p - config.p0) < 1e-4:
            note = f" (H0: False Kill = {s.pct_killed:.2f}%, nominal <= {config.alpha * 100:.1f}%)"
        elif abs(s.true_p - config.p1) < 1e-4:
            note = f" (H1: False Survive = {s.pct_survived:.2f}%, nominal <= {config.beta * 100:.1f}%)"

        lines.append(
            f"{s.true_p:<8.2f} | {s.pct_killed:<10.2f} | {s.pct_survived:<11.2f} | "
            f"{s.pct_inconclusive:<14.2f} | {s.asn:<10.2f} | [{s.min_steps}, {s.max_steps}]{note}"
        )
    n, c = fixed_sample_size(config)
    lines.append("-" * 80)
    lines.append(f" Fixed-N test, same alpha/beta: N={n}, KILLED if passes <= {c}")
    lines.extend(
        f" p={s.true_p:.2f}: SPRT ASN {s.asn:.2f} vs fixed {n}, saves {(1 - s.asn / n) * 100:.1f}%, "
        f"{s.pct_inconclusive:.1f}% INCONCLUSIVE"
        for s in stats_list
        if abs(s.true_p - config.p0) < 1e-4 or abs(s.true_p - config.p1) < 1e-4
    )
    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


def format_full_flow_report(title: str, rows: list[tuple[float, SimulationStats]]) -> str:
    lines: list[str] = [
        "=" * 80,
        f" {title} ",
        f" Baseline: {BASELINE_RUNS} runs, Laplace p0 = (s+1)/(n+2), rejected if raw rate < {MIN_BASELINE_RATE:.2f}",
        "-" * 80,
        f"{'Base P':<7} | {'Mutant P':<8} | {'REJECTED %':<10} | {'KILLED %':<8} | {'SURVIVED %':<10} | "
        f"{'INCONCLUSIVE %':<14} | {'ASN (Mean)'}",
        "-" * 80,
    ]
    for baseline_p, s in rows:
        lines.append(
            f"{baseline_p:<7.2f} | {s.true_p:<8.2f} | {s.pct_rejected:<10.2f} | {s.pct_killed:<8.2f} | "
            f"{s.pct_survived:<10.2f} | {s.pct_inconclusive:<14.2f} | {s.asn:.2f}"
        )
    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


def print_simulation_report(title: str, config: SPRTConfig, stats_list: list[SimulationStats]) -> None:
    print(format_simulation_report(title, config, stats_list))


def main() -> None:
    trials = 10_000

    config_1 = SPRTConfig(alpha=0.05, beta=0.10, p0=0.90, p1=0.60, max_steps=20)
    test_probs_1 = [0.95, 0.90, 0.85, 0.75, 0.60, 0.40, 0.10, 0.0]
    stats_1 = [run_simulation(config_1, p, trials=trials) for p in test_probs_1]
    print_simulation_report(
        "Scenario 1: Standard LLM Eval (p0=0.90, p1=0.60, max_steps=20)",
        config_1,
        stats_1,
    )

    config_2 = SPRTConfig.from_absolute_drop(p0=0.75, delta=0.25, alpha=0.05, beta=0.10, max_steps=25)
    test_probs_2 = [0.85, 0.75, 0.65, 0.50, 0.35, 0.10, 0.0]
    stats_2 = [run_simulation(config_2, p, trials=trials) for p in test_probs_2]
    print_simulation_report(
        "Scenario 2: Flaky Baseline Eval (p0=0.75, delta=0.25 -> p1=0.50, max_steps=25)",
        config_2,
        stats_2,
    )

    flow_rows = [
        (base, run_full_flow_simulation(base, mutant, trials=trials))
        for base, mutant in [(0.95, 0.95), (0.90, 0.90), (0.85, 0.85), (0.95, 0.60), (0.95, 0.30), (0.95, 0.0)]
    ]
    print(
        format_full_flow_report(
            "Scenario 3: Full plugin flow (10-run Laplace baseline -> delta=0.30 SPRT, max_steps=20)", flow_rows
        )
    )

    print("=" * 80)
    print(" Scenario 4: assert_pass_rate defaults (min_rate=0.90, tolerance=0.20, confidence=0.95, max_samples=50)")
    print("-" * 80)
    print(f"{'True P':<8} | {'PASSED %':<10} | {'ASN (Mean)'}")
    print("-" * 80)
    for true_p in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60]:
        pct_passed, asn = run_pass_rate_simulation(true_p, trials=trials)
        print(f"{true_p:<8.2f} | {pct_passed:<10.2f} | {asn:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
