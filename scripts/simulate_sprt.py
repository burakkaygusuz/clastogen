from __future__ import annotations

import random  # Seeded PRNG: the simulations must be reproducible, not unpredictable.
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clastogen import SPRT, Decision, SPRTConfig
from clastogen.stats.assertions import compute_wilson_interval, evaluate_pass_rate
from clastogen.stats.sprt import BASELINE_RUNS, MIN_BASELINE_RATE, binom_cdf, config_from_baseline, fixed_sample_size

FLOW_BASELINES = (0.70, 0.80, 0.90, 0.95, 0.99)
FIXED_SIZES = (10, 20, 30, 50, 100)
FAMILY_SIZE = 20


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
    pct_any_killed: float = 0.0


def _tally(
    decisions: list[Decision], steps: list[int], true_p: float, rejected: float, any_killed: float = 0.0
) -> SimulationStats:
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
        pct_any_killed=any_killed,
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


def _baseline_config(rng: random.Random, baseline_p: float, delta: float, max_steps: int) -> SPRTConfig | None:
    """Mimics the plugin baseline: the passing test run plus BASELINE_RUNS - 1 reruns; None when rejected as flaky."""
    baseline = [True] + [rng.random() < baseline_p for _ in range(BASELINE_RUNS - 1)]  # NOSONAR
    return config_from_baseline(baseline, delta=delta, max_steps=max_steps)


def run_full_flow_simulation(
    baseline_p: float,
    mutant_p: float,
    delta: float = 0.30,
    max_steps: int = 20,
    trials: int = 10_000,
    seed: int = 42,
    family_size: int = FAMILY_SIZE,
) -> SimulationStats:
    """Simulates the plugin decision path: baseline runs -> Laplace p0 -> SPRT on the mutant.

    Rates are over baselines that were accepted; rejected (flaky) baselines are reported in pct_rejected.
    pct_any_killed is the share of accepted baselines where at least one of family_size further mutants with the
    same true pass rate is KILLED; those mutants share the baseline p0, so their verdicts are correlated.
    """
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be an integer >= 1")
    if type(family_size) is not int or family_size < 1:
        raise ValueError("family_size must be an integer >= 1")
    if not (0.0 <= baseline_p <= 1.0 and 0.0 <= mutant_p <= 1.0):
        raise ValueError("baseline_p and mutant_p must be in [0, 1]")

    rng = random.Random(seed)
    family_rng = random.Random(seed + 1)  # Separate stream: the single-mutant rates stay independent of family_size.
    decisions: list[Decision] = []
    steps: list[int] = []
    any_killed = 0

    def mutant_evaluator() -> bool:
        return rng.random() < mutant_p  # NOSONAR

    def family_evaluator() -> bool:
        return family_rng.random() < mutant_p  # NOSONAR

    for _ in range(trials):
        config = _baseline_config(rng, baseline_p, delta, max_steps)
        if config is None:
            continue
        sprt = SPRT(config)
        result = sprt.run_evaluator(mutant_evaluator)
        decisions.append(result.decision)
        steps.append(result.sample_count)
        any_killed += any(sprt.run_evaluator(family_evaluator).decision == Decision.KILLED for _ in range(family_size))

    rejected = (trials - len(decisions)) / trials * 100.0
    if not decisions:
        return SimulationStats(mutant_p, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, rejected)
    return _tally(decisions, steps, mutant_p, rejected, any_killed / len(decisions) * 100.0)


def run_fixed_n_flow(
    baseline_p: float, mutant_p: float, n: int, delta: float = 0.30, trials: int = 10_000, seed: int = 42
) -> float:
    """Percent KILLED by a fixed-N exact binomial test at the plugin's estimated p0, over accepted baselines."""
    rng = random.Random(seed)
    killed = accepted = 0
    for _ in range(trials):
        config = _baseline_config(rng, baseline_p, delta, max_steps=1)
        if config is None:
            continue
        accepted += 1
        passes = sum(rng.random() < mutant_p for _ in range(n))  # NOSONAR
        killed += fixed_n_decision(passes, n, config.p0, config.alpha)
    return killed / accepted * 100.0 if accepted else 0.0


def run_pass_rate_simulation(true_p: float, trials: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Returns (pass %, mean samples) of evaluate_pass_rate with its default arguments."""
    rng = random.Random(seed)
    results = [evaluate_pass_rate(lambda: rng.random() < true_p) for _ in range(trials)]  # NOSONAR
    return sum(r.passed for r in results) / trials * 100.0, sum(r.sample_count for r in results) / trials


@cache
def fixed_n_decision(passes: int, n: int, p0: float, alpha: float) -> bool:
    """One-sided exact binomial test: KILLED iff P(X <= passes | n, p0) <= alpha."""
    return binom_cdf(n, passes, p0) <= alpha


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
    lines.extend(["-" * 80, f" Fixed-N test, same alpha/beta: N={n}, KILLED if passes <= {c}"])
    lines.extend(
        f" p={s.true_p:.2f}: SPRT ASN {s.asn:.2f} vs fixed {n}, saves {(1 - s.asn / n) * 100:.1f}%, "
        f"{s.pct_inconclusive:.1f}% INCONCLUSIVE"
        for s in stats_list
        if abs(s.true_p - config.p0) < 1e-4 or abs(s.true_p - config.p1) < 1e-4
    )
    lines.extend(["=" * 80, ""])
    return "\n".join(lines)


def _with_ci(pct: float, n: int) -> str:
    """Formats a simulated percentage with its Wilson 95% interval."""
    if n == 0:
        return "n/a"
    low, high = compute_wilson_interval(round(pct * n / 100), n)
    return f"{pct:.2f} [{low * 100:.2f}, {high * 100:.2f}]"


def format_full_flow_report(
    title: str, rows: list[tuple[float, SimulationStats]], family_size: int = FAMILY_SIZE
) -> str:
    lines: list[str] = [
        "=" * 100,
        f" {title} ",
        f" Baseline: {BASELINE_RUNS} runs, Laplace p0 = (s+1)/(n+2), rejected if raw rate < {MIN_BASELINE_RATE:.2f}",
        " KILLED and SURVIVED show Wilson 95% intervals over accepted baselines",
        "-" * 100,
        f"{'Base P':<7} | {'Mutant P':<8} | {'REJECTED %':<10} | {'KILLED %':<22} | {'SURVIVED %':<22} | "
        f"{'INCONCL. %':<10} | {'ASN'}",
        "-" * 100,
    ]
    for baseline_p, s in rows:
        lines.append(
            f"{baseline_p:<7.2f} | {s.true_p:<8.2f} | {s.pct_rejected:<10.2f} | {_with_ci(s.pct_killed, s.trials):<22} | "
            f"{_with_ci(s.pct_survived, s.trials):<22} | {s.pct_inconclusive:<10.2f} | {s.asn:.2f}"
        )
    lines.append("-" * 100)
    if unchanged := [s for baseline_p, s in rows if s.true_p == baseline_p]:
        lines.append(
            f" Unchanged mutants: worst false-kill rate {max(s.pct_killed for s in unchanged):.2f}%; "
            f"worst P(>= 1 false kill among {family_size} sharing one baseline) = "
            f"{max(s.pct_any_killed for s in unchanged):.1f}%"
        )
    lines.extend(["=" * 100, ""])
    return "\n".join(lines)


def format_fixed_n_report(title: str, rows: list[tuple[float, float, dict[int, float]]]) -> str:
    sizes = list(rows[0][2])
    lines: list[str] = [
        "=" * 80,
        f" {title} ",
        "-" * 80,
        f"{'Base P':<7} | {'Mutant P':<8} | " + " | ".join(f"{f'N={n}':<7}" for n in sizes),
        "-" * 80,
    ]
    lines.extend(
        f"{baseline_p:<7.2f} | {mutant_p:<8.2f} | " + " | ".join(f"{killed[n]:<7.2f}" for n in sizes)
        for baseline_p, mutant_p, killed in rows
    )
    lines.extend(["=" * 80, ""])
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

    pairs = [(base, mutant) for base in FLOW_BASELINES for mutant in (base, round(base - 0.30, 2))]
    pairs += [(0.95, 0.30), (0.95, 0.0)]
    flow_rows = [(base, run_full_flow_simulation(base, mutant, trials=trials)) for base, mutant in pairs]
    print(
        format_full_flow_report(
            "Scenario 3: Full plugin flow (10-run Laplace baseline -> delta=0.30 SPRT, max_steps=20)", flow_rows
        )
    )
    fixed_rows = [
        (base, mutant, {n: run_fixed_n_flow(base, mutant, n, trials=trials) for n in FIXED_SIZES})
        for base, mutant in pairs
    ]
    print(
        format_fixed_n_report(
            "Scenario 3b: same flow, fixed-N exact binomial test at alpha=0.05 (KILLED %)", fixed_rows
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
