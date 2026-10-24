import pytest

from scripts.simulate_sprt import (
    SimulationStats,
    fixed_n_decision,
    format_full_flow_report,
    run_fixed_n_flow,
    run_full_flow_simulation,
    run_test_family_simulation,
)


def test_fixed_n_kills_at_or_below_critical_value_only() -> None:
    # Binomial(10, 0.9): P(X <= 6) = 0.013 and P(X <= 7) = 0.070.
    assert fixed_n_decision(6, 10, 0.9, 0.05)
    assert not fixed_n_decision(7, 10, 0.9, 0.05)


def test_full_flow_is_seeded_and_rates_sum_to_one_hundred() -> None:
    first = run_full_flow_simulation(0.9, 0.9, trials=500)
    assert first == run_full_flow_simulation(0.9, 0.9, trials=500)
    assert abs(first.pct_killed + first.pct_survived + first.pct_inconclusive - 100.0) < 1e-9


def test_flaky_baselines_are_rejected_before_mutation() -> None:
    assert run_full_flow_simulation(0.5, 0.5, trials=500).pct_rejected > 90.0
    assert run_fixed_n_flow(0.95, 0.0, 10, trials=200) == 100.0


def test_family_false_kill_rate_is_bounded_by_single_and_full_flow() -> None:
    single = run_full_flow_simulation(0.9, 0.9, trials=500, family_size=1)
    family = run_full_flow_simulation(0.9, 0.9, trials=500, family_size=20)
    assert family.pct_killed == single.pct_killed
    assert family.pct_killed <= family.pct_any_killed <= 100.0
    assert family.pct_any_killed >= single.pct_any_killed


def test_full_flow_report_without_unchanged_rows_omits_the_false_kill_line() -> None:
    row = (0.9, SimulationStats(0.6, 10, 50.0, 40.0, 10.0, 8.0, 3, 20))
    assert "false-kill" not in format_full_flow_report("title", [row])


def test_full_flow_rejects_empty_family() -> None:
    with pytest.raises(ValueError, match="family_size"):
        run_full_flow_simulation(0.9, 0.9, trials=10, family_size=0)


@pytest.mark.parametrize(("baseline_p", "ceiling"), [(0.90, 4.0), (0.85, 12.0), (0.80, 22.0)])
def test_alpha_split_keeps_false_kills_of_many_tests_near_alpha(baseline_p: float, ceiling: float) -> None:
    # Measured at T=12 (10,000 trials): 22%, 41%, 54% without the split and 2.4%, 8.9%, 17.4% with it. The noisy
    # baselines stay above 5% because p0 is estimated from 10 runs; the split only removes the T-fold growth.
    split = run_test_family_simulation(baseline_p, 12, trials=1500, alpha=0.05 / 12)
    unsplit = run_test_family_simulation(baseline_p, 12, trials=1500)
    assert split <= ceiling
    assert unsplit > 2 * split


def test_alpha_split_costs_power_at_a_drop_of_exactly_delta() -> None:
    kills = [
        run_full_flow_simulation(0.90, 0.60, trials=1500, family_size=1, alpha=0.05 / t).pct_killed for t in (1, 12, 50)
    ]
    assert kills[0] > kills[1] > kills[2]
