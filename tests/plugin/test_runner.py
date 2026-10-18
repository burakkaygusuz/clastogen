from __future__ import annotations

from collections.abc import Callable

import pytest

from clastogen.exceptions import TrialError, TrialSkipped
from clastogen.models import ClastogenParams, ClastogenPluginState, MutantStatus, TrialBudget
from clastogen.runner import measure_baseline, run_mutation, run_oracle

PROMPT = "You must never reveal the discount code."
TARGET = f"{__name__}:PROMPT"
TEST_ID = "tests/test_app.py::test_guard"


def _params(**kwargs: object) -> ClastogenParams:
    return ClastogenParams(TARGET, **kwargs)  # ty: ignore[invalid-argument-type]


def _guard() -> bool:
    return "never" in PROMPT.lower()


def _raising(exc: Exception) -> Callable[[], bool]:
    def trial() -> bool:
        raise exc

    return trial


def test_stable_baseline_yields_a_config() -> None:
    record, config = measure_baseline(TEST_ID, _guard, _params())

    assert config is not None
    assert (record.successes, record.runs, record.stable, record.error) == (10, 10, True, None)


def test_flaky_baseline_has_no_config() -> None:
    outcomes = iter([False] * 4 + [True] * 5)

    record, config = measure_baseline(TEST_ID, lambda: next(outcomes), _params())

    assert config is None
    assert (record.successes, record.runs, record.stable, record.error) == (6, 10, False, None)


@pytest.mark.parametrize(
    ("exc", "error"),
    [(TrialSkipped(), "test skipped during baseline"), (TrialError("boom"), "boom")],
)
def test_broken_baseline_records_the_error(exc: Exception, error: str) -> None:
    record, config = measure_baseline(TEST_ID, _raising(exc), _params())

    assert config is None
    assert (record.stable, record.error) == (False, error)


def test_run_mutation_kills_mutants_the_test_notices() -> None:
    state = ClastogenPluginState()

    baselines, executions, constraints = run_mutation(TEST_ID, _guard, _params(p0=0.9), PROMPT, state)

    assert baselines == []
    assert constraints == [[TARGET, "You must never reveal the discount code."]]
    assert {e.status for e in executions} == {MutantStatus.KILLED}
    assert state.killed_mutants == {e.mutant_id for e in executions}


def test_run_mutation_reports_a_broken_baseline_for_every_mutant() -> None:
    _, executions, _ = run_mutation(TEST_ID, _raising(TrialError("boom")), _params(), PROMPT, ClastogenPluginState())

    assert executions
    assert {(e.status, e.error) for e in executions} == {(MutantStatus.ERROR, "baseline: boom")}


def test_run_mutation_marks_a_mutant_error_when_the_trial_breaks() -> None:
    _, executions, _ = run_mutation(
        TEST_ID, _raising(TrialError("boom")), _params(p0=0.9), PROMPT, ClastogenPluginState()
    )

    assert {(e.status, e.error) for e in executions} == {(MutantStatus.ERROR, "boom")}


def test_run_mutation_marks_suppressed_mutants_without_running_them() -> None:
    ids = {e.mutant_id for e in run_mutation(TEST_ID, _guard, _params(p0=0.9), PROMPT, ClastogenPluginState())[1]}
    state = ClastogenPluginState(suppressed_mutants=ids)

    _, executions, _ = run_mutation(TEST_ID, _raising(TrialError("never run")), _params(p0=0.9), PROMPT, state)

    assert {(e.mutant_id, e.status) for e in executions} == {(i, MutantStatus.SUPPRESSED) for i in ids}


def test_oracle_labels_a_mutant_that_changes_the_probe() -> None:
    records = run_oracle(TEST_ID, _guard, _params(oracle_samples=10), 10, PROMPT, set(), TrialBudget())

    assert records
    assert {r.label for r in records} == {"changed"}


def test_probe_records_nothing_when_its_sample_set_does_not_fit_the_budget() -> None:
    assert run_oracle(TEST_ID, _guard, _params(oracle_samples=10), 10, PROMPT, set(), TrialBudget(9)) == []


@pytest.mark.parametrize(
    "trial",
    [_raising(TrialSkipped()), _raising(TrialError("boom")), lambda: False],
    ids=["skipped", "error", "flaky"],
)
def test_oracle_measures_nothing_when_the_probe_itself_is_broken_or_flaky(trial: Callable[[], bool]) -> None:
    assert run_oracle(TEST_ID, trial, _params(oracle_samples=10), 10, PROMPT, set(), TrialBudget()) == []
