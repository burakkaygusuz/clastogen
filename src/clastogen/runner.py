import logging
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, replace

from clastogen.exceptions import TrialError, TrialSkipped
from clastogen.models import (
    BASELINE_RUNS,
    CANARY_RUNS,
    MIN_BASELINE_RATE,
    BaselineRecord,
    Behavior,
    ClastogenParams,
    ClastogenPluginState,
    Mutant,
    MutantExecution,
    MutantStatus,
    OracleRecord,
    RecordsPayload,
    RunPlan,
    SPRTConfig,
    TrialBudget,
)
from clastogen.mutation.http_hook import rewrite_requests
from clastogen.mutation.injection import override_prompt, resolve_target
from clastogen.mutation.mutator import PromptMutator
from clastogen.stats.assertions import compare_pass_counts, drop_upper_bound
from clastogen.stats.sprt import SPRT, config_from_baseline, estimate_p0, fixed_sample_size

logger = logging.getLogger(__name__)

_BASELINE_SKIPPED = "test skipped during baseline"
_BASELINE_UNSTABLE = "baseline: unstable"
_CANARY_PASSED = "canary: test passes with an empty prompt; read the prompt at call time, or assert on behavior that the prompt causes"
_BUDGET_EXHAUSTED = "max-calls budget exhausted"


def measure_baseline(
    test_id: str, trial: Callable[[], bool], params: ClastogenParams
) -> tuple[BaselineRecord, SPRTConfig | None]:
    """Reruns the passing test to estimate p0; the config is None when the baseline is too flaky or broken to mutate."""
    outcomes = [True]
    error: str | None = None
    for _ in range(BASELINE_RUNS - 1):
        try:
            outcomes.append(trial())
        except TrialSkipped:
            error = _BASELINE_SKIPPED
            break
        except TrialError as exc:
            logger.exception("Baseline trial failed for test '%s'", test_id)
            error = str(exc)
            break

    config = (
        None
        if error
        else config_from_baseline(
            outcomes, delta=params.delta, max_steps=params.max_steps, alpha=params.alpha, beta=params.beta
        )
    )
    if config is None:
        logger.warning(
            "Baseline rejected (%d/%d passed, minimum %.0f%%) for test '%s' targeting '%s'%s.",
            sum(outcomes),
            len(outcomes),
            MIN_BASELINE_RATE * 100,
            test_id,
            params.target,
            f": {error}" if error else "",
        )
    record = BaselineRecord(
        test_id=test_id,
        target=params.target,
        successes=sum(outcomes),
        runs=len(outcomes),
        p0=estimate_p0(outcomes),
        stable=config is not None,
        error=error,
    )
    return record, config


def _execution(
    test_id: str,
    mutant: Mutant,
    status: MutantStatus,
    sample_count: int | None = None,
    llr: float | None = None,
    error: str | None = None,
    fixed_n: int | None = None,
) -> MutantExecution:
    return MutantExecution(
        test_id=test_id,
        target=mutant.target_symbol,
        mutant_id=mutant.id,
        description=mutant.description,
        operator_name=mutant.operator_name,
        original_snippet=mutant.original_snippet,
        mutated_snippet=mutant.mutated_snippet,
        status=status,
        sample_count=sample_count,
        llr=llr,
        error=error,
        fixed_n=fixed_n,
    )


def _inject(stack: ExitStack, target: str, prompt: str) -> None:
    """Patches the prompt in for the life of stack, as an attribute and in httpx requests; a failure becomes a TrialError."""
    try:
        stack.enter_context(rewrite_requests(getattr(*resolve_target(target)), prompt))
        stack.enter_context(override_prompt(target, prompt))
    except Exception as exc:
        raise TrialError(f"prompt injection failed: {type(exc).__name__}") from exc


def _evaluate_mutant(test_id: str, trial: Callable[[], bool], mutant: Mutant, sprt: SPRT) -> MutantExecution:
    """Runs the SPRT with the mutant injected; clastogen-side failures and unexpected test errors become ERROR."""

    try:
        with ExitStack() as stack:
            _inject(stack, mutant.target_symbol, mutant.mutated_prompt)
            res = sprt.run_evaluator(trial)
    except TrialSkipped:
        logger.debug("Mutant [%s] SKIPPED in test '%s'", mutant.id, test_id)
        return _execution(test_id, mutant, MutantStatus.SKIPPED)
    except TrialError as exc:
        logger.exception("Mutant [%s] ERROR in test '%s'", mutant.id, test_id)
        return _execution(test_id, mutant, MutantStatus.ERROR, error=str(exc))

    status = MutantStatus(res.decision.value)
    logger.debug(
        "Mutant [%s] %s (%d runs, LLR=%.2f) in test '%s'",
        mutant.id,
        status,
        res.sample_count,
        res.cumulative_llr,
        test_id,
    )
    fixed = fixed_sample_size(sprt.config)
    return _execution(
        test_id, mutant, status, res.sample_count, res.cumulative_llr, fixed_n=fixed[0] if fixed else None
    )


def plan_test(
    test_id: str, params: ClastogenParams, prompt_text: str, suppressed: set[str], with_oracle: bool
) -> RunPlan:
    """Worst case of a marked test, from the mutants run_mutation or run_oracle would generate, without running them.

    A normal test runs the 9 extra baseline trials only with p0 unset, the canary trials and up to max_steps trials per
    mutant, all only with a mutant to test. A probe runs only with with_oracle, and then samples trials on the original
    prompt and on each mutant. The original test call is not counted, and a kill by an earlier test can only lower
    these numbers.
    """
    mutator = PromptMutator(target_symbol=params.target)
    mutants = mutator.generate_mutants(prompt_text, max_mutants=params.max_mutants)
    suppressed_count = sum(m.id in suppressed for m in mutants)
    pending = len(mutants) - suppressed_count
    if params.oracle_samples is not None:
        if not with_oracle or not pending:
            return RunPlan(test_id, params.target, (), 0, 0, 0)
        return RunPlan(test_id, params.target, (), pending, suppressed_count, params.oracle_samples * (1 + pending))
    rules = tuple(mutator.extract_candidate_rules(prompt_text))
    calls = _baseline_calls(params) + CANARY_RUNS + pending * params.max_steps if pending else 0
    return RunPlan(test_id, params.target, rules, pending, suppressed_count, calls)


def _baseline_calls(params: ClastogenParams) -> int:
    return BASELINE_RUNS - 1 if params.p0 is None else 0


def _canary_passes(trial: Callable[[], bool], target: str) -> bool:
    """True when the test passes every run on an empty prompt; a crash or skip there proves nothing and is not a pass."""
    try:
        with ExitStack() as stack:
            _inject(stack, target, "")
            return all(trial() for _ in range(CANARY_RUNS))
    except (TrialSkipped, TrialError):
        return False


def _apply_canary(
    trial: Callable[[], bool], params: ClastogenParams, budget: TrialBudget, evaluated: list[MutantExecution]
) -> list[MutantExecution]:
    """Turns the survivors of a test that killed nothing into ERROR when it also passes on an empty prompt.

    A kill already proves the test depends on the prompt, so the canary runs only without one.
    """
    unproven = (MutantStatus.SURVIVED, MutantStatus.INCONCLUSIVE)
    if (
        any(e.status == MutantStatus.KILLED for e in evaluated)
        or not any(e.status in unproven for e in evaluated)
        or not budget.fits(CANARY_RUNS)
        or not _canary_passes(trial, params.target)
    ):
        return evaluated
    return [
        replace(e, status=MutantStatus.ERROR, error=_CANARY_PASSED, sample_count=None, llr=None, fixed_n=None)
        if e.status in unproven
        else e
        for e in evaluated
    ]


def _baseline_rejected(test_id: str, mutants: Sequence[Mutant], error: str | None) -> list[MutantExecution]:
    """Records the mutants a broken or flaky baseline left unmeasured, so a kill by another test takes precedence."""
    if error == _BASELINE_SKIPPED:
        status, reason = MutantStatus.SKIPPED, None
    elif error:
        status, reason = MutantStatus.ERROR, f"baseline: {error}"
    else:
        status, reason = MutantStatus.SKIPPED, _BASELINE_UNSTABLE
    return [_execution(test_id, m, status, error=reason) for m in mutants]


def run_mutation(
    test_id: str, trial: Callable[[], bool], params: ClastogenParams, prompt_text: str, state: ClastogenPluginState
) -> tuple[list[BaselineRecord], list[MutantExecution], list[list[str]], list[list[str]]]:
    """Mutates the target prompt and runs the SPRT on every mutant not yet killed or suppressed.

    Records newly killed mutants in state so later tests skip them; returns (baselines, executions, constraints, unmutated),
    where constraints are the [target, rule] pairs of the prompt whether or not a mutant was made for them and
    unmutated are the [target, sentence] pairs that rule discovery skipped.
    """
    # Bonferroni: an unchanged mutant faces every test of the target, and any one kill counts, so each SPRT gets alpha / T.
    params = replace(params, alpha=params.alpha / state.tests.get(params.target, 1))
    mutator = PromptMutator(target_symbol=params.target)
    constraints = [[params.target, rule] for rule in mutator.extract_candidate_rules(prompt_text)]
    unmutated = [[params.target, s] for s in mutator.extract_unmatched_sentences(prompt_text)]
    mutants = mutator.generate_mutants(prompt_text, max_mutants=params.max_mutants)
    mutants = [m for m in mutants if m.id not in state.killed_mutants]

    executions = [_execution(test_id, m, MutantStatus.SUPPRESSED) for m in mutants if m.id in state.suppressed_mutants]
    pending = [m for m in mutants if m.id not in state.suppressed_mutants]
    if not pending:
        return [], executions, constraints, unmutated

    # The test starts only when at least one mutant can be funded after it.
    if not state.budget.fits(_baseline_calls(params) + params.max_steps):
        unmeasured = [_execution(test_id, m, MutantStatus.UNMEASURED, error=_BUDGET_EXHAUSTED) for m in pending]
        return [], executions + unmeasured, constraints, unmutated

    baselines: list[BaselineRecord] = []
    if params.p0 is None:
        baseline, config = measure_baseline(test_id, trial, params)
        baselines.append(baseline)
        if config is None:
            return baselines, executions + _baseline_rejected(test_id, pending, baseline.error), constraints, unmutated
    else:
        config = SPRTConfig.from_absolute_drop(
            p0=params.p0, delta=params.delta, alpha=params.alpha, beta=params.beta, max_steps=params.max_steps
        )
    sprt = SPRT(config)
    evaluated: list[MutantExecution] = []
    for mutant in pending:
        if not state.budget.fits(params.max_steps):
            evaluated.append(_execution(test_id, mutant, MutantStatus.UNMEASURED, error=_BUDGET_EXHAUSTED))
            continue
        execution = _evaluate_mutant(test_id, trial, mutant, sprt)
        if execution.status == MutantStatus.KILLED:
            state.killed_mutants.add(mutant.id)
        evaluated.append(execution)
    return baselines, executions + _apply_canary(trial, params, state.budget, evaluated), constraints, unmutated


def run_oracle(
    test_id: str,
    trial: Callable[[], bool],
    params: ClastogenParams,
    samples: int,
    prompt_text: str,
    suppressed: set[str],
    budget: TrialBudget,
) -> list[OracleRecord]:
    """Runs a probe on the original prompt and on every unsuppressed mutant; a mutant it cannot measure gets no record.

    Mutants another test already killed are labelled too, and every probe is judged alone at its own alpha, so adding
    a probe never lowers a label. A drop the test is too weak to call is suspected, not unchanged.
    """
    mutants = PromptMutator(target_symbol=params.target).generate_mutants(prompt_text, params.max_mutants)
    mutants = [m for m in mutants if m.id not in suppressed]
    if not mutants:
        return []
    if not budget.fits(2 * samples):
        logger.warning("Probe '%s' not measured: %s", test_id, _BUDGET_EXHAUSTED)
        return []
    try:
        baseline = [trial() for _ in range(samples)]
    except (TrialSkipped, TrialError) as exc:
        logger.warning("Probe '%s' not measured: %s", test_id, exc)
        return []
    if sum(baseline) / samples < MIN_BASELINE_RATE:
        logger.warning("Probe '%s' is flaky (%d/%d passed), not measured.", test_id, sum(baseline), samples)
        return []

    records: list[OracleRecord] = []
    for mutant in mutants:
        if not budget.fits(samples):
            logger.warning("Probe '%s' not measured on mutant [%s]: %s", test_id, mutant.id, _BUDGET_EXHAUSTED)
            continue
        try:
            with ExitStack() as stack:
                _inject(stack, mutant.target_symbol, mutant.mutated_prompt)
                candidate = [trial() for _ in range(samples)]
        except (TrialSkipped, TrialError) as exc:
            logger.warning("Probe '%s' not measured on mutant [%s]: %s", test_id, mutant.id, exc)
            continue
        res = compare_pass_counts(sum(baseline), sum(candidate), samples, confidence=1 - params.alpha)
        drop = res.baseline_rate - res.candidate_rate
        # The tolerance keeps a drop of exactly delta (e.g. 1.0 - 0.8 vs 0.2) from failing on float rounding.
        label: Behavior = "no_change_found"
        if res.regressed and drop >= params.delta - 1e-9:
            label = "changed"
        elif drop_upper_bound(sum(baseline), sum(candidate), samples, params.alpha) >= params.delta - 1e-9:
            label = "suspected"
        records.append(OracleRecord(mutant.id, label))
    return records


def to_payload(
    baselines: Sequence[BaselineRecord] = (),
    executions: Sequence[MutantExecution] = (),
    constraints: Sequence[list[str]] = (),
    unmutated: Sequence[list[str]] = (),
    oracle: Sequence[OracleRecord] = (),
    tests: dict[str, int] | None = None,
) -> RecordsPayload:
    # execnet serializes only exact builtin types, so the StrEnum status becomes a plain str.
    return {
        "baselines": [asdict(b) for b in baselines],
        "executions": [{**asdict(r), "status": str(r.status)} for r in executions],
        "constraints": list(constraints),
        "unmutated": list(unmutated),
        "oracle": [asdict(o) for o in oracle],
        "tests": tests or {},
    }
