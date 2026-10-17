import logging
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict

from clastogen.exceptions import TrialError, TrialSkipped
from clastogen.models import (
    BASELINE_RUNS,
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
    SPRTConfig,
)
from clastogen.mutation.injection import override_prompt
from clastogen.mutation.mutator import PromptMutator
from clastogen.stats.assertions import compare_pass_counts, drop_upper_bound
from clastogen.stats.sprt import SPRT, config_from_baseline, estimate_p0, fixed_sample_size

logger = logging.getLogger(__name__)

_BASELINE_SKIPPED = "test skipped during baseline"


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
            logger.error("Baseline trial failed for test '%s': %s", test_id, exc)
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


def _inject(stack: ExitStack, mutant: Mutant) -> None:
    """Patches the mutated prompt in for the life of stack; an injection failure becomes a TrialError."""
    try:
        stack.enter_context(override_prompt(mutant.target_symbol, mutant.mutated_prompt))
    except Exception as exc:
        raise TrialError(f"prompt injection failed: {type(exc).__name__}") from exc


def _evaluate_mutant(test_id: str, trial: Callable[[], bool], mutant: Mutant, sprt: SPRT) -> MutantExecution:
    """Runs the SPRT with the mutant injected; clastogen-side failures and unexpected test errors become ERROR."""

    try:
        with ExitStack() as stack:
            _inject(stack, mutant)
            res = sprt.run_evaluator(trial)
    except TrialSkipped:
        logger.debug("Mutant [%s] SKIPPED in test '%s'", mutant.id, test_id)
        return _execution(test_id, mutant, MutantStatus.SKIPPED)
    except TrialError as exc:
        logger.error("Mutant [%s] ERROR in test '%s': %s", mutant.id, test_id, exc)
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


def run_mutation(
    test_id: str, trial: Callable[[], bool], params: ClastogenParams, prompt_text: str, state: ClastogenPluginState
) -> tuple[list[BaselineRecord], list[MutantExecution], list[list[str]]]:
    """Mutates the target prompt and runs the SPRT on every mutant not yet killed or suppressed.

    Records newly killed mutants in state so later tests skip them; returns (baselines, executions, constraints),
    where constraints are the [target, rule] pairs of the prompt whether or not a mutant was made for them.
    """
    mutator = PromptMutator(target_symbol=params.target)
    constraints = [[params.target, rule] for rule in mutator.extract_candidate_rules(prompt_text)]
    mutants = mutator.generate_mutants(prompt_text, max_mutants=params.max_mutants)
    mutants = [m for m in mutants if m.id not in state.killed_mutants]

    executions = [_execution(test_id, m, MutantStatus.SUPPRESSED) for m in mutants if m.id in state.suppressed_mutants]
    pending = [m for m in mutants if m.id not in state.suppressed_mutants]
    if not pending:
        return [], executions, constraints

    baselines: list[BaselineRecord] = []
    if params.p0 is None:
        baseline, config = measure_baseline(test_id, trial, params)
        baselines.append(baseline)
        # A broken baseline leaves its mutants unmeasured; recording them lets a kill by another test take precedence.
        if baseline.error == _BASELINE_SKIPPED:
            executions += [_execution(test_id, m, MutantStatus.SKIPPED) for m in pending]
        elif baseline.error:
            executions += [
                _execution(test_id, m, MutantStatus.ERROR, error=f"baseline: {baseline.error}") for m in pending
            ]
        if config is None:
            return baselines, executions, constraints
    else:
        config = SPRTConfig.from_absolute_drop(
            p0=params.p0, delta=params.delta, alpha=params.alpha, beta=params.beta, max_steps=params.max_steps
        )

    sprt = SPRT(config)
    for mutant in pending:
        execution = _evaluate_mutant(test_id, trial, mutant, sprt)
        if execution.status == MutantStatus.KILLED:
            state.killed_mutants.add(mutant.id)
        executions.append(execution)
    return baselines, executions, constraints


def run_oracle(
    test_id: str,
    trial: Callable[[], bool],
    params: ClastogenParams,
    samples: int,
    prompt_text: str,
    suppressed: set[str],
) -> list[OracleRecord]:
    """Runs a probe on the original prompt and on every unsuppressed mutant; a mutant it cannot measure gets no record.

    Mutants another test already killed are labelled too, and every probe is judged alone at its own alpha, so adding
    a probe never lowers a label. A drop the test is too weak to call is suspected, not unchanged.
    """
    mutants = PromptMutator(target_symbol=params.target).generate_mutants(prompt_text, params.max_mutants)
    mutants = [m for m in mutants if m.id not in suppressed]
    if not mutants:
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
        try:
            with ExitStack() as stack:
                _inject(stack, mutant)
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
    oracle: Sequence[OracleRecord] = (),
) -> RecordsPayload:
    # execnet serializes only exact builtin types, so the StrEnum status becomes a plain str.
    return {
        "baselines": [asdict(b) for b in baselines],
        "executions": [{**asdict(r), "status": str(r.status)} for r in executions],
        "constraints": list(constraints),
        "oracle": [asdict(o) for o in oracle],
    }
