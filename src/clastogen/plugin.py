import logging
import tomllib
from collections import Counter
from collections.abc import Generator, Sequence
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
from _pytest.tmpdir import tmppath_result_key

from clastogen.exceptions import TrialError, TrialSkipped
from clastogen.models import (
    BASELINE_RUNS,
    MIN_BASELINE_RATE,
    NOISY_BASELINE_RATE,
    BaselineRecord,
    Behavior,
    ClastogenParams,
    ClastogenPluginState,
    Mutant,
    MutantExecution,
    MutantStatus,
    MutationSummary,
    OracleRecord,
    RecordsPayload,
    SPRTConfig,
)
from clastogen.mutation.injection import override_prompt, resolve_target
from clastogen.mutation.mutator import PromptMutator
from clastogen.reporting.render import STATUS_LABELS, render_html, render_json, render_markdown
from clastogen.reporting.scoring import (
    behavior_label,
    calls_line,
    coverage_line,
    exposure_line,
    operator_line,
    score_line,
    summarize,
    survivor_tests,
)
from clastogen.stats.assertions import drop_upper_bound, evaluate_regression
from clastogen.stats.sprt import SPRT, config_from_baseline, estimate_p0, fixed_sample_size

logger = logging.getLogger(__name__)


STATE_KEY = pytest.StashKey[ClastogenPluginState]()
_RECORDS_KEY = "clastogen_records"
SUPPRESSIONS_PATH = Path(".clastogen") / "suppressions.toml"
_UNTESTED_SHOWN = 10
_BASELINE_SKIPPED = "test skipped during baseline"


def load_suppressions(root_path: Path) -> set[str]:
    """Parses suppressed mutant IDs from <rootdir>/.clastogen/suppressions.toml; every entry needs a reason."""
    candidate = root_path / SUPPRESSIONS_PATH
    if not candidate.is_file():
        return set()
    try:
        raw_data = tomllib.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise pytest.UsageError(f"Cannot read '{candidate}': {exc}") from exc
    items = raw_data.get("suppressions", [])
    if not isinstance(items, list):
        raise pytest.UsageError(f"Invalid suppression format in '{candidate}': expected [[suppressions]] tables.")

    parsed: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            raise pytest.UsageError(
                f"Suppression entry {entry!r} in '{candidate}' must be a table with mutant_id and reason."
            )
        m_id = str(entry.get("mutant_id") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if not m_id:
            raise pytest.UsageError(f"Suppression entry in '{candidate}' missing 'mutant_id': {entry}")
        if not reason:
            raise pytest.UsageError(
                f"Suppression for mutant '{m_id}' in '{candidate}' must specify a non-empty 'reason'."
            )
        parsed.add(m_id)
    return parsed


def _reset_function_fixtures(item: pytest.Item) -> None:
    """Tears down and re-executes all function-scoped fixtures for honest, isolated trials."""
    if not isinstance(item, pytest.Function):
        raise RuntimeError(f"Cannot reset fixtures for '{item.nodeid}': only Python test functions are supported.")
    req = item._request
    fixture_defs = req._fixture_defs
    for name, fixturedef in list(fixture_defs.items()):
        if fixturedef.scope == "function":
            # tmp_path teardown reads this key, which pytest only sets after the call phase finishes.
            item.stash.setdefault(tmppath_result_key, {})
            fixturedef.finish(request=req)  # ty: ignore[invalid-argument-type]  # finish only reads request.node, which TopRequest has
            del fixture_defs[name]
            new_val = req.getfixturevalue(name)
            if name in item.funcargs:
                item.funcargs[name] = new_val


def _run_trial(item: pytest.Item) -> bool:
    """Reruns the test on fresh fixtures: assertion failures are False, anything else unexpected is a TrialError."""
    try:
        _reset_function_fixtures(item)
    except Exception as exc:
        raise TrialError(f"fixture reset failed: {type(exc).__name__}: {exc}") from exc
    try:
        item.runtest()
    except pytest.xfail.Exception as exc:
        raise TrialSkipped from exc
    except (AssertionError, pytest.fail.Exception):
        return False
    except pytest.skip.Exception as exc:
        raise TrialSkipped from exc
    except pytest.exit.Exception:
        raise
    except Exception as exc:
        raise TrialError(f"unexpected execution error: {type(exc).__name__}: {exc}") from exc
    return True


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("clastogen", "LLM Mutation Testing")
    group.addoption(
        "--clastogen",
        action="store_true",
        help="Run Clastogen mutation testing on marked LLM tests.",
    )
    group.addoption(
        "--clastogen-oracle",
        action="store_true",
        help="With --clastogen, run the probe tests (oracle_samples=N) on every mutant to label which ones change behavior.",
    )
    group.addoption(
        "--clastogen-fail-under",
        type=float,
        metavar="MIN_SCORE",
        help="Fail test suite if the Mutation Score is below this threshold (0-100).",
    )
    group.addoption(
        "--clastogen-json",
        metavar="PATH",
        help="Output Clastogen mutation testing results to a structured JSON file.",
    )
    group.addoption(
        "--clastogen-html",
        metavar="PATH",
        help="Output a self-contained HTML report of Clastogen mutation testing results.",
    )
    group.addoption(
        "--clastogen-md",
        metavar="PATH",
        help="Output a Markdown summary for GitHub job summaries and pull request comments.",
    )


def _is_xdist_worker(config: pytest.Config) -> bool:
    return hasattr(config, "workerinput")


class _RecordCollector:
    """Ingests the records a marked test attached to its report, whether it ran in-process or on an xdist worker."""

    def __init__(self, state: ClastogenPluginState) -> None:
        self._state = state

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return
        for key, payload in report.user_properties:
            if key != _RECORDS_KEY:
                continue
            # Only pytest_runtest_call attaches this key, always with a RecordsPayload.
            records = cast(RecordsPayload, payload)
            self._state.marked_tests_count += 1
            self._state.constraints.update((target, rule) for target, rule in records["constraints"])
            self._state.baselines.extend(BaselineRecord(**b) for b in records["baselines"])
            self._state.oracle.extend(OracleRecord(**o) for o in records["oracle"])
            for r in records["results"]:
                fields: dict[str, Any] = {**r, "status": MutantStatus(r["status"])}
                self._state.results.append(MutantExecution(**fields))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "clastogen(target, max_mutants=5, delta=0.30, p0=None, max_steps=20, alpha=0.05, beta=0.10, oracle_samples=None): Run Clastogen on target prompt; oracle_samples=N makes the test a behavior probe.",
    )
    suppressed = load_suppressions(config.rootpath) if config.getoption("--clastogen") else set()
    state = ClastogenPluginState(suppressed_mutants=suppressed)
    config.stash[STATE_KEY] = state
    if not _is_xdist_worker(config):
        config.pluginmanager.register(_RecordCollector(state), "clastogen-record-collector")


def _parse_marker(item: pytest.Item, marker: pytest.Mark) -> tuple[ClastogenParams, str]:
    """Validates the marker arguments and returns them with the current text of the target prompt."""
    try:
        params = ClastogenParams(*marker.args, **marker.kwargs)
    except (TypeError, ValueError) as exc:
        raise pytest.UsageError(f"Invalid @pytest.mark.clastogen arguments on '{item.nodeid}': {exc}") from exc
    if not isinstance(params.target, str) or not params.target:
        raise pytest.UsageError(
            f"Test '{item.nodeid}' marked with @pytest.mark.clastogen must specify a string 'target' argument (e.g. target='module:PROMPT')."
        )

    try:
        owner, final_attr = resolve_target(params.target)
    except Exception as exc:
        raise pytest.UsageError(f"Failed to resolve target '{params.target}' for test '{item.nodeid}': {exc}") from exc

    prompt_text = getattr(owner, final_attr)
    if not isinstance(prompt_text, str):
        raise pytest.UsageError(
            f"Target '{params.target}' in test '{item.nodeid}' must be a string, got {type(prompt_text).__name__}."
        )
    return params, prompt_text


def _measure_baseline(item: pytest.Item, params: ClastogenParams) -> tuple[BaselineRecord, SPRTConfig | None]:
    """Reruns the passing test to estimate p0; the config is None when the baseline is too flaky or broken to mutate."""
    outcomes = [True]
    error: str | None = None
    for _ in range(BASELINE_RUNS - 1):
        try:
            outcomes.append(_run_trial(item))
        except TrialSkipped:
            error = _BASELINE_SKIPPED
            break
        except TrialError as exc:
            logger.error("Baseline trial failed for test '%s': %s", item.nodeid, exc, exc_info=exc)
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
            item.nodeid,
            params.target,
            f": {error}" if error else "",
        )
    record = BaselineRecord(
        test_id=item.nodeid,
        target=params.target,
        successes=sum(outcomes),
        runs=len(outcomes),
        p0=estimate_p0(outcomes),
        stable=config is not None,
        error=error,
    )
    return record, config


def _execution(
    item: pytest.Item,
    mutant: Mutant,
    status: MutantStatus,
    sample_count: int | None = None,
    llr: float | None = None,
    error: str | None = None,
    fixed_n: int | None = None,
) -> MutantExecution:
    return MutantExecution(
        test_id=item.nodeid,
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
        raise TrialError(f"prompt injection failed: {type(exc).__name__}: {exc}") from exc


def _evaluate_mutant(item: pytest.Item, mutant: Mutant, sprt: SPRT) -> MutantExecution:
    """Runs the SPRT with the mutant injected; clastogen-side failures and unexpected test errors become ERROR."""

    try:
        with ExitStack() as stack:
            _inject(stack, mutant)
            res = sprt.run_evaluator(lambda: _run_trial(item))
    except TrialSkipped:
        logger.debug("Mutant [%s] SKIPPED in test '%s'", mutant.id, item.nodeid)
        return _execution(item, mutant, MutantStatus.SKIPPED)
    except TrialError as exc:
        logger.error("Mutant [%s] ERROR in test '%s': %s", mutant.id, item.nodeid, exc, exc_info=exc)
        return _execution(item, mutant, MutantStatus.ERROR, error=str(exc))

    status = MutantStatus(res.decision.value)
    logger.debug(
        "Mutant [%s] %s (%d runs, LLR=%.2f) in test '%s'",
        mutant.id,
        status,
        res.sample_count,
        res.cumulative_llr,
        item.nodeid,
    )
    fixed = fixed_sample_size(sprt.config)
    return _execution(item, mutant, status, res.sample_count, res.cumulative_llr, fixed_n=fixed[0] if fixed else None)


def _run_mutation(
    item: pytest.Item, params: ClastogenParams, prompt_text: str, state: ClastogenPluginState
) -> tuple[list[BaselineRecord], list[MutantExecution], list[list[str]]]:
    """Mutates the target prompt and runs the SPRT on every mutant not yet killed or suppressed.

    Records newly killed mutants in state so later tests skip them; returns (baselines, results, constraints),
    where constraints are the [target, rule] pairs of the prompt whether or not a mutant was made for them.
    """
    mutator = PromptMutator(target_symbol=params.target)
    constraints = [[params.target, rule] for rule in mutator.extract_candidate_rules(prompt_text)]
    mutants = mutator.generate_mutants(prompt_text, max_mutants=params.max_mutants)
    mutants = [m for m in mutants if m.id not in state.killed_mutants]

    results = [_execution(item, m, MutantStatus.SUPPRESSED) for m in mutants if m.id in state.suppressed_mutants]
    pending = [m for m in mutants if m.id not in state.suppressed_mutants]
    if not pending:
        return [], results, constraints

    baselines: list[BaselineRecord] = []
    if params.p0 is None:
        baseline, config = _measure_baseline(item, params)
        baselines.append(baseline)
        # A broken baseline leaves its mutants unmeasured; recording them lets a kill by another test take precedence.
        if baseline.error == _BASELINE_SKIPPED:
            results += [_execution(item, m, MutantStatus.SKIPPED) for m in pending]
        elif baseline.error:
            results += [_execution(item, m, MutantStatus.ERROR, error=f"baseline: {baseline.error}") for m in pending]
        if config is None:
            return baselines, results, constraints
    else:
        config = SPRTConfig.from_absolute_drop(
            p0=params.p0, delta=params.delta, alpha=params.alpha, beta=params.beta, max_steps=params.max_steps
        )

    sprt = SPRT(config)
    for mutant in pending:
        execution = _evaluate_mutant(item, mutant, sprt)
        if execution.status == MutantStatus.KILLED:
            state.killed_mutants.add(mutant.id)
        results.append(execution)
    return baselines, results, constraints


def _run_oracle(
    item: pytest.Item, params: ClastogenParams, samples: int, prompt_text: str, suppressed: set[str]
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
        baseline = [_run_trial(item) for _ in range(samples)]
    except (TrialSkipped, TrialError) as exc:
        logger.warning("Probe '%s' not measured: %s", item.nodeid, exc)
        return []
    if sum(baseline) / samples < MIN_BASELINE_RATE:
        logger.warning("Probe '%s' is flaky (%d/%d passed), not measured.", item.nodeid, sum(baseline), samples)
        return []

    records: list[OracleRecord] = []
    for mutant in mutants:
        try:
            with ExitStack() as stack:
                _inject(stack, mutant)
                candidate = [_run_trial(item) for _ in range(samples)]
        except (TrialSkipped, TrialError) as exc:
            logger.warning("Probe '%s' not measured on mutant [%s]: %s", item.nodeid, mutant.id, exc)
            continue
        res = evaluate_regression(
            iter(baseline).__next__,
            iter(candidate).__next__,
            confidence=1 - params.alpha,
            sample_count=samples,
            paired=False,
        )
        drop = res.baseline_rate - res.candidate_rate
        # The tolerance keeps a drop of exactly delta (e.g. 1.0 - 0.8 vs 0.2) from failing on float rounding.
        label: Behavior = "no_change_found"
        if res.regressed and drop >= params.delta - 1e-9:
            label = "changed"
        elif drop_upper_bound(sum(baseline), sum(candidate), samples, params.alpha) >= params.delta - 1e-9:
            label = "suspected"
        records.append(OracleRecord(mutant.id, label))
    return records


def _to_payload(
    baselines: Sequence[BaselineRecord] = (),
    results: Sequence[MutantExecution] = (),
    constraints: Sequence[list[str]] = (),
    oracle: Sequence[OracleRecord] = (),
) -> RecordsPayload:
    # execnet serializes only exact builtin types, so the StrEnum status becomes a plain str.
    return {
        "baselines": [asdict(b) for b in baselines],
        "results": [{**asdict(r), "status": str(r.status)} for r in results],
        "constraints": list(constraints),
        "oracle": [asdict(o) for o in oracle],
    }


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    """Runs the mutation loop after the real test passed; a failing test propagates from the yield untouched."""
    res = yield

    if not item.config.getoption("--clastogen", False):
        return res

    marker = item.get_closest_marker("clastogen")
    if not marker:
        return res

    params, prompt_text = _parse_marker(item, marker)
    state = item.config.stash[STATE_KEY]
    if (samples := params.oracle_samples) is not None:
        oracle = []
        if item.config.getoption("--clastogen-oracle", False):
            oracle = _run_oracle(item, params, samples, prompt_text, state.suppressed_mutants)
        item.user_properties.append((_RECORDS_KEY, _to_payload(oracle=oracle)))
        return res

    baselines, results, constraints = _run_mutation(item, params, prompt_text, state)
    item.user_properties.append((_RECORDS_KEY, _to_payload(baselines, results, constraints)))
    return res


def _fail_under_failed(summary: MutationSummary, threshold: float) -> bool:
    """A session without measurable mutants fails any positive threshold."""
    if summary.score is None:
        return threshold > 0.0
    return summary.score < threshold


def _write_report(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Computes the mutation summary once, then exports JSON/HTML/Markdown and enforces --clastogen-fail-under from it."""
    if not session.config.getoption("--clastogen", False) or _is_xdist_worker(session.config):
        return

    state = session.config.stash.get(STATE_KEY, None)
    if state is None:
        return
    state.baselines.sort(key=lambda b: b.test_id)
    state.results.sort(key=lambda r: (r.mutant_id, r.test_id))
    summary = summarize(state.results, state.constraints, state.oracle)
    state.summary = summary

    json_path = session.config.getoption("--clastogen-json", None)
    if json_path:
        _write_report(json_path, render_json(summary, state.results, state.baselines))

    html_path = session.config.getoption("--clastogen-html", None)
    if html_path:
        _write_report(html_path, render_html(summary, state.results, state.baselines))

    md_path = session.config.getoption("--clastogen-md", None)
    if md_path:
        _write_report(md_path, render_markdown(summary, state.results, state.baselines))

    fail_under = session.config.getoption("--clastogen-fail-under", None)
    # Errored mutants drop out of the score, so they fail the run instead of inflating it.
    mutant_errors = summary.counts[MutantStatus.ERROR]
    errored = mutant_errors > 0
    fail_under_failed = fail_under is not None and _fail_under_failed(summary, float(fail_under))
    if session.exitstatus == pytest.ExitCode.OK and (fail_under_failed or errored):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    if fail_under_failed and session.exitstatus == pytest.ExitCode.TESTS_FAILED:
        # pytest prints shouldfail in red just above its final "N passed" line; keep a reason it already set (-x).
        detail = "no measurable mutants" if summary.score is None else f"score {summary.score:.1f}%"
        message = f"Clastogen fail-under {float(fail_under):.1f}%: FAILED ({detail})"
    elif errored and session.exitstatus == pytest.ExitCode.TESTS_FAILED:
        message = f"Clastogen: {mutant_errors} mutant error(s)"
    else:
        return
    prev = session.shouldfail
    session.shouldfail = f"{prev}; {message}" if isinstance(prev, str) and prev else message


_STATUS_MARKUP: dict[MutantStatus, dict[str, bool]] = {
    MutantStatus.SURVIVED: {"red": True, "bold": True},
    MutantStatus.INCONCLUSIVE: {"yellow": True},
    MutantStatus.ERROR: {"purple": True},
    MutantStatus.KILLED: {"green": True},
    MutantStatus.SKIPPED: {"light": True},
    MutantStatus.SUPPRESSED: {"light": True},
}


def _write_baselines(terminalreporter: pytest.TerminalReporter, baselines: list[BaselineRecord], verbose: bool) -> None:
    """Writes one line for all stable baselines (one per test with -v) after the full list of flaky ones."""
    stable = [b for b in baselines if b.stable]
    # Errored baselines are reported on their mutants' ERROR and SKIPPED rows.
    flaky = [b for b in baselines if not b.stable and not b.error]
    if flaky:
        title = f"Flaky tests (not mutated: baseline pass rate < {MIN_BASELINE_RATE:.0%})"
        terminalreporter.write_sep("-", title, yellow=True)
        for b in flaky:
            terminalreporter.write_line(
                f"  {b.test_id}: {b.successes}/{b.runs} baseline runs passed -> {b.target}", yellow=True
            )
    else:
        terminalreporter.write_sep("-")
    if noisy := [b for b in stable if b.noisy]:
        terminalreporter.write_line(
            f"Noisy baselines (pass rate < {NOISY_BASELINE_RATE:.0%}): false KILLs can exceed alpha for "
            + ", ".join(b.test_id for b in noisy),
            yellow=True,
        )
    if not stable:
        return
    if verbose:
        for b in stable:
            terminalreporter.write_line(
                f"Baseline {b.test_id}: {b.successes}/{b.runs} runs passed, p0={b.p0:.3f} -> {b.target}", light=True
            )
        return
    rates = sorted({f"{b.successes}/{b.runs}": b.successes / b.runs for b in stable}.items(), key=lambda kv: kv[1])
    passed = rates[0][0] if len(rates) == 1 else f"{rates[0][0]}-{rates[-1][0]}"
    terminalreporter.write_line(
        f"Baselines: {len(stable)} stable tests, {passed} runs passed (-v shows each test)", light=True
    )


def _write_summary(
    terminalreporter: pytest.TerminalReporter,
    state: ClastogenPluginState,
    summary: MutationSummary,
    fail_under: float | None,
) -> None:
    verbose = terminalreporter.config.get_verbosity() > 0
    order = list(STATUS_LABELS)
    show_target = len({r.target for r in summary.results}) > 1
    # A test is named without its file path unless another file has a test with the same name.
    short = {e.test_id: e.test_id.split("::", 1)[-1] for e in state.results}
    clashes = {n for n, c in Counter(short.values()).items() if c > 1}
    name = {t: t if n in clashes else n for t, n in short.items()}
    passed_in = {m: [name[t] for t in tests] for m, tests in survivor_tests(state.results).items()}

    terminalreporter.write_sep("=", "Clastogen Mutation Testing Summary", bold=True)
    for r in sorted(summary.results, key=lambda r: order.index(r.status)):
        details = [f"[{r.mutant_id}]"]
        if show_target:
            details.append(r.target)
        if r.sample_count is not None:
            details.append(f"{r.sample_count} runs")
        if verbose and r.llr is not None:
            details.append(f"LLR={r.llr:+.2f}")
        if r.status == MutantStatus.KILLED:
            details.append(f"by {name[r.test_id]}")
        if r.status == MutantStatus.SURVIVED:
            details.append(f"passed in {', '.join(passed_in[r.mutant_id])}")
        if r.error:
            details.append(f"[{r.error}]")
        if label := behavior_label(summary, r.mutant_id):
            details.append(label)
        terminalreporter.write_line(f"{STATUS_LABELS[r.status]:<14}  {'  '.join(details)}", **_STATUS_MARKUP[r.status])
        terminalreporter.write_line(f"    {r.description}", light=True)

    _write_baselines(terminalreporter, state.baselines, verbose)

    survived = summary.counts[MutantStatus.SURVIVED]
    if survived:
        terminalreporter.write_line(f"{survived} survived: your tests still pass with these prompt faults.", red=True)
        terminalreporter.write_line(
            "    Add an assertion that fails for each one, or suppress its ID in .clastogen/suppressions.toml.",
            light=True,
        )
    inconclusive = summary.counts[MutantStatus.INCONCLUSIVE]
    if inconclusive:
        terminalreporter.write_line(
            f"{inconclusive} inconclusive: no decision after max_steps runs. Increase max_steps or delta.",
            yellow=True,
        )

    # Without --clastogen-fail-under, any measured blind spot (score below 100%) shows red and N/A has no colour.
    if fail_under is None:
        ok = None if summary.score is None else summary.score == 100.0
    else:
        ok = not _fail_under_failed(summary, fail_under)
    terminalreporter.write_line(score_line(summary), bold=True, green=ok is True, red=ok is False)
    if calls := calls_line(summary):
        terminalreporter.write_line(calls, light=True)
    if line := exposure_line(summary):
        terminalreporter.write_line(line, bold=True)
    if line := operator_line(summary):
        terminalreporter.write_line(line, light=True)
    if line := coverage_line(summary):
        terminalreporter.write_line(line, bold=True)
    if summary.untested:
        shown = summary.untested if verbose else summary.untested[:_UNTESTED_SHOWN]
        terminalreporter.write_line(
            "Untested constraints (no measured mutant: raise max_mutants, fix errors or add tests):", yellow=True
        )
        for rule in shown:
            terminalreporter.write_line(f"    {rule if len(rule) <= 100 else rule[:99] + '…'}", light=True)
        if len(shown) < len(summary.untested):
            terminalreporter.write_line(
                f"    … and {len(summary.untested) - len(shown)} more (-v shows all)", light=True
            )


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config) -> None:
    """Renders the terminal summary from the session-wide MutationSummary computed in sessionfinish."""
    if not config.getoption("--clastogen", False):
        return

    state = config.stash.get(STATE_KEY, None)
    summary = state.summary if state else None
    if state is None or summary is None:
        return

    raw_fail_under = config.getoption("--clastogen-fail-under", None)
    fail_under = None if raw_fail_under is None else float(raw_fail_under)
    empty = summary.total == 0 and all(b.stable for b in state.baselines)
    if not empty:
        _write_summary(terminalreporter, state, summary, fail_under)
    elif state.marked_tests_count > 0:
        if state.oracle:
            hint = " Probes labelled mutants, but no suite test measured them."
        elif config.getoption("--clastogen-oracle", False):
            hint = ""
        else:
            hint = " Tests with oracle_samples are probes and run only with --clastogen-oracle."
        terminalreporter.write_line(
            f"Clastogen: 0 mutants generated across marked tests (check prompt constraints).{hint}"
        )
    else:
        terminalreporter.write_line("Clastogen: No @pytest.mark.clastogen tests executed.")
    if not empty:
        terminalreporter.write_sep("=", bold=True)
