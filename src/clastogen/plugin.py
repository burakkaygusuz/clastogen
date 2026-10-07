import json
import logging
import tomllib
from collections import Counter
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
from _pytest.tmpdir import tmppath_result_key

from clastogen.exceptions import TrialError, TrialSkipped
from clastogen.models import (
    BaselineRecord,
    ClastogenParams,
    ClastogenPluginState,
    Mutant,
    MutantExecution,
    MutationSummary,
    SPRTConfig,
)
from clastogen.mutation.injection import override_prompt, resolve_target
from clastogen.mutation.mutator import PromptMutator
from clastogen.reporting.render import STATUS_LABELS, render_html, render_markdown
from clastogen.reporting.scoring import score_line, summarize
from clastogen.stats.sprt import BASELINE_RUNS, MIN_BASELINE_RATE, SPRT, config_from_baseline, estimate_p0
from clastogen.types import MutantStatus, RecordsPayload

logger = logging.getLogger(__name__)


STATE_KEY = pytest.StashKey[ClastogenPluginState]()
_RECORDS_KEY = "clastogen_records"
SUPPRESSIONS_PATH = Path(".clastogen") / "suppressions.toml"


def load_suppressions(root_path: Path) -> set[str]:
    """Parses suppressed mutant IDs from <rootdir>/.clastogen/suppressions.toml; every entry needs a reason."""
    candidate = root_path / SUPPRESSIONS_PATH
    if not candidate.is_file():
        return set()
    try:
        raw_data = tomllib.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise pytest.UsageError(f"Cannot read '{candidate}': {exc}") from exc
    items = raw_data.get("suppressions")
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
            fixturedef.finish(request=req)  # type: ignore[arg-type]  # finish only reads request.node, which TopRequest has
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
        default=False,
        help="Run Clastogen mutation testing on marked LLM tests.",
    )
    group.addoption(
        "--clastogen-fail-under",
        action="store",
        type=float,
        default=None,
        metavar="MIN_SCORE",
        help="Fail test suite if the Mutation Score is below this threshold (0-100).",
    )
    group.addoption(
        "--clastogen-json",
        action="store",
        type=str,
        default=None,
        metavar="PATH",
        help="Output Clastogen mutation testing results to a structured JSON file.",
    )
    group.addoption(
        "--clastogen-html",
        action="store",
        type=str,
        default=None,
        metavar="PATH",
        help="Output a self-contained HTML report of Clastogen mutation testing results.",
    )
    group.addoption(
        "--clastogen-md",
        action="store",
        type=str,
        default=None,
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
            self._state.baselines.extend(BaselineRecord(**b) for b in records["baselines"])
            self._state.results.extend(
                MutantExecution(
                    test_id=r["test_id"],
                    target=r["target"],
                    mutant_id=r["mutant_id"],
                    description=r["description"],
                    status=MutantStatus(r["status"]),
                    sample_count=r["sample_count"],
                    llr=r["llr"],
                    error=r["error"],
                )
                for r in records["results"]
            )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "clastogen(target, max_mutants=5, delta=0.30, p0=None, max_steps=20, alpha=0.05, beta=0.10): Run Clastogen on target prompt.",
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
            error = "test skipped during baseline"
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
) -> MutantExecution:
    return MutantExecution(
        test_id=item.nodeid,
        target=mutant.target_symbol,
        mutant_id=mutant.id,
        description=mutant.description,
        status=status,
        sample_count=sample_count,
        llr=llr,
        error=error,
    )


def _evaluate_mutant(item: pytest.Item, mutant: Mutant, sprt: SPRT) -> MutantExecution:
    """Runs the SPRT with the mutant injected; clastogen-side failures and unexpected test errors become ERROR."""

    def evaluator() -> bool:
        with ExitStack() as stack:
            try:
                stack.enter_context(override_prompt(mutant.target_symbol, mutant.mutated_prompt))
            except Exception as exc:
                raise TrialError(f"prompt injection failed: {type(exc).__name__}: {exc}") from exc
            return _run_trial(item)

    try:
        res = sprt.run_evaluator(evaluator)
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
    return _execution(item, mutant, status, res.sample_count, res.cumulative_llr)


def _run_mutation(
    item: pytest.Item, marker: pytest.Mark, state: ClastogenPluginState
) -> tuple[list[BaselineRecord], list[MutantExecution]]:
    """Mutates the target prompt and runs the SPRT on every mutant not yet killed or suppressed.

    Records newly killed mutants in state so later tests skip them; returns (baselines, results).
    """
    params, prompt_text = _parse_marker(item, marker)
    mutants = PromptMutator(target_symbol=params.target).generate_mutants(prompt_text, max_mutants=params.max_mutants)
    mutants = [m for m in mutants if m.id not in state.killed_mutants]

    results = [_execution(item, m, MutantStatus.SUPPRESSED) for m in mutants if m.id in state.suppressed_mutants]
    pending = [m for m in mutants if m.id not in state.suppressed_mutants]
    if not pending:
        return [], results

    baselines: list[BaselineRecord] = []
    if params.p0 is None:
        baseline, config = _measure_baseline(item, params)
        baselines.append(baseline)
        if config is None:
            return baselines, results
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
    return baselines, results


def _to_payload(baselines: list[BaselineRecord], results: list[MutantExecution]) -> RecordsPayload:
    # execnet serializes only exact builtin types, so the StrEnum status becomes a plain str.
    return {
        "baselines": [asdict(b) for b in baselines],
        "results": [{**asdict(r), "status": str(r.status)} for r in results],
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

    baselines, results = _run_mutation(item, marker, item.config.stash[STATE_KEY])
    item.user_properties.append((_RECORDS_KEY, _to_payload(baselines, results)))
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
    summary = summarize(state.results)
    state.summary = summary

    json_path = session.config.getoption("--clastogen-json", None)
    if json_path:
        out_dict = {
            "mutation_score": summary.score,
            "total_mutants": summary.total,
            "counts": {status.value: n for status, n in summary.counts.items()},
            "results": [asdict(r) for r in summary.results],
            "executions": [asdict(r) for r in state.results],
            "baselines": [asdict(b) for b in state.baselines],
        }
        _write_report(json_path, json.dumps(out_dict, indent=2))

    html_path = session.config.getoption("--clastogen-html", None)
    if html_path:
        _write_report(html_path, render_html(summary, state.results, state.baselines))

    md_path = session.config.getoption("--clastogen-md", None)
    if md_path:
        _write_report(md_path, render_markdown(summary, state.results, state.baselines))

    fail_under = session.config.getoption("--clastogen-fail-under", None)
    all_errors = summary.total > 0 and summary.counts[MutantStatus.ERROR] == summary.total
    fail_under_failed = fail_under is not None and _fail_under_failed(summary, float(fail_under))
    if session.exitstatus == pytest.ExitCode.OK and (fail_under_failed or all_errors):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    if fail_under_failed and session.exitstatus == pytest.ExitCode.TESTS_FAILED:
        # pytest prints shouldfail in red just above its final "N passed" line; keep a reason it already set (-x).
        detail = "no measurable mutants" if summary.score is None else f"score {summary.score:.1f}%"
        message = f"Clastogen fail-under {float(fail_under):.1f}%: FAILED ({detail})"
    elif all_errors and session.exitstatus == pytest.ExitCode.TESTS_FAILED:
        message = f"Clastogen: all {summary.total} mutants errored"
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
    flaky = [b for b in baselines if not b.stable]
    if flaky:
        title = f"Flaky tests (not mutated: baseline pass rate < {MIN_BASELINE_RATE:.0%})"
        terminalreporter.write_sep("-", title, yellow=True)
        for b in flaky:
            reason = f" [{b.error}]" if b.error else ""
            terminalreporter.write_line(
                f"  {b.test_id}: {b.successes}/{b.runs} baseline runs passed{reason} -> {b.target}", yellow=True
            )
    else:
        terminalreporter.write_sep("-")
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
    passed_in: dict[str, list[str]] = {}
    for e in state.results:
        if e.status == MutantStatus.SURVIVED:
            passed_in.setdefault(e.mutant_id, []).append(name[e.test_id])

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
        terminalreporter.write_line("Clastogen: 0 mutants generated across marked tests (check prompt constraints).")
    else:
        terminalreporter.write_line("Clastogen: No @pytest.mark.clastogen tests executed.")
    if not empty:
        terminalreporter.write_sep("=", bold=True)
