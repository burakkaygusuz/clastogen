import logging
import tomllib
from collections import Counter
from collections.abc import Generator
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest

from clastogen.models import (
    MIN_BASELINE_RATE,
    NOISY_BASELINE_RATE,
    BaselineRecord,
    ClastogenParams,
    ClastogenPluginState,
    MutantExecution,
    MutantStatus,
    MutationSummary,
    OracleRecord,
    RecordsPayload,
)
from clastogen.mutation.injection import resolve_target
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
from clastogen.runner import run_mutation, run_oracle, to_payload
from clastogen.trial import run_trial

logger = logging.getLogger(__name__)


STATE_KEY = pytest.StashKey[ClastogenPluginState]()
_RECORDS_KEY = "clastogen_records"
SUPPRESSIONS_PATH = Path(".clastogen") / "suppressions.toml"
_UNTESTED_SHOWN = 10


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
            for r in records["executions"]:
                fields: dict[str, Any] = {**r, "status": MutantStatus(r["status"])}
                self._state.executions.append(MutantExecution(**fields))


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
    trial = partial(run_trial, item)
    if (samples := params.oracle_samples) is not None:
        oracle = []
        if item.config.getoption("--clastogen-oracle", False):
            oracle = run_oracle(item.nodeid, trial, params, samples, prompt_text, state.suppressed_mutants)
        item.user_properties.append((_RECORDS_KEY, to_payload(oracle=oracle)))
        return res

    baselines, executions, constraints = run_mutation(item.nodeid, trial, params, prompt_text, state)
    item.user_properties.append((_RECORDS_KEY, to_payload(baselines, executions, constraints)))
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
    state.executions.sort(key=lambda r: (r.mutant_id, r.test_id))
    summary = summarize(state.executions, state.constraints, state.oracle)
    state.summary = summary

    json_path = session.config.getoption("--clastogen-json", None)
    if json_path:
        _write_report(json_path, render_json(summary, state.executions, state.baselines))

    html_path = session.config.getoption("--clastogen-html", None)
    if html_path:
        _write_report(html_path, render_html(summary, state.executions, state.baselines))

    md_path = session.config.getoption("--clastogen-md", None)
    if md_path:
        _write_report(md_path, render_markdown(summary, state.executions, state.baselines))

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
    short = {e.test_id: e.test_id.split("::", 1)[-1] for e in state.executions}
    clashes = {n for n, c in Counter(short.values()).items() if c > 1}
    name = {t: t if n in clashes else n for t, n in short.items()}
    passed_in = {m: [name[t] for t in tests] for m, tests in survivor_tests(state.executions).items()}

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
