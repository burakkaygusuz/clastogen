import logging
import os
from collections.abc import Generator
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest

from clastogen.custom_mutants import load_custom_mutants
from clastogen.models import (
    BaselineRecord,
    ClastogenParams,
    ClastogenPluginState,
    MutantExecution,
    MutantStatus,
    OracleRecord,
    RecordsPayload,
    TrialBudget,
)
from clastogen.mutation.injection import resolve_target
from clastogen.mutation.mutator import PromptMutator
from clastogen.reporting import terminal
from clastogen.reporting.render import render_html, render_json, render_markdown
from clastogen.reporting.scoring import below_threshold, summarize
from clastogen.runner import plan_test, run_mutation, run_oracle, to_payload
from clastogen.suppressions import load_suppressions
from clastogen.trial import run_trial

logger = logging.getLogger(__name__)


STATE_KEY = pytest.StashKey[ClastogenPluginState]()
_RECORDS_KEY = "clastogen_records"


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
        "--clastogen-max-calls",
        type=int,
        metavar="N",
        help="With --clastogen, run at most N trials per process (xdist: N divided by the workers); mutants left without budget are UNMEASURED.",
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
            self._state.unmutated.update((target, sentence) for target, sentence in records["unmutated"])
            self._state.tests.update(records["tests"])
            self._state.measured_targets.update(records["targets"])
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
    enabled = config.getoption("--clastogen")
    suppressed = load_suppressions(config.rootpath) if enabled else set()
    custom = load_custom_mutants(config.rootpath) if enabled else ()
    limit: int | None = config.getoption("--clastogen-max-calls", None)
    if limit is not None and limit < 1:
        raise pytest.UsageError(f"--clastogen-max-calls must be at least 1, got {limit}")
    # Checked on the controller, which knows the worker count before any worker starts; a worker would get 0 calls.
    workers = len(config.getoption("tx", ()))
    if limit is not None and workers > limit:
        raise pytest.UsageError(f"--clastogen-max-calls {limit} is below the {workers} xdist workers; each needs one")
    if limit is not None and _is_xdist_worker(config):
        limit //= int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
    state = ClastogenPluginState(suppressed_mutants=suppressed, custom_mutants=custom, budget=TrialBudget(limit))
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
    trial = state.budget.wrap(partial(run_trial, item))
    if (samples := params.oracle_samples) is not None:
        oracle = []
        oracle_ran = item.config.getoption("--clastogen-oracle", False)
        if oracle_ran:
            oracle = run_oracle(
                item.nodeid,
                trial,
                params,
                samples,
                prompt_text,
                state.suppressed_mutants,
                state.budget,
                state.custom_mutants,
            )
        item.user_properties.append(
            (_RECORDS_KEY, to_payload(oracle=oracle, targets=[params.target] if oracle_ran else []))
        )
        return res

    baselines, executions, constraints, unmutated = run_mutation(item.nodeid, trial, params, prompt_text, state)
    tests = {params.target: state.tests.get(params.target, 1)}
    item.user_properties.append(
        (_RECORDS_KEY, to_payload(baselines, executions, constraints, unmutated, tests=tests, targets=[params.target]))
    )
    return res


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:  # noqa: ANN001  # xdist WorkerController, xdist is optional
    node.workerinput["clastogen_repeats"] = node.workerinput["workercount"] if node.config.option.dist == "each" else 1


def _check_custom_mutants(session: pytest.Session, state: ClastogenPluginState) -> None:
    """Fails the run before any trial when a custom mutant does not fit the current prompt of its target.

    A target no collected test marks is not an error here: running a subset of the suite legitimately leaves it out,
    so the terminal summary warns about it instead. An xdist worker skips the check, because an exception while it
    collects crashes the controller; there the same ValueError fails the first test that generates the mutants.
    """
    if not state.custom_mutants or _is_xdist_worker(session.config):
        return
    seen: set[str] = set()
    for item in session.items:
        if (marker := item.get_closest_marker("clastogen")) is None:
            continue
        try:
            params, prompt_text = _parse_marker(item, marker)
        except pytest.UsageError:
            continue  # reported when the test runs
        if params.target in seen:
            continue
        seen.add(params.target)
        try:
            PromptMutator(params.target, state.custom_mutants).custom_mutants(prompt_text)
        except ValueError as exc:
            raise pytest.UsageError(f"{exc} (.clastogen/mutants.toml, used by '{item.nodeid}')") from exc


def pytest_collection_finish(session: pytest.Session) -> None:
    """Counts the tests per target after every deselection (-k, --lf), and under --collect-only previews the run."""
    config = session.config
    if not config.getoption("--clastogen", False):
        return

    state = config.stash[STATE_KEY]
    # Probes get no SPRT verdict, so they are not part of the family. Every worker collects every selected test,
    # so each counts the whole family; with --dist each a worker runs all of them again, so the family grows by the
    # worker count (the controller says so in workerinput). A bad marker is reported when the test runs, not here.
    repeats = getattr(config, "workerinput", {}).get("clastogen_repeats", 1)
    for item in session.items:
        if marker := item.get_closest_marker("clastogen"):
            try:
                params = ClastogenParams(*marker.args, **marker.kwargs)
            except (TypeError, ValueError):
                continue
            if params.oracle_samples is None:
                state.tests[params.target] = state.tests.get(params.target, 0) + repeats
    _check_custom_mutants(session, state)
    if config.option.collectonly:
        with_oracle = config.getoption("--clastogen-oracle", False)
        for item in session.items:
            if marker := item.get_closest_marker("clastogen"):
                params, prompt_text = _parse_marker(item, marker)
                state.preview.append(
                    plan_test(
                        item.nodeid, params, prompt_text, state.suppressed_mutants, with_oracle, state.custom_mutants
                    )
                )


def _write_report(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Computes the mutation summary once, then exports JSON/HTML/Markdown and enforces --clastogen-fail-under from it."""
    if (
        not session.config.getoption("--clastogen", False)
        or _is_xdist_worker(session.config)
        or session.config.option.collectonly
    ):
        return

    state = session.config.stash.get(STATE_KEY, None)
    if state is None:
        return
    state.baselines.sort(key=lambda b: b.test_id)
    state.executions.sort(key=lambda r: (r.mutant_id, r.test_id))
    summary = summarize(state.executions, state.constraints, state.oracle, state.unmutated, state.tests)
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
    fail_under_failed = fail_under is not None and below_threshold(summary, float(fail_under))
    # Unmeasured mutants fail the run only when a score threshold was asked for; without one, the budget is the user's choice.
    unmeasured = summary.counts[MutantStatus.UNMEASURED] if fail_under is not None else 0
    if session.exitstatus == pytest.ExitCode.OK and (fail_under_failed or errored or unmeasured):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    if fail_under_failed and session.exitstatus == pytest.ExitCode.TESTS_FAILED:
        # pytest prints shouldfail in red just above its final "N passed" line; keep a reason it already set (-x).
        detail = "no measurable mutants" if summary.score is None else f"score {summary.score:.1f}%"
        message = f"Clastogen fail-under {float(fail_under):.1f}%: FAILED ({detail})"
    elif errored and session.exitstatus == pytest.ExitCode.TESTS_FAILED:
        message = f"Clastogen: {mutant_errors} mutant error(s)"
    elif unmeasured and session.exitstatus == pytest.ExitCode.TESTS_FAILED:
        message = f"Clastogen: {unmeasured} mutant(s) unmeasured within --clastogen-max-calls"
    else:
        return
    prev = session.shouldfail
    session.shouldfail = f"{prev}; {message}" if isinstance(prev, str) and prev else message


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config) -> None:
    """Renders the terminal summary from the session-wide MutationSummary computed in sessionfinish."""
    if not config.getoption("--clastogen", False):
        return

    state = config.stash.get(STATE_KEY, None)
    if state is None:
        return
    if config.option.collectonly:
        terminal.write_preview(terminalreporter, state.preview)
        return
    if state.summary is None:
        return

    terminal.write(terminalreporter, state, state.summary)
