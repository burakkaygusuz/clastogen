"""Writes the Clastogen summary into pytest's terminal report."""

from collections import Counter

import pytest

from clastogen.models import (
    MIN_BASELINE_RATE,
    NOISY_BASELINE_RATE,
    BaselineRecord,
    ClastogenPluginState,
    MutantStatus,
    MutationSummary,
    TestPlan,
)
from clastogen.reporting.render import STATUS_LABELS, kill_hint
from clastogen.reporting.scoring import (
    behavior_label,
    below_threshold,
    calls_line,
    coverage_line,
    exposure_line,
    operator_line,
    score_line,
    survivor_tests,
)

_UNTESTED_SHOWN = 10

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
        if hint := kill_hint(r):
            terminalreporter.write_line(f"    Assert: {hint}", light=True)

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
        ok = not below_threshold(summary, fail_under)
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


def write_preview(terminalreporter: pytest.TerminalReporter, plans: list[TestPlan]) -> None:
    """Writes what a run would do for the collected marked tests; nothing was executed."""
    if not plans:
        terminalreporter.write_line("Clastogen: No @pytest.mark.clastogen tests collected.")
        return
    rules = {(p.target, rule) for p in plans for rule in p.rules}
    terminalreporter.write_sep("=", "Clastogen Preview (--collect-only: nothing was run)", bold=True)
    terminalreporter.write_line(
        f"{len(plans)} marked tests, {len(rules)} rules, {sum(p.mutants for p in plans)} mutant runs "
        f"({sum(p.suppressed for p in plans)} suppressed)",
        bold=True,
    )
    terminalreporter.write_line(
        f"At most {sum(p.calls for p in plans)} test calls (worst case: every mutant runs to max_steps)", light=True
    )
    terminalreporter.write_sep("=", bold=True)


def write(terminalreporter: pytest.TerminalReporter, state: ClastogenPluginState, summary: MutationSummary) -> None:
    """Writes the summary of a --clastogen session, or the reason there is none."""
    config = terminalreporter.config
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
    # A subset of tests or a small max_mutants legitimately leaves entries unmatched, so this only warns.
    unused = sorted(state.suppressed_mutants - {e.mutant_id for e in state.executions})
    if unused and state.marked_tests_count > 0:
        terminalreporter.write_line(
            f"Suppressions matching no mutant in this run: {', '.join(unused)} (stale ID, or filtered out)", yellow=True
        )
