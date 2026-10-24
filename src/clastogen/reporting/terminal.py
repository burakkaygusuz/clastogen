"""Writes the Clastogen summary into pytest's terminal report."""

from collections import Counter

import pytest

from clastogen.models import (
    CUSTOM_OPERATOR,
    MIN_BASELINE_RATE,
    NOISY_BASELINE_RATE,
    BaselineRecord,
    ClastogenPluginState,
    MutantStatus,
    MutationSummary,
    RunPlan,
    snippet_covers,
)
from clastogen.reporting.render import STATUS_LABELS, kill_hint
from clastogen.reporting.scoring import (
    MERGE_NOTE,
    behavior_label,
    below_threshold,
    calls_line,
    coverage_line,
    exposure_line,
    operator_line,
    rule_line,
    score_line,
    survivor_tests,
    weakest_first,
)

_SHOWN = 10

_STATUS_MARKUP: dict[MutantStatus, dict[str, bool]] = {
    MutantStatus.SURVIVED: {"red": True, "bold": True},
    MutantStatus.INCONCLUSIVE: {"yellow": True},
    MutantStatus.ERROR: {"purple": True},
    MutantStatus.KILLED: {"green": True},
    MutantStatus.SKIPPED: {"light": True},
    MutantStatus.UNMEASURED: {"light": True},
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


def _write_mutants(
    terminalreporter: pytest.TerminalReporter, state: ClastogenPluginState, summary: MutationSummary
) -> None:
    verbose = terminalreporter.config.get_verbosity() > 0
    order = list(STATUS_LABELS)
    show_target = len({r.target for r in summary.results}) > 1
    # A test is named without its file path unless another file has a test with the same name.
    short = {e.test_id: e.test_id.split("::", 1)[-1] for e in state.executions}
    clashes = {n for n, c in Counter(short.values()).items() if c > 1}
    name = {t: t if n in clashes else n for t, n in short.items()}
    passed_in = {m: [name[t] for t in tests] for m, tests in survivor_tests(state.executions).items()}

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


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _write_capped(terminalreporter: pytest.TerminalReporter, lines: list[str], verbose: bool) -> None:
    shown = lines if verbose else lines[:_SHOWN]
    for line in shown:
        terminalreporter.write_line(f"    {line}", light=True)
    if len(shown) < len(lines):
        terminalreporter.write_line(f"    … and {len(lines) - len(shown)} more (-v shows all)", light=True)


def _write_rules(terminalreporter: pytest.TerminalReporter, summary: MutationSummary, verbose: bool) -> None:
    if summary.rules:
        terminalreporter.write_line(f"Rules, weakest first ({MERGE_NOTE})", yellow=True)
        _write_capped(
            terminalreporter, [f"{rule_line(r)}  {_clip(r.rule, 60)}" for r in weakest_first(summary.rules)], verbose
        )
        # A rule whose only mutant is a suppressed custom one is untested by choice, so max_mutants would not help.
        suppressed = [
            r for r in summary.results if r.operator_name == CUSTOM_OPERATOR and r.status == MutantStatus.SUPPRESSED
        ]
        if any(
            not any(s.target == r.target and snippet_covers(s.original_snippet, r.rule) for s in suppressed)
            for r in summary.rules
            if r.status == "untested"
        ):
            terminalreporter.write_line("    Untested: raise max_mutants, fix errors or add tests.", light=True)
    if summary.unmutated:
        terminalreporter.write_line(
            "Unmutated sentences (no rule keyword or digit, so no rule-level mutant):", yellow=True
        )
        _write_capped(terminalreporter, [_clip(sentence, 100) for _, sentence in summary.unmutated], verbose)


def _write_summary(
    terminalreporter: pytest.TerminalReporter,
    state: ClastogenPluginState,
    summary: MutationSummary,
    fail_under: float | None,
) -> None:
    verbose = terminalreporter.config.get_verbosity() > 0
    terminalreporter.write_sep("=", "Clastogen Mutation Testing Summary", bold=True)
    _write_mutants(terminalreporter, state, summary)
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
        split = max(
            (summary.tests.get(r.target, 1) for r in summary.results if r.status == MutantStatus.INCONCLUSIVE),
            default=1,
        )
        hint = (
            f" With {split} tests on one target, alpha is divided by {split}: that needs more runs."
            if split > 1
            else ""
        )
        terminalreporter.write_line(
            f"{inconclusive} inconclusive: no decision after max_steps runs.{hint} Increase max_steps or delta.",
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
    _write_rules(terminalreporter, summary, verbose)


def write_preview(terminalreporter: pytest.TerminalReporter, plans: list[RunPlan]) -> None:
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
    unused_targets = sorted({c.target for c in state.custom_mutants} - state.measured_targets)
    if unused_targets and state.marked_tests_count > 0:
        terminalreporter.write_line(
            f"Custom mutants for targets no test measured in this run: {', '.join(unused_targets)} (no marked test, or filtered out)",
            yellow=True,
        )
