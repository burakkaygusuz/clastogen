import re
from collections.abc import Sequence
from dataclasses import replace
from html import escape
from importlib.resources import files

from clastogen.models import BaselineRecord, MutantExecution, MutationSummary
from clastogen.reporting.scoring import MEASURED, score_line
from clastogen.stats.sprt import MIN_BASELINE_RATE
from clastogen.types import MutantStatus

# Declaration order is the display order: blind spots first, measured kills next, excluded mutants last.
STATUS_LABELS = {
    MutantStatus.SURVIVED: "✗ SURVIVED",
    MutantStatus.INCONCLUSIVE: "? INCONCLUSIVE",
    MutantStatus.ERROR: "! ERROR",
    MutantStatus.KILLED: "✓ KILLED",
    MutantStatus.SKIPPED: "↷ SKIPPED",
    MutantStatus.SUPPRESSED: "⊘ SUPPRESSED",
}

_COLORS = {
    MutantStatus.KILLED: "var(--killed)",
    MutantStatus.SURVIVED: "var(--survived)",
    MutantStatus.INCONCLUSIVE: "var(--inconclusive)",
    MutantStatus.ERROR: "var(--error)",
    MutantStatus.SKIPPED: "var(--other)",
    MutantStatus.SUPPRESSED: "var(--suppressed)",
}

_CSS = files("clastogen").joinpath("reporting/report.css").read_text(encoding="utf-8")


def _badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{escape(label)}</span>'


def _status_badge(status: MutantStatus) -> str:
    return _badge(status.value, _COLORS[status])


def _error(error: str | None) -> str:
    return f'<div class="err">{escape(error)}</div>' if error else ""


def _cards(summary: MutationSummary) -> str:
    score = "N/A" if summary.score is None else f"{summary.score:.1f}%"
    bar = "".join(
        f'<span style="width:{summary.counts[s] / summary.total * 100:.2f}%;background:{_COLORS[s]}"></span>'
        for s in MutantStatus
        if summary.total and s in MEASURED
    )
    cards = [
        f'<div class="card score"><div class="l">Mutation score</div><div class="n">{score}</div>'
        f'<div class="bar">{bar}</div></div>',
        f'<div class="card"><div class="l">Mutants</div><div class="n">{summary.total}</div></div>',
    ]
    cards += [
        f'<div class="card"><div class="l">{s.value.lower()}</div>'
        f'<div class="n" style="color:{_COLORS[s]}">{n}</div></div>'
        for s, n in summary.counts.items()
        if n
    ]
    return f'<div class="cards">{"".join(cards)}</div>'


def _donut(summary: MutationSummary) -> str:
    """Status distribution as SVG circle segments; r=15.915 makes the circumference 100 so dash lengths are percents."""
    arcs: list[str] = []
    legend: list[str] = []
    offset = 25.0  # starts the first segment at 12 o'clock
    for s, n in summary.counts.items():
        pct = n / summary.total * 100 if summary.total else 0.0
        if not n:
            continue
        arcs.append(
            f'<circle cx="21" cy="21" r="15.915" fill="none" stroke-width="6" style="stroke:{_COLORS[s]}" '
            f'stroke-dasharray="{pct:.3f} {100 - pct:.3f}" stroke-dashoffset="{offset:.3f}"/>'
        )
        legend.append(
            f'<li><span class="sw" style="background:{_COLORS[s]}"></span>{s.value.lower()}'
            f" <code>{n} · {pct:.0f}%</code></li>"
        )
        offset -= pct
    return (
        '<div class="donut"><svg viewBox="0 0 42 42" role="img" aria-label="Mutant status distribution">'
        '<circle cx="21" cy="21" r="15.915" fill="none" stroke-width="6" style="stroke:var(--line)"/>'
        f'{"".join(arcs)}<text x="21" y="22" text-anchor="middle" class="dn">{summary.total}</text>'
        '<text x="21" y="26.5" text-anchor="middle" class="dl">mutants</text></svg>'
        f'<ul class="legend">{"".join(legend)}</ul></div>'
    )


def _llr_chart(summary: MutationSummary) -> str:
    measured = [(r, r.llr) for r in summary.results if r.llr is not None]
    if not measured:
        return '<p class="empty">No measured mutants.</p>'
    scale = max(abs(llr) for _, llr in measured) or 1.0
    rows = "".join(
        f"<code>{escape(r.mutant_id)}</code>"
        f'<div class="track" title="{escape(r.description)}"><span style="{"left" if llr >= 0 else "right"}:50%;'
        f'width:{abs(llr) / scale * 50:.2f}%;background:{_COLORS[r.status]}"></span></div>'
        f"<code>{llr:+.2f} · {r.sample_count} runs</code>"
        for r, llr in measured
    )
    return (
        f'<div class="llr">{rows}</div>'
        '<p class="note">Cumulative SPRT log-likelihood ratio at the decision: right of center favours a broken '
        "prompt (KILLED), left favours an unchanged one (SURVIVED).</p>"
    )


def _matrix(summary: MutationSummary, executions: Sequence[MutantExecution]) -> str:
    tests = sorted({e.test_id for e in executions})
    if not tests or not summary.results:
        return '<p class="empty">No mutant was evaluated by a test.</p>'
    cells = {(e.mutant_id, e.test_id): e.status for e in executions}
    head = "".join(f'<th title="{escape(t)}"><code>{escape(t.split("::")[-1])}</code></th>' for t in tests)
    rows = []
    for r in summary.results:
        tds = []
        for t in tests:
            status = cells.get((r.mutant_id, t))
            if status is None:
                tds.append('<td class="cell none" title="not evaluated">·</td>')
            else:
                tds.append(
                    f'<td class="cell"><span class="sq" style="background:{_COLORS[status]}" '
                    f'title="{status.value}">{STATUS_LABELS[status][0]}</span></td>'
                )
        rows.append(f"<tr><td><code>{escape(r.mutant_id)}</code></td>{''.join(tds)}</tr>")
    return (
        f'<div class="scroll"><table class="matrix"><thead><tr><th>Mutant</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="note">Each cell is what a test decided for a mutant; · means the test never ran it '
        "(for example because an earlier test had already killed it).</p>"
    )


_STATUS_ORDER = {status: i for i, status in enumerate(STATUS_LABELS)}


def _test_names(test_ids: Sequence[str]) -> str:
    return "<br>".join(f'<code title="{escape(t)}">{escape(t.split("::")[-1])}</code>' for t in test_ids)


def _survivor_tests(executions: Sequence[MutantExecution]) -> dict[str, list[str]]:
    passed_in: dict[str, list[str]] = {}
    for e in executions:
        if e.status == MutantStatus.SURVIVED:
            passed_in.setdefault(e.mutant_id, []).append(e.test_id)
    return passed_in


def _results_table(summary: MutationSummary, executions: Sequence[MutantExecution]) -> str:
    if not summary.results:
        return '<p class="empty">No mutants were evaluated.</p>'
    # A survivor passed every test that ran it, so name all of them instead of the one that represents the mutant.
    passed_in = _survivor_tests(executions)
    rows = "".join(
        f"<tr><td>{_status_badge(r.status)}</td>"
        f"<td>{escape(r.description)}{_error(r.error)}<br><code>{escape(r.mutant_id)} · {escape(r.target)}</code></td>"
        f'<td class="tests">{_test_names(passed_in.get(r.mutant_id, [r.test_id]))}</td>'
        + (
            f'<td class="num">{r.sample_count}</td><td class="num">{r.llr:+.2f}</td></tr>'
            if r.llr is not None
            else '<td class="num">&ndash;</td><td class="num">&ndash;</td></tr>'
        )
        for r in summary.results
    )
    return (
        '<div class="scroll"><table><thead><tr><th>Status</th><th>Mutant</th><th>Tests</th><th>Runs</th><th>LLR</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
        '<p class="note">Tests: the test that killed the mutant, or every test that still passed with it.</p>'
    )


def _baselines_table(baselines: Sequence[BaselineRecord]) -> str:
    if not baselines:
        return '<p class="empty">No baselines were recorded.</p>'
    rows = "".join(
        f"<tr><td>{_badge('STABLE', 'var(--killed)') if b.stable else _badge('FLAKY', 'var(--survived)')}</td>"
        f"<td><code>{escape(b.test_id)}</code>{_error(b.error)}</td><td><code>{escape(b.target)}</code></td>"
        f'<td class="num"><div class="rate"><span style="width:{b.successes / b.runs * 100 if b.runs else 0:.1f}%;'
        f'background:{"var(--killed)" if b.stable else "var(--survived)"}"></span>'
        f'<i style="left:{MIN_BASELINE_RATE * 100:.0f}%"></i></div>{b.successes}/{b.runs}</td>'
        f'<td class="num">{b.p0:.3f}</td></tr>'
        for b in baselines
    )
    return (
        "<table><thead><tr><th>Baseline</th><th>Test</th><th>Target</th><th>Passed</th><th>p0</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f'<p class="note">The marker is the {MIN_BASELINE_RATE:.0%} minimum baseline pass rate; '
        "p0 is the Laplace estimate fed to the SPRT.</p>"
    )


def render_html(
    summary: MutationSummary, executions: Sequence[MutantExecution], baselines: Sequence[BaselineRecord]
) -> str:
    """Renders a self-contained HTML report with inline SVG/CSS charts and no external assets."""
    summary = replace(summary, results=tuple(sorted(summary.results, key=lambda r: _STATUS_ORDER[r.status])))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Clastogen Report</title><style>{_CSS}</style></head><body><main>"
        "<h1>Clastogen Mutation Testing Report</h1>"
        '<p class="sub">Score = killed / (killed + survived + inconclusive).</p>'
        f"{_cards(summary)}"
        f'<div class="charts"><div class="panel"><h3>Status distribution</h3>{_donut(summary)}</div>'
        f'<div class="panel"><h3>SPRT evidence per mutant</h3>{_llr_chart(summary)}</div></div>'
        f"<h2>Kill matrix</h2>{_matrix(summary, executions)}"
        f"<h2>Mutants</h2>{_results_table(summary, executions)}"
        f"<h2>Baselines</h2>{_baselines_table(baselines)}</main></body></html>"
    )


def _code(text: str) -> str:
    # Prompt text is untrusted: a code span renders it literally, with no HTML, links or @mentions in a PR comment.
    text = " ".join(text.split()).replace("|", "\\|")
    fence = "`" * (max(map(len, re.findall("`+", text)), default=0) + 1)
    return f"{fence} {text} {fence}"


def render_markdown(
    summary: MutationSummary, executions: Sequence[MutantExecution], baselines: Sequence[BaselineRecord]
) -> str:
    """Renders a compact GitHub-flavored Markdown summary for job summaries and pull request comments."""
    lines = ["## Clastogen mutation testing", "", f"**{score_line(summary)}**"]
    if flaky := [b.test_id.split("::")[-1] for b in baselines if not b.stable]:
        lines += ["", f"Flaky tests (not mutated): {', '.join(map(_code, flaky))}"]
    if summary.results:
        passed_in = _survivor_tests(executions)
        lines += ["", "| Status | Mutant | Change | Tests |", "| --- | --- | --- | --- |"]
        for r in sorted(summary.results, key=lambda r: _STATUS_ORDER[r.status]):
            tests = ", ".join(t.split("::")[-1] for t in passed_in.get(r.mutant_id, [r.test_id]))
            lines.append(f"| {STATUS_LABELS[r.status]} | `{r.mutant_id}` | {_code(r.description)} | {_code(tests)} |")
    return "\n".join(lines) + "\n"
