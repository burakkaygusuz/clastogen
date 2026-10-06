from collections.abc import Sequence
from dataclasses import replace
from html import escape

from clastogen.core.sprt import MIN_BASELINE_RATE
from clastogen.models import BaselineRecord, MutantExecution, MutationSummary
from clastogen.scoring import MEASURED
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

_CSS = """
:root {
  --bg: #f6f7f9; --card: #fff; --fg: #1d2330; --muted: #677189; --line: #e3e6ec;
  --killed: #1f9d55; --survived: #d64545; --inconclusive: #c98a00; --error: #8e5cd9;
  --other: #7a8499; --suppressed: #a0a8b8;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #11141a; --card: #1a1f28; --fg: #e6e9ef; --muted: #98a2b8; --line: #2a303c; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }
h1 { margin: 0 0 4px; font-size: 22px; }
h2 { margin: 32px 0 12px; font-size: 16px; }
h3 { margin: 0 0 12px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.sub, .note { color: var(--muted); }
.sub { margin: 0 0 24px; }
.note { font-size: 12px; margin: 10px 0 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card, .panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.card .n { font-size: 26px; font-weight: 650; }
.card .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.score .n { font-size: 34px; }
.bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--line); margin-top: 10px; }
.bar span { display: block; }
.charts { display: grid; grid-template-columns: minmax(260px, 1fr) 2fr; gap: 12px; margin-top: 12px; }
@media (max-width: 760px) { .charts { grid-template-columns: 1fr; } }
.donut { display: flex; align-items: center; gap: 20px; }
.donut svg { width: 150px; flex: none; }
.donut .dn { font-size: 8px; font-weight: 650; fill: var(--fg); }
.donut .dl { font-size: 3px; fill: var(--muted); text-transform: uppercase; }
.legend { list-style: none; margin: 0; padding: 0; font-size: 13px; }
.legend li { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.sw { width: 10px; height: 10px; border-radius: 3px; flex: none; }
.llr { display: grid; grid-template-columns: auto 1fr auto; gap: 6px 10px; align-items: center; }
.track { position: relative; height: 12px; background: var(--line); border-radius: 3px; }
.track::after { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--muted); }
.track span { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; background: var(--card);
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
tr:last-child td { border-bottom: 0; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
td.cell { text-align: center; vertical-align: middle; }
.matrix th:not(:first-child) { text-transform: none; text-align: center; }
.sq { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 6px;
  color: #fff; font-size: 13px; font-weight: 700; }
.none { color: var(--muted); }
code { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); overflow-wrap: anywhere; }
td.tests code { overflow-wrap: normal; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 650;
  color: #fff; white-space: nowrap; }
.rate { position: relative; width: 120px; height: 8px; border-radius: 4px; background: var(--line); margin-bottom: 4px; }
.rate span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; }
.rate i { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--fg); }
.err { color: var(--survived); font-size: 12px; }
.empty { color: var(--muted); }
"""


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


def _test_names(test_ids: Sequence[str]) -> str:
    return "<br>".join(f'<code title="{escape(t)}">{escape(t.split("::")[-1])}</code>' for t in test_ids)


def _results_table(summary: MutationSummary, executions: Sequence[MutantExecution]) -> str:
    if not summary.results:
        return '<p class="empty">No mutants were evaluated.</p>'
    # A survivor passed every test that ran it, so name all of them instead of the one that represents the mutant.
    passed_in: dict[str, list[str]] = {}
    for e in executions:
        if e.status == MutantStatus.SURVIVED:
            passed_in.setdefault(e.mutant_id, []).append(e.test_id)
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
    order = list(STATUS_LABELS)
    summary = replace(summary, results=tuple(sorted(summary.results, key=lambda r: order.index(r.status))))
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
