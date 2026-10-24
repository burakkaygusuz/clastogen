"""Adds the per-rule evidence table to summary.md and summary.json from the recorded run_NN.json files, offline.

The runs were recorded by clastogen 0.8.0, which did not save the unmutated sentences, so that column is not
rebuilt. Rules are the original snippets of the rule operators (section and example mutants cover no single rule).

    PYTHONPATH=src uv run python benchmarks/support_desk/rules.py
"""

import json
from pathlib import Path
from typing import Any

import pi_client

from clastogen.models import MutantExecution, MutantStatus
from clastogen.reporting.scoring import MERGE_NOTE, summarize

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / pi_client.MODEL.removeprefix("openrouter/").replace("/", "_")
HEADING = "## Per-rule evidence"


def rule_rows(runs: list[dict]) -> list[dict]:
    per_run = []
    for run in runs:
        fields: list[dict[str, Any]] = [{**e, "status": MutantStatus(e["status"])} for e in run["executions"]]
        executions = [MutantExecution(**f) for f in fields]
        rules = {
            (r["target"], r["original_snippet"])
            for r in run["results"]
            if not r["operator_name"].startswith(("delete_section", "delete_example"))
        }
        summary = summarize(executions, rules)
        assert (summary.constraints_killed, summary.constraints_total) == (
            run["constraint_coverage"]["killed"],
            run["constraint_coverage"]["total"],
        )
        per_run.append({r.rule: r for r in summary.rules})
    rows = []
    for rule in per_run[0]:
        covs = [p[rule] for p in per_run]
        rows.append(
            {
                "rule": rule,
                "mutants": [c.mutants for c in covs],
                "killed": [c.killed for c in covs],
                "killed_by": sorted({c.killed_by.split("::")[-1] for c in covs if c.killed_by}),
            }
        )
    return sorted(rows, key=lambda r: (sum(r["killed"]) > 0, r["rule"]))


def markdown(rows: list[dict], provenance: dict) -> str:
    lines = [
        f"{HEADING} (clastogen {provenance['clastogen']}, commit `{provenance['git_commit'][:9]}`)",
        "",
        "Rebuilt offline from the recorded run_01-05.json with the current report code; these are not results of a newer "
        f"clastogen version. {MERGE_NOTE} Columns list the 5 runs in order. `killed_by` is the lowest test ID among the killers in each run, "
        "joined over the runs. The unmutated sentences are not in these files (0.8.0 did not record them), so they are not listed.",
        "",
        "| Rule | Mutants per run | Killed per run | killed_by |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        tests = ", ".join(r["killed_by"]) or "-"
        lines.append(
            f"| {r['rule']} | {'/'.join(map(str, r['mutants']))} | {'/'.join(map(str, r['killed']))} | {tests} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("run_[0-9][0-9].json"))]
    document = json.loads((RESULTS / "summary.json").read_text())
    rows = rule_rows(runs)
    document["rules"] = {"merge_rule": MERGE_NOTE, "rows": rows}
    (RESULTS / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    md = (RESULTS / "summary.md").read_text().split(HEADING)[0].rstrip("\n")
    (RESULTS / "summary.md").write_text(md + "\n\n" + markdown(rows, document["provenance"]))


if __name__ == "__main__":
    main()
