"""Runs the support-desk case study: REPEATS clastogen runs and one oracle labelling of the same mutants.

Results go to results/<model>/. oracle.json and summary.* are committed; the raw runs (run_NN.*) are git-ignored.
Every clastogen run is a separate serial `pytest --clastogen` process, so the Calls line is not inflated by
workers that cannot see each other's kills. A run costs real LLM calls; finished runs are skipped on a rerun.

    PYTHONPATH=src:benchmarks/support_desk uv run python benchmarks/support_desk/run.py
"""

import json
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import oracle
import pi_client
import support_agent
from test_suite import pytestmark

from clastogen import PromptMutator, compute_wilson_interval

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / pi_client.MODEL.removeprefix("openrouter/").replace("/", "_")
MAX_MUTANTS = pytestmark.kwargs["max_mutants"]
REPEATS = 5
LABELLINGS = 3  # the oracle is statistical too: borderline mutants flip between labellings
WORKERS = 4


def run_once(repeat: int) -> None:
    out = RESULTS / f"run_{repeat:02d}.json"
    if out.exists():
        return
    log = RESULTS / f"run_{repeat:02d}.log"
    with log.open("w") as f:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--clastogen",
                f"--clastogen-json={out}",
                "-p",
                "no:cacheprovider",
                str(HERE / "test_suite.py"),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    # A failing test on the original prompt exits 1 but still writes the JSON; only a missing JSON is a crash.
    if not out.exists():
        tail = "\n".join(log.read_text().splitlines()[-20:])
        raise RuntimeError(f"run {repeat} failed (exit {proc.returncode}):\n{tail}")


def label_mutants() -> dict[str, list[bool | None]]:
    """The oracle's labels for the mutants clastogen generates (same hashes as IDs), one list per mutant."""
    path = RESULTS / "oracle.json"
    if not path.exists():
        mutants = PromptMutator(oracle.TARGET).generate_mutants(support_agent.SYSTEM_PROMPT, MAX_MUTANTS)
        labellings = [oracle.label_all(mutants) for _ in range(LABELLINGS)]
        path.write_text(json.dumps({m.id: [lab[m.id] for lab in labellings] for m in mutants}, indent=2) + "\n")
    return json.loads(path.read_text())


def quadrant(killed: int, runs: int, labels: list[bool | None]) -> str:
    """The oracle's majority label against clastogen's verdict; a mutant the labellings disagree on is borderline."""
    if len(set(labels)) > 1:
        return "borderline"
    return oracle.quadrant(killed, runs, labels[0])


def summarize(labels: dict[str, list[bool | None]]) -> dict[str, Any]:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("run_*.json"))]
    verdicts: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    for run in runs:
        for r in run["results"]:
            verdicts.setdefault(r["mutant_id"], []).append(r["status"])
            descriptions[r["mutant_id"]] = r["description"]
    mutants = {}
    for mutant_id, statuses in verdicts.items():
        killed = statuses.count("KILLED")
        low, high = compute_wilson_interval(killed, len(statuses))
        mutants[mutant_id] = {
            "description": descriptions[mutant_id],
            "killed": killed,
            "runs": len(statuses),
            "wilson95": [round(low, 3), round(high, 3)],
            "oracle_labels": labels[mutant_id],
            "quadrant": quadrant(killed, len(statuses), labels[mutant_id]),
        }

    def spread(values: list[float]) -> dict[str, float]:
        return {"mean": round(statistics.mean(values), 1), "min": round(min(values), 1), "max": round(max(values), 1)}

    return {
        "runs": len(runs),
        "score": spread([r["mutation_score"] for r in runs]),
        "constraint_coverage_percent": spread([r["constraint_coverage"]["percent"] for r in runs]),
        "calls": spread([r["calls"] for r in runs]),
        "noisy_baselines": sum(b["noisy"] for r in runs for b in r["baselines"]),
        "quadrants": dict(Counter(m["quadrant"] for m in mutants.values())),
        "mutants": mutants,
    }


def markdown(info: dict[str, Any], s: dict[str, Any]) -> str:
    lines = [
        f"Model `{info['model']}` through pi {info['pi']}, clastogen {info['clastogen']}, "
        f"commit `{info['git_commit'][:9]}`{' (dirty)' if info['git_dirty'] else ''}, {info['date']}.",
        "",
        f"{s['runs']} runs. Mutation Score {s['score']['mean']}% (min {s['score']['min']}, max {s['score']['max']}), "
        f"constraint coverage {s['constraint_coverage_percent']['mean']}%, "
        f"{s['calls']['mean']} calls per run, {s['noisy_baselines']} noisy baselines in all runs.",
        "",
        "Clastogen's verdict (killed in at least half of the runs) against the oracle: "
        + ", ".join(f"{n} {q}" for q, n in sorted(s["quadrants"].items()))
        + ".",
        "",
        "| Mutant | Killed in runs [Wilson 95%] | Oracle: behavior changed (3 labellings) | Result |",
        "| --- | --- | --- | --- |",
    ]
    for m in s["mutants"].values():
        changed = "/".join({True: "yes", False: "no", None: "-"}[label] for label in m["oracle_labels"])
        lines.append(
            f"| {m['description']} | {m['killed']}/{m['runs']} [{m['wilson95'][0]:.2f}, {m['wilson95'][1]:.2f}] "
            f"| {changed} | {m['quadrant']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(run_once, range(1, REPEATS + 1)))
    summary = summarize(label_mutants())
    document = {"provenance": {**pi_client.provenance(), "repeats": REPEATS}, **summary}
    (RESULTS / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    (RESULTS / "summary.md").write_text(markdown(document["provenance"], summary))
    print((RESULTS / "summary.md").read_text())


if __name__ == "__main__":
    main()
