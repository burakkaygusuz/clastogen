"""Runs each eval suite against every prompt mutant REPEATS times and summarizes the runs.

Results go to results/<model>/. The summaries are committed; the raw runs (*_NN.json and *_NN.log) are git-ignored.

Every run is a separate serial `pytest --clastogen` process (no xdist), so the Calls line is not inflated by
workers that cannot see each other's kills. A run costs real LLM calls; finished runs are skipped on a rerun.

    python benchmarks/run.py
"""

import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import bank_agent

from clastogen import compute_wilson_interval

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / bank_agent.MODEL.removeprefix("openrouter/").replace("/", "_")
SUITES = ("weak", "strong")
REPEATS = 20
WORKERS = 4


def run_once(suite: str, repeat: int) -> None:
    out = RESULTS / f"{suite}_{repeat:02d}.json"
    if out.exists():
        return
    log = RESULTS / f"{suite}_{repeat:02d}.log"
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
                str(HERE / f"test_{suite}.py"),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode or not out.exists():
        tail = "\n".join(log.read_text().splitlines()[-20:])
        raise RuntimeError(f"{suite} run {repeat} failed (exit {proc.returncode}):\n{tail}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()


def provenance() -> dict[str, Any]:
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "clastogen": version("clastogen"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "pi": subprocess.run(["pi", "--version"], capture_output=True, text=True, check=True).stdout.strip(),
        "model": bank_agent.MODEL,
        "pi_flags": bank_agent.PI_FLAGS,
        "repeats": REPEATS,
    }


def summarize(suite: str) -> dict[str, Any]:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"{suite}_*.json"))]
    for r in runs:
        # The headline numbers must be the sum of the per-execution records.
        assert r["calls"] == sum(e["sample_count"] for e in r["executions"])
        assert r["fixed_calls"] == sum(e["fixed_n"] for e in r["executions"])
    verdicts: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    for r in runs:
        for m in r["results"]:
            verdicts.setdefault(m["mutant_id"], []).append(m["status"])
            descriptions[m["mutant_id"]] = m["description"]
    mutants = {}
    for mutant_id, statuses in verdicts.items():
        killed = statuses.count("KILLED")
        low, high = compute_wilson_interval(killed, len(statuses))
        mutants[mutant_id] = {
            "description": descriptions[mutant_id],
            "killed": killed,
            "runs": len(statuses),
            "wilson95": [round(low, 3), round(high, 3)],
            "statuses": sorted(set(statuses)),
        }
    savings = [100 * (1 - r["calls"] / r["fixed_calls"]) for r in runs]

    def spread(values: list[float]) -> dict[str, float]:
        return {"mean": round(statistics.mean(values), 1), "min": round(min(values), 1), "max": round(max(values), 1)}

    return {
        "runs": len(runs),
        "score": spread([r["mutation_score"] for r in runs]),
        "calls": spread([r["calls"] for r in runs]),
        "fixed_calls": spread([r["fixed_calls"] for r in runs]),
        "saving_percent": spread(savings),
        "mutants": mutants,
    }


def markdown(info: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"Model `{info['model']}` through pi {info['pi']}, clastogen {info['clastogen']}, "
        f"commit `{str(info['git_commit'])[:9]}`{' (dirty)' if info['git_dirty'] else ''}, {info['date']}.",
        "",
        "| Suite | Runs | Mutation Score % (min-max) | Calls (min-max) | Fixed-N calls | Saving % (min-max) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for suite, s in summaries.items():
        score, calls, fixed, saving = s["score"], s["calls"], s["fixed_calls"], s["saving_percent"]
        lines.append(
            f"| {suite} | {s['runs']} | {score['mean']} ({score['min']}-{score['max']}) "
            f"| {calls['mean']} ({calls['min']}-{calls['max']}) | {fixed['mean']} "
            f"| {saving['mean']} ({saving['min']}-{saving['max']}) |"
        )
    ids = list(summaries["strong"]["mutants"])
    lines += [
        "",
        "Share of runs that killed each mutant, with the Wilson 95% interval:",
        "",
        "| Mutant | " + " | ".join(summaries) + " |",
        "| --- | " + " | ".join("---" for _ in summaries) + " |",
    ]
    for mutant_id in ids:
        cells = []
        for s in summaries.values():
            m = s["mutants"][mutant_id]
            cells.append(f"{m['killed']}/{m['runs']} [{m['wilson95'][0]:.2f}, {m['wilson95'][1]:.2f}]")
        description = summaries["strong"]["mutants"][mutant_id]["description"]
        lines.append(f"| {description} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    jobs = [(suite, repeat) for repeat in range(1, REPEATS + 1) for suite in SUITES]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda job: run_once(*job), jobs))
    info = provenance()
    summaries = {suite: summarize(suite) for suite in SUITES}
    (RESULTS / "summary.json").write_text(json.dumps({"provenance": info, **summaries}, indent=2) + "\n")
    (RESULTS / "summary.md").write_text(markdown(info, summaries))
    print((RESULTS / "summary.md").read_text())


if __name__ == "__main__":
    main()
