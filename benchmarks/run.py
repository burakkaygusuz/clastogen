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
OFF_TOPIC = "off_topic"
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
    # A failing test on the original prompt exits 1 but still writes the JSON; only a missing JSON is a crash.
    if not out.exists():
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


RANK = {s: i for i, s in enumerate(["KILLED", "SURVIVED", "INCONCLUSIVE", "ERROR", "SKIPPED", "SUPPRESSED"])}


def summarize(suite: str, exclude: str | None = None) -> dict[str, Any]:
    """Summarizes a suite from its execution records, leaving out the tests whose id contains `exclude`."""
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"{suite}_*.json"))]
    per_run = [[e for e in r["executions"] if exclude is None or exclude not in e["test_id"]] for r in runs]
    descriptions = {e["mutant_id"]: e["description"] for execs in per_run for e in execs}
    verdicts: dict[str, list[str]] = {m: [] for m in descriptions}
    scores, calls, fixed = [], [], []
    for execs in per_run:
        merged: dict[str, str] = {}
        for e in execs:  # the strongest status among a mutant's tests is its verdict
            merged[e["mutant_id"]] = min(merged.get(e["mutant_id"], e["status"]), e["status"], key=lambda x: RANK[x])
        for mutant_id, status in merged.items():
            verdicts[mutant_id].append(status)
        scores.append(100 * list(merged.values()).count("KILLED") / len(merged))
        calls.append(sum(e["sample_count"] or 0 for e in execs))
        fixed.append(sum(e["fixed_n"] or 0 for e in execs))
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
    # A test that fails on the original prompt is not mutated and leaves no execution records.
    tests = {e["test_id"] for execs in per_run for e in execs}
    unmutated_runs = sum({e["test_id"] for e in execs} != tests for execs in per_run)
    executions = [e for execs in per_run for e in execs]
    savings = [100 * (1 - c / f) for c, f in zip(calls, fixed, strict=True)]

    def spread(values: list[float]) -> dict[str, float]:
        return {"mean": round(statistics.mean(values), 1), "min": round(min(values), 1), "max": round(max(values), 1)}

    return {
        "runs": len(runs),
        "executions": len(executions),
        "inconclusive": sum(e["status"] == "INCONCLUSIVE" for e in executions),
        "unmutated_runs": unmutated_runs,
        "score": spread(scores),
        "calls": spread(calls),
        "fixed_calls": spread(fixed),
        "saving_percent": spread(savings),
        "mutants": mutants,
    }


def off_topic_compliance() -> tuple[int, int]:
    """Runs of the strong suite in which the off-topic test passed on the original prompt, out of all runs."""
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("strong_*.json"))]
    return sum(any(OFF_TOPIC in e["test_id"] for e in r["executions"]) for r in runs), len(runs)


def markdown(info: dict[str, Any], summaries: dict[str, dict[str, Any]], off_topic: tuple[int, int]) -> str:
    lines = [
        f"Model `{info['model']}` through pi {info['pi']}, clastogen {info['clastogen']}, "
        f"commit `{str(info['git_commit'])[:9]}`{' (dirty)' if info['git_dirty'] else ''}, {info['date']}.",
        "",
        "| Suite | Runs | Mutation Score % (min-max) | Calls (min-max) | Fixed-N calls | Saving % (min-max) "
        "| INCONCLUSIVE executions | Runs with an unmutated test |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for suite, s in summaries.items():
        score, calls, fixed, saving = s["score"], s["calls"], s["fixed_calls"], s["saving_percent"]
        lines.append(
            f"| {suite} | {s['runs']} | {score['mean']} ({score['min']}-{score['max']}) "
            f"| {calls['mean']} ({calls['min']}-{calls['max']}) | {fixed['mean']} "
            f"| {saving['mean']} ({saving['min']}-{saving['max']}) "
            f"| {s['inconclusive']}/{s['executions']} | {s['unmutated_runs']}/{s['runs']} |"
        )
    passed, total = off_topic
    lines += [
        "",
        f"The off-topic test passed on the original prompt in {passed} of {total} runs: the model followed "
        '"only banking topics" in that share of single replies.',
    ]
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


def write_summary(info: dict[str, Any]) -> None:
    summaries = {
        "weak": summarize("weak", OFF_TOPIC),
        "strong": summarize("strong", OFF_TOPIC),
        "strong, with the off-topic test": summarize("strong"),
    }
    off_topic = off_topic_compliance()
    document = {"provenance": info, "off_topic_passed": off_topic, **summaries}
    (RESULTS / "summary.json").write_text(json.dumps(document, indent=2) + "\n")
    (RESULTS / "summary.md").write_text(markdown(info, summaries, off_topic))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    jobs = [(suite, repeat) for repeat in range(1, REPEATS + 1) for suite in SUITES]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda job: run_once(*job), jobs))
    write_summary(provenance())
    print((RESULTS / "summary.md").read_text())


if __name__ == "__main__":
    main()
