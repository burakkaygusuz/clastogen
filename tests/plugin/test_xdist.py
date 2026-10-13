from __future__ import annotations

import json

import pytest

from tests.plugin import AGENT, STRONG, WEAK


def test_parallel_workers_reach_the_same_verdicts_as_a_serial_run(
    pytester: pytest.Pytester, subprocess_env: None
) -> None:
    pytester.makepyfile(agent=AGENT, test_a_strong=STRONG, test_b_weak=WEAK)
    runs = {}
    for name, args in (("serial", ()), ("parallel", ("-n", "2", "--dist", "loadfile"))):
        json_out = pytester.path / f"{name}.json"
        result = pytester.runpytest_subprocess("--clastogen", f"--clastogen-json={json_out}", *args)
        assert result.ret == pytest.ExitCode.OK, result.stdout.str()
        runs[name] = json.loads(json_out.read_text(encoding="utf-8"))

    def verdicts(data: dict) -> dict[str, str]:
        return {r["mutant_id"]: r["status"] for r in data["results"]}

    assert verdicts(runs["parallel"]) == verdicts(runs["serial"])
    assert {b["test_id"].split("::")[-1] for b in runs["parallel"]["baselines"]} == {"test_strong", "test_weak"}
    assert runs["parallel"]["constraint_coverage"] == runs["serial"]["constraint_coverage"]
    assert runs["parallel"]["by_operator"] == runs["serial"]["by_operator"]
    assert runs["parallel"]["mutation_score"] == runs["serial"]["mutation_score"] == 100.0
    # The serial run skips mutants that the strong test killed; each worker only knows its own kills.
    assert runs["serial"]["calls"] < runs["parallel"]["calls"]
