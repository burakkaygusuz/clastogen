from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STATISTICS = (REPO / "docs" / "statistics.md").read_text(encoding="utf-8")
GUIDE = (REPO / "docs" / "guide.md").read_text(encoding="utf-8")


def test_docs_stats_example_matches_runnable_file() -> None:
    marker = "<!-- example: examples/test_stats_usage.py -->\n```python\n"
    start = STATISTICS.index(marker) + len(marker)
    block = STATISTICS[start : STATISTICS.index("\n```\n", start) + 1]
    assert block == (REPO / "examples" / "test_stats_usage.py").read_text(encoding="utf-8")


def test_demo_reports_documented_score(tmp_path: Path) -> None:
    json_out = tmp_path / "demo.json"
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--clastogen",
            f"--clastogen-json={json_out}",
            "examples/test_banking_eval.py",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["counts"]["KILLED"] == 3
    assert data["counts"]["SURVIVED"] == 2
    assert data["mutation_score"] == pytest.approx(60.0)

    by_id = {r["mutant_id"]: r for r in data["results"]}
    inverted_identity = next(r for r in by_id.values() if "NEVER verify customer identity" in r["description"])
    assert inverted_identity["status"] == "KILLED"
    assert inverted_identity["operator_name"] == "invert_negation"
    assert inverted_identity["original_snippet"] == (
        "You must always verify customer identity before providing balance details"
    )
    assert inverted_identity["mutated_snippet"] == (
        "You must NEVER verify customer identity before providing balance details"
    )

    documented = re.search(r'mutant_id = "([0-9a-f]{12})"', GUIDE)
    assert documented is not None
    assert by_id[documented[1]]["status"] == "SURVIVED"
