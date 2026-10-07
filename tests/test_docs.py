from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")


def test_readme_stats_example_matches_runnable_file() -> None:
    marker = "<!-- example: examples/test_stats_usage.py -->\n```python\n"
    start = README.index(marker) + len(marker)
    block = README[start : README.index("\n```\n", start) + 1]
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
    assert data["mutation_score"] == 60.0

    by_id = {r["mutant_id"]: r for r in data["results"]}
    inverted_identity = next(r for r in by_id.values() if "NEVER verify customer identity" in r["description"])
    assert inverted_identity["status"] == "KILLED"

    assert "7d1281020f96" in by_id
    assert "7d1281020f96" in README
