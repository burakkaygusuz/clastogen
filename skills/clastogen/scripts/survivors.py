"""Summarize a Clastogen JSON report: the score, the coverage and the mutants that need work.

Usage: python scripts/survivors.py [REPORT]

REPORT is the file from --clastogen-json (default: clastogen.json).
Prints one JSON object to stdout. Exit codes: 0 done, 2 report missing or not a Clastogen report.

Example:
  python scripts/survivors.py clastogen.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

OPEN_STATUSES = ("SURVIVED", "INCONCLUSIVE", "ERROR")


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    behavior = report.get("behavior") or {}
    labels = behavior.get("labels", {})
    return {
        "mutation_score": report["mutation_score"],
        "mutation_score_ci95": report["mutation_score_ci95"],
        "counts": report["counts"],
        "by_operator": report["by_operator"],
        "constraint_coverage": report["constraint_coverage"],
        "behavior_exposure_percent": behavior.get("exposure_percent"),
        "open": [
            {
                "mutant_id": r["mutant_id"],
                "status": r["status"],
                "operator_name": r["operator_name"],
                "behavior": labels.get(r["mutant_id"]),
                "error": r["error"],
                "original_snippet": r["original_snippet"],
                "mutated_snippet": r["mutated_snippet"],
            }
            for r in report["results"]
            if r["status"] in OPEN_STATUSES
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a Clastogen JSON report.")
    parser.add_argument("report", nargs="?", default="clastogen.json", type=Path)
    path = parser.parse_args().report
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        result = summarize(report)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"Error: {path} is not a readable Clastogen report ({type(exc).__name__}).", file=sys.stderr)
        print("Run: pytest --clastogen --clastogen-json=clastogen.json", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
