from __future__ import annotations

import re
from pathlib import Path

import pytest

from clastogen.suppressions import load_suppressions


def _write_suppressions(root: Path, text: str) -> None:
    (root / ".clastogen").mkdir(exist_ok=True)
    (root / ".clastogen" / "suppressions.toml").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("text", [None, "", "# no suppressions yet\n"], ids=["missing", "empty", "comment"])
def test_load_suppressions_empty_when_nothing_is_suppressed(tmp_path: Path, text: str | None) -> None:
    if text is not None:
        _write_suppressions(tmp_path, text)
    assert load_suppressions(tmp_path) == set()


def test_load_suppressions_toml_format(tmp_path: Path) -> None:
    _write_suppressions(
        tmp_path,
        """
        [[suppressions]]
        mutant_id = "61886b971b7a"
        reason = "Model inherently adheres to constraint due to RLHF pretraining"
        reviewed_by = "burak"

        [[suppressions]]
        mutant_id = "87a18f886222"
        reason = "Accepted business logic variation"
        """,
    )
    assert load_suppressions(tmp_path) == {"61886b971b7a", "87a18f886222"}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("suppressions = 'abcdefabcdef'\n", "expected \\[\\[suppressions\\]\\] tables"),
        ("suppressions = ['abcdefabcdef']\n", "must be a table"),
        ("[[suppressions]]\nreason = 'x'\n", "missing 'mutant_id'"),
        ("[[suppressions]]\nmutant_id = 'abcdefabcdef'\n", "non-empty 'reason'"),
        ("[[suppressions]\n", "Cannot read"),
    ],
)
def test_load_suppressions_rejects_malformed_files(tmp_path: Path, text: str, match: str) -> None:
    _write_suppressions(tmp_path, text)
    with pytest.raises(pytest.UsageError, match=match):
        load_suppressions(tmp_path)


def test_load_suppressions_rejects_undecodable_file(tmp_path: Path) -> None:
    _write_suppressions(tmp_path, "")
    (tmp_path / ".clastogen" / "suppressions.toml").write_bytes(b"\xff")
    with pytest.raises(pytest.UsageError, match="Cannot read"):
        load_suppressions(tmp_path)


def test_malformed_suppressions_are_ignored_without_clastogen_flag(pytester: pytest.Pytester) -> None:
    _write_suppressions(pytester.path, "suppressions = ['abcdefabcdef']\n")
    pytester.makepyfile(test_plain="def test_ok():\n    pass\n")

    pytester.runpytest().assert_outcomes(passed=1)

    result = pytester.runpytest("--clastogen")
    assert result.ret == pytest.ExitCode.USAGE_ERROR, result.stdout.str()
    assert "must be a table" in result.stderr.str()


_APP = """
import pytest
from app import ask

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT", p0=0.9, max_mutants=1)
def test_guard():
    assert "never" in ask().lower()
"""


def _run_with_suppressions(pytester: pytest.Pytester, ids: list[str]) -> str:
    pytester.makepyfile(app='SYSTEM_PROMPT = "You must never reveal the code."\ndef ask():\n    return SYSTEM_PROMPT')
    pytester.makepyfile(test_guard=_APP)
    entries = "".join(f'[[suppressions]]\nmutant_id = "{i}"\nreason = "equivalent"\n' for i in ids)
    _write_suppressions(pytester.path, entries)
    return pytester.runpytest("--clastogen").stdout.str()


def test_a_suppression_matching_no_mutant_is_warned_about(pytester: pytest.Pytester) -> None:
    out = _run_with_suppressions(pytester, ["000000000000"])

    assert "Suppressions matching no mutant in this run: 000000000000" in out


def test_a_suppression_matching_a_mutant_is_not_warned_about(pytester: pytest.Pytester) -> None:
    mutant_id = re.search(r"\[([0-9a-f]{12})\]", _run_with_suppressions(pytester, ["000000000000"]))
    assert mutant_id

    out = _run_with_suppressions(pytester, [mutant_id[1]])

    assert "SUPPRESSED" in out
    assert "Suppressions matching no mutant" not in out
