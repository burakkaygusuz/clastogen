from pathlib import Path

import pytest

from clastogen.plugin import load_suppressions


def _write_suppressions(root: Path, text: str) -> None:
    (root / ".clastogen").mkdir()
    (root / ".clastogen" / "suppressions.toml").write_text(text, encoding="utf-8")


def test_load_suppressions_empty_when_file_missing(tmp_path: Path) -> None:
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
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "must be a table" in result.stderr.str()
