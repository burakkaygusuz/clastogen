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
        ("[[suppression]]\nmutant_id = 'abcdefabcdef'\nreason = 'x'\n", "Unknown top-level key.*suppression"),
        ("[[suppressions]]\nmutant_id = 'abcdefabcdef'\nreason = 'x'\nreasn = 'y'\n", "Unknown key.*reasn"),
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


def test_clastogen_dir_outside_the_rootdir_fails_instead_of_being_ignored(pytester: pytest.Pytester) -> None:
    _write_suppressions(pytester.path, "")
    (pytester.path / "sub").mkdir()
    result = pytester.runpytest("--clastogen", "--rootdir=sub")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*is ignored: Clastogen reads .clastogen/ only from the pytest rootdir*"])


def test_clastogen_dir_of_a_path_argument_fails_even_when_the_rootdir_has_one(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    _write_suppressions(pytester.path, "")
    (pytester.path / "sub").mkdir()
    _write_suppressions(pytester.path / "sub", "")
    (pytester.path / "sub" / "test_a.py").write_text("def test_a(): pass\n")
    result = pytester.runpytest("--clastogen", "sub")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*sub*.clastogen*is ignored*"])


def test_clastogen_dir_in_the_rootdir_is_not_reported(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    _write_suppressions(pytester.path, "")
    (pytester.path / "sub").mkdir()
    (pytester.path / "sub" / "test_a.py").write_text("def test_a(): pass\n")
    result = pytester.runpytest("--clastogen", "sub")

    assert result.ret == pytest.ExitCode.OK


def test_clastogen_dir_of_the_invocation_dir_without_arguments_fails(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makeini("[pytest]\n")
    sub = pytester.path / "sub"
    sub.mkdir()
    _write_suppressions(sub, "")
    (sub / "test_a.py").write_text("def test_a(): pass\n")
    monkeypatch.chdir(sub)
    result = pytester.runpytest_inprocess("--clastogen")

    assert result.ret == pytest.ExitCode.USAGE_ERROR


def test_a_symlinked_rootdir_is_not_reported(pytester: pytest.Pytester) -> None:
    _write_suppressions(pytester.path, "")
    link = pytester.path.parent / f"{pytester.path.name}_link"
    link.symlink_to(pytester.path, target_is_directory=True)
    result = pytester.runpytest("--clastogen", f"--rootdir={link}")

    assert result.ret != pytest.ExitCode.USAGE_ERROR


def _nested_project(pytester: pytest.Pytester) -> Path:
    pytester.makeini("[pytest]\n")
    sub = pytester.path / "tests" / "sub"
    sub.mkdir(parents=True)
    _write_suppressions(sub, "")
    (sub / "test_a.py").write_text("def test_a(): pass\n")
    return sub


@pytest.mark.parametrize("args", [(), ("tests",)], ids=["no_args", "tests_arg"])
def test_clastogen_dir_between_a_test_and_the_rootdir_fails(pytester: pytest.Pytester, args: tuple[str, ...]) -> None:
    _nested_project(pytester)
    result = pytester.runpytest("--clastogen", *args)

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*sub*.clastogen*is ignored*"])


def test_clastogen_dir_between_a_test_and_the_rootdir_fails_from_a_subdirectory(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _nested_project(pytester)
    monkeypatch.chdir(pytester.path / "tests")
    result = pytester.runpytest_inprocess("--clastogen")

    assert result.ret == pytest.ExitCode.USAGE_ERROR


def test_only_the_rootdir_clastogen_dir_is_fine_for_nested_tests(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    _write_suppressions(pytester.path, "")
    sub = pytester.path / "tests" / "sub"
    sub.mkdir(parents=True)
    (sub / "test_a.py").write_text("def test_a(): pass\n")
    result = pytester.runpytest("--clastogen")

    assert result.ret == pytest.ExitCode.OK


def test_nested_check_holds_with_a_symlinked_rootdir(pytester: pytest.Pytester) -> None:
    _write_suppressions(pytester.path, "")
    sub = pytester.path / "tests"
    sub.mkdir()
    (sub / "test_a.py").write_text("def test_a(): pass\n")
    link = pytester.path.parent / f"{pytester.path.name}_nested_link"
    link.symlink_to(pytester.path, target_is_directory=True)
    result = pytester.runpytest("--clastogen", f"--rootdir={link}", str(link / "tests"))

    assert result.ret == pytest.ExitCode.OK


def test_nested_clastogen_dir_fails_under_xdist(pytester: pytest.Pytester) -> None:
    _nested_project(pytester)
    result = pytester.runpytest("--clastogen", "-n", "2")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "is ignored" in result.stdout.str() + result.stderr.str()


def test_clean_layout_passes_under_xdist(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    _write_suppressions(pytester.path, "")
    (pytester.path / "tests").mkdir()
    (pytester.path / "tests" / "test_a.py").write_text("def test_a(): pass\n")
    result = pytester.runpytest("--clastogen", "-n", "2")

    assert result.ret == pytest.ExitCode.OK


def test_clastogen_dir_without_tests_is_fine_under_xdist(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    (pytester.path / "docs").mkdir()
    _write_suppressions(pytester.path / "docs", "")
    (pytester.path / "conftest_only").mkdir()
    (pytester.path / "conftest_only" / "conftest.py").write_text("")
    _write_suppressions(pytester.path / "conftest_only", "")
    (pytester.path / "test_a.py").write_text("def test_a(): pass\n")
    result = pytester.runpytest("--clastogen", "-n", "2")

    assert result.ret == pytest.ExitCode.OK
