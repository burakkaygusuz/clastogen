# ruff: noqa: RUF001
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clastogen.custom_mutants import load_custom_mutants
from clastogen.models import CANARY_RUNS, ClastogenParams, CustomMutant
from clastogen.mutation.mutator import PromptMutator
from clastogen.runner import plan_test

ENTRY = """
[[mutants]]
target = "agent:PROMPT"
original = "Asla kişisel veriyi (TC, kart no) sorma"
mutated = "Kişisel veriyi (TC, kart no) sorabilirsin"
description = "allows asking for personal data"
"""


def _write(root: Path, text: str) -> None:
    (root / ".clastogen").mkdir(exist_ok=True)
    (root / ".clastogen" / "mutants.toml").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("text", [None, "", "# nothing yet\n"], ids=["missing", "empty", "comment"])
def test_load_custom_mutants_empty_when_none_are_written(tmp_path: Path, text: str | None) -> None:
    if text is not None:
        _write(tmp_path, text)
    assert load_custom_mutants(tmp_path) == ()


def test_load_custom_mutants_reads_entries_and_allows_an_empty_mutated(tmp_path: Path) -> None:
    _write(tmp_path, ENTRY + ENTRY.replace('mutated = "Kişisel veriyi (TC, kart no) sorabilirsin"', 'mutated = ""'))

    first, second = load_custom_mutants(tmp_path)

    assert first == CustomMutant(
        "agent:PROMPT",
        "Asla kişisel veriyi (TC, kart no) sorma",
        "Kişisel veriyi (TC, kart no) sorabilirsin",
        "allows asking for personal data",
    )
    assert second.mutated == ""


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("mutants = 'x'\n", "expected \\[\\[mutants\\]\\] tables"),
        ("mutants = ['x']\n", "must be a table"),
        ("[[mutants]\n", "Cannot read"),
        (ENTRY.replace("[[mutants]]", "[[mutant]]"), "Unknown top-level key.*mutant"),
        (ENTRY + "other = 1\n", "Unknown key.*other"),
        (ENTRY.replace("description", "descripton"), "Unknown key.*descripton"),
        (ENTRY.replace('target = "agent:PROMPT"\n', ""), "missing 'target'"),
        (ENTRY.replace('description = "allows asking for personal data"', ""), "missing 'description'"),
        (ENTRY.replace('mutated = "Kişisel veriyi (TC, kart no) sorabilirsin"', ""), "missing 'mutated'"),
        (
            ENTRY.replace('original = "Asla kişisel veriyi (TC, kart no) sorma"', "original = 3"),
            "'original' must be a string",
        ),
        (ENTRY.replace('original = "Asla kişisel veriyi (TC, kart no) sorma"', 'original = " "'), "empty 'original'"),
    ],
)
def test_load_custom_mutants_rejects_malformed_files(tmp_path: Path, text: str, match: str) -> None:
    _write(tmp_path, text)
    with pytest.raises(pytest.UsageError, match=match):
        load_custom_mutants(tmp_path)


def test_load_custom_mutants_rejects_undecodable_file(tmp_path: Path) -> None:
    _write(tmp_path, "")
    (tmp_path / ".clastogen" / "mutants.toml").write_bytes(b"\xff")
    with pytest.raises(pytest.UsageError, match="Cannot read"):
        load_custom_mutants(tmp_path)


# No built-in rule keyword or digit in it: built-in discovery finds nothing in this prompt.
TURKISH_AGENT = '''
PROMPT = """Asla kişisel veriyi (TC, kart no) sorma.
Kibar ol ve nazik konuş.
Cevaplar kısa olsun."""

def ask():
    return PROMPT
'''

TURKISH_TEST = """
import pytest, agent

@pytest.mark.clastogen(target="agent:PROMPT", p0=0.9, max_mutants=3)
def test_never_asks_for_personal_data():
    assert "Asla kişisel veriyi" in agent.ask()
"""

TURKISH_TEST_B = TURKISH_TEST.replace("test_never_asks_for_personal_data", "test_second").replace(
    "Asla kişisel veriyi", "Cevaplar kısa"
)


def test_custom_mutants_run_deterministically_serial_and_parallel(
    pytester: pytest.Pytester, subprocess_env: None
) -> None:
    pytester.makepyfile(agent=TURKISH_AGENT, test_a=TURKISH_TEST, test_b=TURKISH_TEST_B)
    _write(pytester.path, ENTRY)
    runs = {}
    for name, args in (("serial", ()), ("parallel", ("-n", "2", "--dist", "loadfile"))):
        out = pytester.path / f"{name}.json"
        result = pytester.runpytest_subprocess("--clastogen", f"--clastogen-json={out}", *args)
        assert result.ret == pytest.ExitCode.OK, result.stdout.str()
        runs[name] = json.loads(out.read_text(encoding="utf-8"))

    for data in runs.values():
        (mutant,) = data["results"]
        assert (mutant["operator_name"], mutant["status"]) == ("custom", "KILLED")
        # The covered sentence is a rule with its own evidence and no longer unmutated.
        (rule,) = data["constraint_coverage"]["rules"]
        assert (rule["rule"], rule["status"], rule["killed"], rule["mutants"]) == (
            "Asla kişisel veriyi (TC, kart no) sorma",
            "killed",
            1,
            1,
        )
        sentences = {u["sentence"] for u in data["unmutated_sentences"]}
        assert sentences == {"Kibar ol ve nazik konuş", "Cevaplar kısa olsun."}
    assert runs["parallel"]["results"][0]["mutant_id"] == runs["serial"]["results"][0]["mutant_id"]


def test_a_stale_entry_fails_the_run_before_any_trial(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=TURKISH_AGENT, test_a=TURKISH_TEST)
    _write(pytester.path, ENTRY.replace("Asla kişisel", "Hep kişisel"))

    result = pytester.runpytest("--clastogen")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "Hep kişisel veriyi" in result.stderr.str()
    assert "NFC" in result.stderr.str()
    assert "must occur exactly once" in result.stderr.str()


def test_an_entry_for_an_unmarked_target_is_warned_about(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=TURKISH_AGENT, test_a=TURKISH_TEST)
    _write(pytester.path, ENTRY + ENTRY.replace("agent:PROMPT", "agent:OTHER"))

    out = pytester.runpytest("--clastogen").stdout.str()

    assert "Custom mutants for targets no test measured in this run: agent:OTHER (no marked" in out


def test_the_preview_counts_custom_mutants(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=TURKISH_AGENT, test_a=TURKISH_TEST)
    _write(pytester.path, ENTRY)

    out = pytester.runpytest("--clastogen", "--collect-only").stdout.str()

    assert "1 mutant runs" in out
    assert "At most 23 test calls" in out


def test_plan_test_counts_custom_mutants_in_the_calls_estimate() -> None:
    params = ClastogenParams("agent:PROMPT", max_mutants=3, p0=0.9)
    prompt = "Asla kişisel veriyi (TC, kart no) sorma.\nKibar ol ve nazik konuş."
    custom = (CustomMutant("agent:PROMPT", "Asla kişisel veriyi (TC, kart no) sorma", "Sor", "d"),)

    assert plan_test("t", params, prompt, set(), False, ()).calls == 0
    plan = plan_test("t", params, prompt, set(), False, custom)

    assert (plan.mutants, plan.calls) == (1, CANARY_RUNS + params.max_steps)
    assert (
        plan_test(
            "t",
            params,
            prompt,
            {m.id for m in PromptMutator(params.target, custom).generate_mutants(prompt, 3)},
            False,
            custom,
        ).mutants
        == 0
    )


def test_a_probe_labels_the_custom_mutant(pytester: pytest.Pytester, subprocess_env: None) -> None:
    probe = TURKISH_TEST.replace("p0=0.9, max_mutants=3", "max_mutants=3, oracle_samples=20").replace(
        "test_never_asks_for_personal_data", "test_probe"
    )
    pytester.makepyfile(agent=TURKISH_AGENT, test_a=TURKISH_TEST, test_p=probe)
    _write(pytester.path, ENTRY)
    out = pytester.path / "out.json"

    result = pytester.runpytest_subprocess("--clastogen", "--clastogen-oracle", f"--clastogen-json={out}")

    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    data = json.loads(out.read_text(encoding="utf-8"))
    (mutant,) = data["results"]
    assert data["behavior"]["labels"] == {mutant["mutant_id"]: "changed"}
    assert data["behavior"]["caught"] == 1


def test_k_selection_keeps_the_custom_mutant_in_the_plan(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=TURKISH_AGENT, test_a=TURKISH_TEST, test_b=TURKISH_TEST_B)
    _write(pytester.path, ENTRY)

    out = pytester.runpytest("--clastogen", "--collect-only", "-k", "test_second").stdout.str()

    assert "1 mutant runs" in out
    assert "At most 23 test calls" in out


def test_a_stale_entry_fails_clearly_under_xdist(pytester: pytest.Pytester, subprocess_env: None) -> None:
    pytester.makepyfile(agent=TURKISH_AGENT, test_a=TURKISH_TEST, test_b=TURKISH_TEST_B)
    _write(pytester.path, ENTRY.replace("Asla kişisel", "Hep kişisel"))

    result = pytester.runpytest_subprocess("--clastogen", "-n", "2", "--dist", "loadfile")

    out = result.stdout.str()
    assert result.ret == pytest.ExitCode.TESTS_FAILED, out
    assert "2 failed" in out  # every marked test of the target fails, not only the first
    assert "must occur exactly once" in out
    assert "INTERNALERROR" not in out


def test_no_marked_test_means_no_unmeasured_target_warning(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_plain="def test_ok():\n    pass\n")
    _write(pytester.path, ENTRY)

    out = pytester.runpytest("--clastogen").stdout.str()

    assert "no test measured" not in out


def test_collection_check_skips_unmarked_items_unusable_markers_and_repeated_targets(
    pytester: pytest.Pytester,
) -> None:
    bad = TURKISH_TEST.replace("test_never_asks_for_personal_data", "test_bad").replace("agent:PROMPT", "agent:MISSING")
    pytester.makepyfile(
        agent=TURKISH_AGENT,
        test_plain="def test_plain():\n    pass\n",
        test_a=TURKISH_TEST,
        test_b=TURKISH_TEST_B,
        test_bad=bad,
    )
    _write(pytester.path, ENTRY)

    result = pytester.runpytest("--clastogen")

    # The unusable marker fails its own test when it runs; it does not stop collection or hide the others.
    result.assert_outcomes(passed=3, failed=1)
    assert "must occur exactly once" not in result.stdout.str()


def test_a_probe_only_target_counts_as_measured(pytester: pytest.Pytester) -> None:
    probe = TURKISH_TEST.replace("p0=0.9, max_mutants=3", "max_mutants=3, oracle_samples=20")
    pytester.makepyfile(agent=TURKISH_AGENT, test_p=probe)
    _write(pytester.path, ENTRY)

    out = pytester.runpytest("--clastogen", "--clastogen-oracle").stdout.str()

    assert "no test measured" not in out


def test_a_suppressed_custom_mutant_does_not_suggest_raising_max_mutants(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(agent=TURKISH_AGENT, test_a=TURKISH_TEST)
    _write(pytester.path, ENTRY)
    prompt = TURKISH_AGENT.split('"""')[1]
    mutant_id = PromptMutator("agent:PROMPT", load_custom_mutants(pytester.path)).generate_mutants(prompt, 1)[0].id
    (pytester.path / ".clastogen" / "suppressions.toml").write_text(
        f'[[suppressions]]\nmutant_id = "{mutant_id}"\nreason = "equivalent"\n', encoding="utf-8"
    )

    out = pytester.runpytest("--clastogen").stdout.str()

    assert "untested: no measured mutant" in out
    assert "raise max_mutants" not in out
