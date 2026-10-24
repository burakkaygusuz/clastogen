# ruff: noqa: RUF001
from __future__ import annotations

import pytest

from clastogen.models import CUSTOM_OPERATOR, CustomMutant, Mutant, MutantStatus
from clastogen.mutation.mutator import PromptMutator
from clastogen.reporting.scoring import summarize
from clastogen.runner import _execution

TURKISH = "Asla kişisel veriyi (TC, kart no) sorma.\nKibar ol ve nazik konuş.\nCevaplar kısa olsun."
PERSONAL = CustomMutant(
    "T", "Asla kişisel veriyi (TC, kart no) sorma", "Kişisel veriyi (TC, kart no) sorabilirsin", "asks"
)
ENGLISH = "You must never approve refunds over $50.\nYou must always verify identity.\nKeep it short and friendly."


def _gen(prompt: str, custom: list[CustomMutant], max_mutants: int = 5) -> list[Mutant]:
    return list(PromptMutator("T", custom).generate_mutants(prompt, max_mutants))


def test_a_prompt_without_builtin_matches_still_gets_a_custom_mutant() -> None:
    assert PromptMutator("T").generate_mutants(TURKISH, 5) == []

    (mutant,) = _gen(TURKISH, [PERSONAL])

    assert mutant.operator_name == CUSTOM_OPERATOR
    assert mutant.mutated_prompt == TURKISH.replace(PERSONAL.original, PERSONAL.mutated)
    assert mutant.description == "asks"


def test_custom_mutants_come_first_and_ignore_other_targets() -> None:
    other = CustomMutant("OTHER", "You must always verify identity", "You may verify identity", "x")

    mutants = _gen(ENGLISH, [other, CustomMutant("T", "Keep it short", "Ramble", "long")])

    assert all(m.operator_name == CUSTOM_OPERATOR for m in mutants[:1])
    assert "Ramble" in mutants[0].mutated_prompt
    assert all(m.target_symbol == "T" for m in mutants)


def test_max_mutants_caps_only_the_builtin_mutants() -> None:
    custom = [CustomMutant("T", s, s.upper(), s) for s in ("Keep it short", "and friendly", "verify identity")]

    for cap in (1, 3):
        mutants = _gen(ENGLISH, custom, max_mutants=cap)

        assert [m.operator_name == CUSTOM_OPERATOR for m in mutants] == [True] * 3 + [False] * cap


def test_identical_custom_entries_and_a_builtin_with_the_same_prompt_count_once() -> None:
    builtin = PromptMutator("T").generate_mutants(ENGLISH, 5)[0]
    same_as_builtin = CustomMutant("T", builtin.original_snippet, builtin.mutated_snippet, "dup")

    mutants = _gen(ENGLISH, [same_as_builtin, same_as_builtin])

    assert len({m.mutated_prompt for m in mutants}) == len(mutants)
    assert [m.operator_name for m in mutants].count(CUSTOM_OPERATOR) == 1
    assert builtin.id not in {m.id for m in mutants}


def test_an_empty_mutated_text_deletes_the_snippet() -> None:
    (mutant,) = _gen(TURKISH, [CustomMutant("T", " ve nazik konuş", "", "drops")])

    assert mutant.mutated_prompt == TURKISH.replace(" ve nazik konuş", "")
    assert mutant.mutated_snippet == "[DELETED]"


def test_the_id_is_stable_and_hashes_the_texts() -> None:
    a = _gen(TURKISH, [PERSONAL])[0]
    b = _gen(TURKISH, [PERSONAL])[0]
    c = _gen(TURKISH, [CustomMutant("T", PERSONAL.original, "Kişisel veriyi sor", "x")])[0]

    assert a.id == b.id != c.id


@pytest.mark.parametrize(
    ("original", "mutated", "match"),
    [
        ("Asla bir şey", "Belki", "exactly once"),
        ("Kibar ol", "Kibar ol", "changes nothing"),
    ],
)
def test_a_stale_or_identical_entry_raises_naming_it(original: str, mutated: str, match: str) -> None:
    with pytest.raises(ValueError, match=match) as err:
        _gen(TURKISH, [CustomMutant("T", original, mutated, "d")])
    assert original in str(err.value)


def test_a_covered_sentence_leaves_the_unmutated_list_and_becomes_a_rule() -> None:
    plain = PromptMutator("T")
    covered = PromptMutator("T", [PERSONAL])
    sentence = "Asla kişisel veriyi (TC, kart no) sorma"

    assert sentence in plain.extract_unmatched_sentences(TURKISH)
    assert sentence not in covered.extract_unmatched_sentences(TURKISH)
    assert sentence in covered.extract_candidate_rules(TURKISH)
    assert plain.extract_candidate_rules(TURKISH) == []
    # A snippet inside a sentence covers that sentence too.
    inside = PromptMutator("T", [CustomMutant("T", "nazik konuş", "sert konuş", "d")])
    assert "Kibar ol ve nazik konuş" not in inside.extract_unmatched_sentences(TURKISH)


def test_rule_coverage_credits_a_custom_mutant_to_the_sentence_it_sits_in() -> None:
    mutator = PromptMutator("T", [PERSONAL])
    (mutant,) = mutator.generate_mutants(TURKISH, 5)
    rules = [("T", r) for r in mutator.extract_candidate_rules(TURKISH)]
    killed = _execution("t::a", mutant, MutantStatus.KILLED)

    summary = summarize([killed], rules)

    (row,) = summary.rules
    assert (row.rule, row.killed, row.mutants, row.killed_by) == (
        "Asla kişisel veriyi (TC, kart no) sorma",
        1,
        1,
        "t::a",
    )


def test_an_original_that_does_not_occur_exactly_once_raises() -> None:
    twice = "Kibar ol.\nHer zaman kibar ol."
    in_example = "Kibar ol ve nazik konuş.\n<example>Kibar ol ve nazik konuş.</example>"

    for prompt, original in ((twice, "ibar ol"), (in_example, "nazik konuş"), (twice, "ol")):
        with pytest.raises(ValueError, match=r"exactly once.*found 2"):
            _gen(prompt, [CustomMutant("T", original, "x", "d")])
    with pytest.raises(ValueError, match=r"found 0.*NFC"):
        _gen(twice, [CustomMutant("T", "kaba ol", "x", "d")])


def test_a_custom_mutant_equal_to_another_makes_one_mutant_and_beats_a_builtin() -> None:
    prompt = "Kibar ol."

    mutants = _gen(prompt, [CustomMutant("T", "Kibar ol", "Kaba ol", "a"), CustomMutant("T", "Kibar", "Kaba", "b")])

    assert [(m.operator_name, m.description) for m in mutants] == [(CUSTOM_OPERATOR, "a")]


def test_rule_coverage_credits_a_snippet_inside_a_sentence_but_not_one_spanning_two() -> None:
    inside = PromptMutator("T", [CustomMutant("T", "nazik konuş", "sert konuş", "d")])
    spanning = PromptMutator("T", [CustomMutant("T", "Kibar ol ve nazik konuş.\nCevaplar", "Kaba ol\nCevaplar", "d")])
    for mutator, credited in ((inside, True), (spanning, False)):
        (mutant,) = mutator.generate_mutants(TURKISH, 5)
        rules = [("T", r) for r in mutator.extract_candidate_rules(TURKISH)]

        summary = summarize(
            [_execution("t::a", mutant, MutantStatus.KILLED)], [("T", "Kibar ol ve nazik konuş"), *rules]
        )

        row = next(r for r in summary.rules if r.rule == "Kibar ol ve nazik konuş")
        assert (row.killed, row.mutants) == ((1, 1) if credited else (0, 0))
        assert ("Kibar ol ve nazik konuş" in mutator.extract_unmatched_sentences(TURKISH)) is not credited


def test_a_sentence_inside_an_example_block_is_not_covered() -> None:
    prompt = "Cevaplar kısa olsun.\n<example>\nKibar ol ve nazik konuş.\n</example>"
    mutator = PromptMutator("T", [CustomMutant("T", "nazik konuş", "sert konuş", "d")])

    assert mutator.extract_candidate_rules(prompt) == []
    assert "Kibar ol ve nazik konuş" not in mutator.extract_unmatched_sentences(prompt)
