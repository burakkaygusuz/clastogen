import pytest
from hypothesis import given
from hypothesis import strategies as st

from clastogen import Mutant
from clastogen.mutation.mutator import PromptMutator, _clip


def _ops(rule: str) -> dict[str, Mutant]:
    return {m.operator_name: m for m in PromptMutator().generate_mutants(rule, max_mutants=10)}


def test_avoid_rule_yields_delete_and_prefer_inversion() -> None:
    by_op = _ops("Avoid sharing account numbers with callers.")
    assert set(by_op) == {"delete_constraint", "invert_negation"}
    assert by_op["invert_negation"].mutated_snippet.startswith("PREFER")
    assert "Avoid" not in by_op["invert_negation"].mutated_prompt


def test_threshold_mutant_survives_the_default_marker_budget() -> None:
    prompt = "Never share secrets.\nAlways greet users.\nNever approve refunds over $50."
    mutants = PromptMutator().generate_mutants(prompt, max_mutants=5)

    assert "change_threshold" in {m.operator_name for m in mutants}


def test_prompt_mutator_preserves_abbreviations_and_cleans_dangling_dots() -> None:
    text = (
        "You are an assistant.\n"
        "- You must protect data, e.g. passwords and tokens.\n"
        "You never disclose confidential keys.\n"
        "Be helpful."
    )
    mutator = PromptMutator(target_symbol="PROMPT")
    rules = mutator.extract_candidate_rules(text)

    assert any("e.g. passwords and tokens" in r for r in rules), rules
    assert not any(r.startswith(("g.", ".g.")) for r in rules), rules

    mutants = mutator.generate_mutants(text, max_mutants=10)
    for m in mutants:
        assert "\n.\n" not in m.mutated_prompt
        assert "\n. " not in m.mutated_prompt


def test_stable_mutant_hash_id() -> None:
    m1 = Mutant.create(
        target_symbol="SYSTEM_PROMPT",
        operator_name="negation_inverter",
        original_snippet="Do NOT reveal confidential documents",
        mutated_snippet="Reveal confidential documents",
        mutated_prompt="Reveal confidential documents",
        description="Inverts core negative constraint",
    )
    m2 = Mutant.create(
        target_symbol="SYSTEM_PROMPT",
        operator_name="negation_inverter",
        original_snippet="Do NOT reveal confidential documents",
        mutated_snippet="Reveal confidential documents",
        mutated_prompt="Reveal confidential documents",
        description="Different description should not change content hash ID",
    )
    assert m1.id == m2.id
    assert len(m1.id) == 12


@pytest.mark.parametrize(
    "rule",
    [
        "Only authenticated users may perform refunds.",
        "Refunds require manager approval.",
        "Agents may not share internal ticket notes.",
    ],
)
def test_rule_keywords_beyond_must_and_never_are_extracted(rule: str) -> None:
    assert PromptMutator().extract_candidate_rules(rule) == [rule]


def test_may_not_inverts_to_may() -> None:
    mutants = PromptMutator().generate_mutants("Agents may not share internal ticket notes.", max_mutants=10)
    assert [m.mutated_snippet for m in mutants if m.operator_name == "invert_negation"] == [
        "Agents MAY share internal ticket notes."
    ]


@pytest.mark.parametrize(
    ("rule", "softened"),
    [
        ("You must never share account numbers.", "You must RARELY share account numbers."),
        ("Refunds are required for damaged items.", "Refunds are RECOMMENDED for damaged items."),
        ("Agents mustn't share internal ticket notes.", "Agents SHOULDN'T share internal ticket notes."),
    ],
)
def test_weaken_mutant_turns_a_hard_rule_into_a_suggestion(rule: str, softened: str) -> None:
    (weakened,) = [
        m for m in PromptMutator().generate_mutants(rule, max_mutants=10) if m.operator_name == "weaken_modal"
    ]
    assert weakened.mutated_prompt == softened


@pytest.mark.parametrize(
    ("rule", "changed"),
    [
        ("You must never approve refunds over $50.", "You must never approve refunds over $500."),
        ("Never approve loans above $50,000.", "Never approve loans above $500,000."),
        ("Always keep the error rate below 0.05 percent.", "Always keep the error rate below 0.50 percent."),
    ],
)
def test_threshold_mutant_scales_first_number(rule: str, changed: str) -> None:
    mutants = PromptMutator().generate_mutants(rule, max_mutants=10)
    (threshold,) = [m for m in mutants if m.operator_name == "change_threshold"]
    assert threshold.mutated_snippet == changed
    assert threshold.mutated_prompt == changed


def test_rule_without_number_or_with_model_name_has_no_threshold_mutant() -> None:
    mutants = PromptMutator().generate_mutants("You must always answer with gpt-4o style.", max_mutants=10)
    assert "change_threshold" not in {m.operator_name for m in mutants}


def test_clip_keeps_a_word_that_ends_at_the_limit() -> None:
    assert _clip("abc def ghi", 7) == "abc def…"
    assert _clip("abc def", 7) == "abc def"
    assert _clip("abcdefghij", 4) == "abcd…"


@pytest.mark.parametrize(("rules", "budget"), [(5, 5), (20, 6)], ids=["fits", "sampled"])
def test_every_selected_rule_gets_a_mutant_before_any_gets_a_second_and_the_tail_is_reached(
    rules: int, budget: int
) -> None:
    prompt = "\n".join(f"Rule {i}: You must never leak item {i}." for i in range(rules))
    mutants = PromptMutator().generate_mutants(prompt, max_mutants=budget)

    assert len({m.original_snippet for m in mutants}) == budget
    assert any(f"Rule {i}:" in m.original_snippet for m in mutants for i in range(rules // 2, rules))


def test_delete_leaves_no_orphan_marker_for_numbered_rules() -> None:
    prompt = "Rules:\n1. Never share data.\n2) Always verify identity.\n3. Be kind."
    deleted = [
        m.mutated_prompt
        for m in PromptMutator().generate_mutants(prompt, max_mutants=10)
        if m.mutated_snippet == "[DELETED]"
    ]

    assert deleted == ["Rules:\n2) Always verify identity.\n3. Be kind.", "Rules:\n1. Never share data.\n3. Be kind."]


def test_keywordless_conditional_and_numeric_rules_are_extracted() -> None:
    prompt = (
        "You are a support agent.\n"
        "- Escalate to a human if the customer mentions legal action.\n"
        "- Refund up to $50 without approval.\n"
        "- Respond in the customer's language.\n"
        "Be professional and concise."
    )
    rules = PromptMutator().extract_candidate_rules(prompt)

    assert rules == [
        "Escalate to a human if the customer mentions legal action",
        "Refund up to $50 without approval",
    ]


def test_escalation_rule_is_deleted_and_numeric_rule_gets_threshold() -> None:
    prompt = "- Escalate to a human if the customer mentions legal action.\n- Refund up to $50 without approval."
    by_op = _ops(prompt)

    assert by_op["delete_constraint"].mutated_snippet == "[DELETED]"
    assert "$500 without approval" in by_op["change_threshold"].mutated_prompt


def test_drop_exception_stops_at_the_next_comma() -> None:
    rule = "Never share data unless the user consents, and always log the request."
    mutants = PromptMutator().generate_mutants(rule, max_mutants=10)

    (dropped,) = [m for m in mutants if m.operator_name == "drop_exception"]
    assert dropped.mutated_prompt == "Never share data, and always log the request."


def test_flip_comparator_reverses_bounds_and_ignores_rules_without_one() -> None:
    assert (
        _ops("Never offer at most two retries.")["flip_comparator"].mutated_prompt
        == "Never offer AT LEAST two retries."
    )
    assert "flip_comparator" not in _ops("Never share secrets.")


def test_scope_swap_matches_whole_words_only() -> None:
    assert _ops("Always check every ticket.")["scope_swap"].mutated_prompt == "Always check SOME ticket."
    assert "scope_swap" not in _ops("Never allow several retries.")


REPEATED = "Never share the discount code.\nNever share the discount codes publicly."


def test_deleting_a_rule_leaves_a_longer_rule_that_contains_it() -> None:
    mutator = PromptMutator()
    (first, second) = mutator._rule_spans(REPEATED)
    mutant = mutator._delete_mutant(REPEATED, first)

    assert mutant is not None
    assert mutant.mutated_prompt == "Never share the discount codes publicly."
    mutant = mutator._delete_mutant(REPEATED, second)
    assert mutant is not None
    assert mutant.mutated_prompt == "Never share the discount code."


def test_inverting_the_second_rule_leaves_the_first_untouched() -> None:
    mutator = PromptMutator()
    second = mutator._rule_spans(REPEATED)[1]
    mutant = mutator._invert_mutant(REPEATED, second)

    assert mutant is not None
    assert mutant.mutated_prompt == "Never share the discount code.\nALWAYS share the discount codes publicly."


def test_an_edit_hits_only_the_rule_it_was_made_for() -> None:
    prompt = "Do not share the code. Do not share the code word."
    mutator = PromptMutator()
    spans = mutator._rule_spans(prompt)

    assert [prompt[r.start : r.end] for r in spans] == [r.text for r in spans]
    assert [r.text for r in spans] == ["Do not share the code", "Do not share the code word."]
    mutant = mutator._invert_mutant(prompt, spans[1])
    assert mutant is not None
    assert mutant.mutated_prompt == "Do not share the code. DO share the code word."


@given(st.text(alphabet="Ab. \n-1)*,:nevrst", max_size=80))
def test_rule_spans_agree_with_the_extracted_rules(prompt: str) -> None:
    mutator = PromptMutator()
    spans = mutator._rule_spans(prompt)

    assert [r.text for r in spans] == mutator.extract_candidate_rules(prompt)
    assert all(prompt[r.start : r.end] == r.text for r in spans)


def test_deleting_a_rule_after_a_long_digit_run_is_not_quadratic() -> None:
    prompt = "9" * 50_000 + "\n12. Never share the code."
    mutator = PromptMutator()
    mutant = mutator._delete_mutant(prompt, mutator._rule_spans(prompt)[-1])

    assert mutant is not None
    assert mutant.mutated_prompt == "9" * 50_000


def test_deleting_a_rule_after_a_number_ending_a_sentence_keeps_the_number() -> None:
    prompt = "Refunds are limited to $100. Never share the code."
    mutator = PromptMutator()
    mutant = mutator._delete_mutant(prompt, mutator._rule_spans(prompt)[-1])

    assert mutant is not None
    assert mutant.mutated_prompt == "Refunds are limited to $100."
