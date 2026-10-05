import dataclasses
import sys
import types

import pytest

from clastogen import Mutant
from clastogen.core.injection import override_prompt
from clastogen.core.mutator import PromptMutator


def _register_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_avoid_rule_yields_delete_and_prefer_inversion() -> None:
    mutants = PromptMutator().generate_mutants("Avoid sharing account numbers with callers.", max_mutants=10)

    by_op = {m.operator_name: m for m in mutants}
    assert set(by_op) == {"delete_constraint", "invert_negation"}
    assert by_op["invert_negation"].mutated_snippet.startswith("PREFER")
    assert "Avoid" not in by_op["invert_negation"].mutated_prompt


def test_override_prompt_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_module = _register_module(monkeypatch, "dummy_agent", SYSTEM_PROMPT="Original safe prompt")

    with override_prompt("dummy_agent:SYSTEM_PROMPT", "Mutated dangerous prompt"):
        assert dummy_module.SYSTEM_PROMPT == "Mutated dangerous prompt"

    assert dummy_module.SYSTEM_PROMPT == "Original safe prompt"


def test_override_prompt_inherited_attribute_clean_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    class BaseAgent:
        PROMPT: str = "Base safe prompt"

    class ChildAgent(BaseAgent):
        pass

    assert "PROMPT" not in ChildAgent.__dict__
    assert ChildAgent.PROMPT == "Base safe prompt"

    _register_module(monkeypatch, "dummy_mod_child", Child=ChildAgent)
    with override_prompt("dummy_mod_child:Child.PROMPT", "Mutated prompt"):
        assert ChildAgent.PROMPT == "Mutated prompt"

    assert "PROMPT" not in ChildAgent.__dict__
    assert ChildAgent.PROMPT == "Base safe prompt"

    BaseAgent.PROMPT = "Updated base prompt"
    assert ChildAgent.PROMPT == "Updated base prompt"


def test_override_prompt_preserves_class_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    class AgentWithMethod:
        @classmethod
        def get_prompt(cls) -> str:
            return "Class method prompt"

    _register_module(monkeypatch, "dummy_mod_desc", Agent=AgentWithMethod)
    assert isinstance(AgentWithMethod.__dict__["get_prompt"], classmethod)

    with override_prompt("dummy_mod_desc:Agent.get_prompt", "Mutated string"):
        assert AgentWithMethod.get_prompt == "Mutated string"  # type: ignore[comparison-overlap]

    assert isinstance(AgentWithMethod.__dict__["get_prompt"], classmethod)
    assert AgentWithMethod.get_prompt() == "Class method prompt"


def test_override_prompt_nested_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_mod = _register_module(monkeypatch, "dummy_mod_nested", SYS_PROMPT="Level 0")

    with override_prompt("dummy_mod_nested:SYS_PROMPT", "Level 1"):
        assert dummy_mod.SYS_PROMPT == "Level 1"
        with override_prompt("dummy_mod_nested:SYS_PROMPT", "Level 2"):
            assert dummy_mod.SYS_PROMPT == "Level 2"
        assert dummy_mod.SYS_PROMPT == "Level 1"
    assert dummy_mod.SYS_PROMPT == "Level 0"


def test_prompt_mutator_preserves_abbreviations_and_cleans_dangling_dots() -> None:
    text = (
        "You are an assistant.\n"
        "- You must protect data, e.g. passwords and tokens.\n"
        "You never disclose confidential keys.\n"
        "Be helpful."
    )
    mutator = PromptMutator(target_symbol="PROMPT")
    rules = mutator.extract_candidate_rules(text)

    assert any("e.g. passwords and tokens" in r for r in rules)
    assert not any(r.startswith(("g.", ".g.")) for r in rules)

    mutants = mutator.generate_mutants(text, max_mutants=10)
    for m in mutants:
        assert "\n.\n" not in m.mutated_prompt
        assert "\n. " not in m.mutated_prompt


def test_prompt_mutator_samples_evenly_across_long_prompts() -> None:
    lines = [f"Rule {i}: You must never leak item {i}." for i in range(20)]
    long_prompt = "\n".join(lines)

    mutator = PromptMutator(target_symbol="LONG_PROMPT")
    mutants = mutator.generate_mutants(long_prompt, max_mutants=6)

    assert len(mutants) == 6
    descriptions = " ".join(m.description for m in mutants)
    assert any(f"item {i}" in descriptions for i in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19))


def test_override_prompt_patches_same_name_aliases_only(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "".join(["origin", "al"])
    agent = _register_module(monkeypatch, "xm_agent", PROMPT=prompt)
    importer = _register_module(monkeypatch, "xm_importer", PROMPT=prompt)
    renamed = _register_module(monkeypatch, "xm_renamed", SYSTEM=prompt)
    captured = {"config": {"system": prompt}}

    with override_prompt("xm_agent:PROMPT", "mutated"):
        assert agent.PROMPT == "mutated"
        assert importer.PROMPT == "mutated"
        assert renamed.SYSTEM == "original"
        assert captured["config"]["system"] == "original"

    assert agent.PROMPT == "original"
    assert importer.PROMPT == "original"
    assert importer.PROMPT is prompt


def test_override_prompt_none_target_raises_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_module(monkeypatch, "none_agent", PROMPT=None)
    with pytest.raises(TypeError, match="resolved to None"), override_prompt("none_agent:PROMPT", "mutated"):
        pass


def test_failed_override_leaves_target_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclasses.dataclass(frozen=True)
    class Frozen:
        system: str = "original"

    owner = Frozen()
    _register_module(monkeypatch, "frozen_target_mod", cfg=owner)
    with pytest.raises(dataclasses.FrozenInstanceError), override_prompt("frozen_target_mod:cfg.system", "mutated"):
        pass

    assert owner.system == "original"


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
