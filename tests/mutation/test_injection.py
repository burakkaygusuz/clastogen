import dataclasses
import sys
import types

import pytest
from pytest_check import check

from clastogen.mutation.injection import override_prompt


def _register_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


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
        assert AgentWithMethod.get_prompt == "Mutated string"

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


def test_override_prompt_patches_same_name_aliases_only(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "".join(["origin", "al"])
    agent = _register_module(monkeypatch, "xm_agent", PROMPT=prompt)
    importer = _register_module(monkeypatch, "xm_importer", PROMPT=prompt)
    renamed = _register_module(monkeypatch, "xm_renamed", SYSTEM=prompt)
    captured = {"config": {"system": prompt}}

    with override_prompt("xm_agent:PROMPT", "mutated"):
        check.equal(agent.PROMPT, "mutated")
        check.equal(importer.PROMPT, "mutated")
        check.equal(renamed.SYSTEM, "original")
        check.equal(captured["config"]["system"], "original")

    check.equal(agent.PROMPT, "original")
    check.is_(importer.PROMPT, prompt)


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
