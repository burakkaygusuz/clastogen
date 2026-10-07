from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from functools import reduce
from unittest.mock import patch

logger = logging.getLogger(__name__)


def resolve_target(target_spec: str) -> tuple[object, str]:
    """Resolves 'module:attr' or 'module:Class.attr' (dots after the colon walk nested attributes) into (owner, attr)."""
    module_name, sep, attr_path = target_spec.partition(":")
    if not sep:
        raise ValueError(f"Target '{target_spec}' must be 'module:attr' or 'module:Class.attr'")
    *owner_path, attr = attr_path.split(".")
    owner = reduce(getattr, owner_path, importlib.import_module(module_name))
    if not hasattr(owner, attr):
        raise AttributeError(f"Object {owner!r} has no attribute '{attr}'")
    return owner, attr


def _aliasing_modules(owner: object, attr: str, original: object) -> list[types.ModuleType]:
    """Modules other than owner that bound the same string under the same name, e.g. via `from agent import PROMPT`."""
    if not isinstance(original, str):
        return []
    return [
        mod
        for mod in list(sys.modules.values())
        if isinstance(mod, types.ModuleType) and mod is not owner and vars(mod).get(attr) is original
    ]


@contextmanager
def override_prompt(target_spec: str, mutated_text: str) -> Generator[str, None, None]:
    """Temporarily replaces the target attribute, and every same-name module alias of it, with mutated_text.

    unittest.mock.patch.object restores descriptors such as @classmethod and removes shadow attributes
    on classes that only inherited the target.
    """
    if not isinstance(mutated_text, str):
        raise TypeError(f"mutated_text must be a string, got {type(mutated_text).__name__}")

    owner, attr = resolve_target(target_spec)
    original = getattr(owner, attr)
    if original is None:
        raise TypeError(f"Target '{target_spec}' resolved to None, expected a prompt string.")

    with ExitStack() as stack:
        stack.enter_context(patch.object(owner, attr, mutated_text))
        aliases = _aliasing_modules(owner, attr, original)
        for mod in aliases:
            stack.enter_context(patch.object(mod, attr, mutated_text))
        logger.debug("Injected prompt into target '%s' (%d module alias(es))", target_spec, len(aliases))
        yield mutated_text
