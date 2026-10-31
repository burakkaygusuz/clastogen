from pathlib import Path

import pytest

from clastogen.models import CustomMutant
from clastogen.toml_tables import read_toml_tables

CUSTOM_MUTANTS_PATH = Path(".clastogen") / "mutants.toml"
_FIELDS = ("target", "original", "mutated", "description")


def load_custom_mutants(root_path: Path) -> tuple[CustomMutant, ...]:
    """Parses <rootdir>/.clastogen/mutants.toml; every entry needs string target, original, mutated and description."""
    candidate = root_path / CUSTOM_MUTANTS_PATH
    return tuple(_parse_entry(candidate, entry) for entry in read_toml_tables(candidate, "mutants", _FIELDS))


def _parse_entry(candidate: Path, entry: dict[str, object]) -> CustomMutant:
    for name in _FIELDS:
        if name not in entry:
            raise pytest.UsageError(f"Mutant entry in '{candidate}' missing '{name}': {entry}")
        if not isinstance(entry[name], str):
            raise pytest.UsageError(f"Mutant entry in '{candidate}': '{name}' must be a string: {entry}")
    # An empty mutated deletes the snippet; every other field must say something.
    for name in ("target", "original", "description"):
        if not str(entry[name]).strip():
            raise pytest.UsageError(f"Mutant entry in '{candidate}' has an empty '{name}': {entry}")
    return CustomMutant(**{name: str(entry[name]) for name in _FIELDS})
