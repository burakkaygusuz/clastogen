import tomllib
from pathlib import Path

import pytest


def read_toml_tables(path: Path, key: str, fields: tuple[str, ...]) -> list[dict[str, object]]:
    """Reads the [[key]] tables of a TOML file; [] when the file is missing.

    Raises pytest.UsageError for an unreadable file, a top-level key other than key, a key that is not a list of
    tables, and an entry key that is not in fields, so a typo like [[mutant]] or `reasn` fails loudly.
    """
    if not path.is_file():
        return []
    try:
        raw_data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise pytest.UsageError(f"Cannot read '{path}': {exc}") from exc
    if unknown := sorted(set(raw_data) - {key}):
        raise pytest.UsageError(
            f"Unknown top-level key(s) in '{path}': {', '.join(unknown)}; expected [[{key}]] tables."
        )
    items = raw_data.get(key, [])
    if not isinstance(items, list):
        raise pytest.UsageError(f"Invalid format in '{path}': expected [[{key}]] tables.")
    for entry in items:
        if not isinstance(entry, dict):
            raise pytest.UsageError(f"Entry {entry!r} in '{path}' must be a table with {', '.join(fields)}.")
        if unknown := sorted(set(entry) - set(fields)):
            raise pytest.UsageError(
                f"Unknown key(s) {', '.join(unknown)} in entry of '{path}': {entry}; allowed: {', '.join(fields)}."
            )
    return items
