from pathlib import Path

import pytest

from clastogen.toml_tables import read_toml_tables

SUPPRESSIONS_PATH = Path(".clastogen") / "suppressions.toml"


def load_suppressions(root_path: Path) -> set[str]:
    """Parses suppressed mutant IDs from <rootdir>/.clastogen/suppressions.toml; every entry needs a reason."""
    candidate = root_path / SUPPRESSIONS_PATH
    if not candidate.is_file():
        return set()
    items = read_toml_tables(candidate, "suppressions", ("mutant_id", "reason"))

    parsed: set[str] = set()
    for entry in items:
        m_id = str(entry.get("mutant_id") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if not m_id:
            raise pytest.UsageError(f"Suppression entry in '{candidate}' missing 'mutant_id': {entry}")
        if not reason:
            raise pytest.UsageError(
                f"Suppression for mutant '{m_id}' in '{candidate}' must specify a non-empty 'reason'."
            )
        parsed.add(m_id)
    return parsed
