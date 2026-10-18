import tomllib
from pathlib import Path

import pytest

SUPPRESSIONS_PATH = Path(".clastogen") / "suppressions.toml"


def load_suppressions(root_path: Path) -> set[str]:
    """Parses suppressed mutant IDs from <rootdir>/.clastogen/suppressions.toml; every entry needs a reason."""
    candidate = root_path / SUPPRESSIONS_PATH
    if not candidate.is_file():
        return set()
    try:
        raw_data = tomllib.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise pytest.UsageError(f"Cannot read '{candidate}': {exc}") from exc
    items = raw_data.get("suppressions", [])
    if not isinstance(items, list):
        raise pytest.UsageError(f"Invalid suppression format in '{candidate}': expected [[suppressions]] tables.")

    parsed: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            raise pytest.UsageError(
                f"Suppression entry {entry!r} in '{candidate}' must be a table with mutant_id and reason."
            )
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
