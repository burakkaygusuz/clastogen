from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def subprocess_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment for pytester.runpytest_subprocess: the source tree on the path, UTF-8 for the report's non-ASCII text."""
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setenv("PYTHONUTF8", "1")
