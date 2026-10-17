import importlib
from pathlib import Path

import pytest

import clastogen
import clastogen.exceptions
import clastogen.models
import clastogen.mutation.injection
import clastogen.mutation.mutator
import clastogen.mutation.structure
import clastogen.plugin
import clastogen.reporting.render
import clastogen.reporting.scoring
import clastogen.runner
import clastogen.stats.assertions
import clastogen.stats.sprt
import clastogen.trial

pytest_plugins = ["pytester"]

# The plugin loads via the pytest11 entry point before pytest-cov starts; reload so module-level lines are measured.
for mod in [
    clastogen.exceptions,
    clastogen.models,
    clastogen.reporting.scoring,
    clastogen.reporting.render,
    clastogen.mutation.structure,
    clastogen.mutation.mutator,
    clastogen.stats.sprt,
    clastogen.mutation.injection,
    clastogen.trial,
    clastogen.stats.assertions,
    clastogen.runner,
    clastogen.plugin,
    clastogen,
]:
    importlib.reload(mod)


@pytest.fixture
def subprocess_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment for pytester.runpytest_subprocess: the source tree on the path, UTF-8 for the report's non-ASCII text."""
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setenv("PYTHONUTF8", "1")
