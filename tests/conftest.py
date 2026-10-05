import importlib

import clastogen
import clastogen.core.injection
import clastogen.core.mutator
import clastogen.core.sprt
import clastogen.exceptions
import clastogen.models
import clastogen.plugin
import clastogen.report
import clastogen.scoring
import clastogen.stats
import clastogen.types

pytest_plugins = ["pytester"]

# The plugin loads via the pytest11 entry point before pytest-cov starts; reload so module-level lines are measured.
for mod in [
    clastogen.types,
    clastogen.exceptions,
    clastogen.models,
    clastogen.scoring,
    clastogen.report,
    clastogen.core.mutator,
    clastogen.core.sprt,
    clastogen.core.injection,
    clastogen.stats,
    clastogen.plugin,
    clastogen,
]:
    importlib.reload(mod)
