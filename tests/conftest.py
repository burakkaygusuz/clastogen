import importlib

import clastogen
import clastogen.exceptions
import clastogen.models
import clastogen.mutation.injection
import clastogen.mutation.mutator
import clastogen.plugin
import clastogen.reporting.render
import clastogen.reporting.scoring
import clastogen.stats.assertions
import clastogen.stats.sprt
import clastogen.types

pytest_plugins = ["pytester"]

# The plugin loads via the pytest11 entry point before pytest-cov starts; reload so module-level lines are measured.
for mod in [
    clastogen.types,
    clastogen.exceptions,
    clastogen.models,
    clastogen.reporting.scoring,
    clastogen.reporting.render,
    clastogen.mutation.mutator,
    clastogen.stats.sprt,
    clastogen.mutation.injection,
    clastogen.stats.assertions,
    clastogen.plugin,
    clastogen,
]:
    importlib.reload(mod)
