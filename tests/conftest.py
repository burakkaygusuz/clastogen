import importlib

import clastogen
import clastogen.core.injection
import clastogen.core.mutator
import clastogen.core.sprt
import clastogen.exceptions.trial
import clastogen.models.mutation
import clastogen.models.plugin
import clastogen.models.sprt
import clastogen.models.stats
import clastogen.plugin
import clastogen.report
import clastogen.scoring
import clastogen.types.enums
import clastogen.types.payloads

pytest_plugins = ["pytester"]

# The plugin loads via the pytest11 entry point before pytest-cov starts; reload so module-level lines are measured.
for mod in [
    clastogen.types.enums,
    clastogen.types.payloads,
    clastogen.types,
    clastogen.exceptions.trial,
    clastogen.exceptions,
    clastogen.models.mutation,
    clastogen.models.sprt,
    clastogen.models.stats,
    clastogen.models.plugin,
    clastogen.models,
    clastogen.scoring,
    clastogen.report,
    clastogen.core.mutator,
    clastogen.core.sprt,
    clastogen.core.injection,
    clastogen.plugin,
    clastogen,
]:
    importlib.reload(mod)
