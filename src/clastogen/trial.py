"""Reruns a collected test as one trial on fresh fixtures.

This is the only module that touches pytest internals (`Function._request`, `TopRequest._fixture_defs`,
`tmppath_result_key`, `caplog_handler_key`, `TestCaseFunction`): pytest offers no public way to re-execute function-scoped fixtures, and pytest-rerunfailures
relies on the same private API.
"""

import pytest
from _pytest.logging import caplog_handler_key
from _pytest.tmpdir import tmppath_result_key
from _pytest.unittest import TestCaseFunction

from clastogen.exceptions import TrialError, TrialSkipped


def _reset_function_fixtures(item: pytest.Item) -> None:
    """Tears down and re-executes all function-scoped fixtures for honest, isolated trials."""
    # unittest reports failures to its result object instead of raising them, so every trial would pass.
    if not isinstance(item, pytest.Function) or isinstance(item, TestCaseFunction):
        raise TypeError(f"Cannot reset fixtures for '{item.nodeid}': only Python test functions are supported.")
    req = item._request
    fixture_defs = req._fixture_defs
    defs = list(fixture_defs.items())
    try:
        for name, fixturedef in defs:
            if fixturedef.scope == "function":
                # tmp_path teardown reads this key, which pytest only sets after the call phase finishes.
                item.stash.setdefault(tmppath_result_key, {})
                fixturedef.finish(request=req)  # ty: ignore[invalid-argument-type]  # finish only reads request.node, which TopRequest has
                del fixture_defs[name]
                new_val = req.getfixturevalue(name)
                if name in item.funcargs:
                    item.funcargs[name] = new_val
    except BaseException:
        # Restoring the setup order makes the next trial rebuild each fixture before the fixtures that use it.
        fixture_defs.clear()
        fixture_defs.update(defs)
        raise
    # caplog reads this handler, which pytest resets once per phase; clearing after the loop drops the fixtures' own logs.
    if (caplog_handler := item.stash.get(caplog_handler_key, None)) is not None:
        caplog_handler.clear()


def run_trial(item: pytest.Item) -> bool:
    """Reruns the test on fresh fixtures: assertion failures are False, anything else unexpected is a TrialError."""
    try:
        _reset_function_fixtures(item)
    except pytest.exit.Exception:
        raise
    except (pytest.skip.Exception, pytest.xfail.Exception) as exc:
        raise TrialSkipped from exc
    except (Exception, pytest.fail.Exception, SystemExit) as exc:
        raise TrialError(f"fixture reset failed: {type(exc).__name__}") from exc
    try:
        item.runtest()
    except pytest.xfail.Exception as exc:
        raise TrialSkipped from exc
    except (AssertionError, pytest.fail.Exception):
        return False
    except pytest.skip.Exception as exc:
        raise TrialSkipped from exc
    except pytest.exit.Exception:
        raise
    except (Exception, SystemExit) as exc:
        raise TrialError(f"unexpected execution error: {type(exc).__name__}") from exc
    return True
