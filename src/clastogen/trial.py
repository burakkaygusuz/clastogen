"""Reruns a collected test as one trial on fresh fixtures.

This is the only module that touches pytest internals (`Function._request`, `TopRequest._fixture_defs`,
`tmppath_result_key`): pytest offers no public way to re-execute function-scoped fixtures, and pytest-rerunfailures
relies on the same private API.
"""

import pytest
from _pytest.tmpdir import tmppath_result_key

from clastogen.exceptions import TrialError, TrialSkipped


def _reset_function_fixtures(item: pytest.Item) -> None:
    """Tears down and re-executes all function-scoped fixtures for honest, isolated trials."""
    if not isinstance(item, pytest.Function):
        raise RuntimeError(f"Cannot reset fixtures for '{item.nodeid}': only Python test functions are supported.")
    req = item._request
    fixture_defs = req._fixture_defs
    for name, fixturedef in list(fixture_defs.items()):
        if fixturedef.scope == "function":
            # tmp_path teardown reads this key, which pytest only sets after the call phase finishes.
            item.stash.setdefault(tmppath_result_key, {})
            fixturedef.finish(request=req)  # ty: ignore[invalid-argument-type]  # finish only reads request.node, which TopRequest has
            del fixture_defs[name]
            new_val = req.getfixturevalue(name)
            if name in item.funcargs:
                item.funcargs[name] = new_val


def run_trial(item: pytest.Item) -> bool:
    """Reruns the test on fresh fixtures: assertion failures are False, anything else unexpected is a TrialError."""
    try:
        _reset_function_fixtures(item)
    except Exception as exc:
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
    except Exception as exc:
        raise TrialError(f"unexpected execution error: {type(exc).__name__}") from exc
    return True
