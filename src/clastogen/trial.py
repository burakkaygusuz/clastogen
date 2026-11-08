"""Reruns a collected test as one trial on fresh fixtures.

This is the only module that touches pytest internals (`Function._request`, `TopRequest._fixture_defs`,
`tmppath_result_key`, `caplog_handler_key`, `TestCaseFunction`): pytest offers no public way to re-execute function-scoped fixtures, and pytest-rerunfailures
relies on the same private API.
"""

import sys
from collections.abc import Callable

import pytest
from _pytest.logging import caplog_handler_key
from _pytest.tmpdir import tmppath_result_key
from _pytest.unittest import TestCaseFunction

from clastogen.exceptions import TrialError, TrialSkipped


def _reset_function_fixtures(item: pytest.Item) -> None:
    """Tears down and re-executes all function-scoped fixtures for honest, isolated trials."""
    if not isinstance(item, pytest.Function) or isinstance(item, TestCaseFunction):
        raise TypeError(f"Cannot reset fixtures for '{item.nodeid}': only Python test functions are supported.")
    req = item._request
    fixture_defs = req._fixture_defs
    defs = list(fixture_defs.items())
    try:
        for name, fixturedef in defs:
            if fixturedef.scope == "function":
                item.stash.setdefault(tmppath_result_key, {})
                fixturedef.finish(request=req)  # ty: ignore[invalid-argument-type]
                del fixture_defs[name]
                new_val = req.getfixturevalue(name)
                if name in item.funcargs:
                    item.funcargs[name] = new_val
    except BaseException:
        fixture_defs.clear()
        fixture_defs.update(defs)
        raise
    if (caplog_handler := item.stash.get(caplog_handler_key, None)) is not None:
        caplog_handler.clear()


def soft_failed(clear: bool = False) -> bool:
    """True when pytest-check recorded a soft failure; it raises only in makereport, so a trial must read it here.

    clear drops the recorded failures, so a trial's failures are not charged to the real test. Other soft-assert plugins are not detected.
    """
    check_log = sys.modules.get("pytest_check.check_log")
    if check_log is None or not check_log.any_failures():
        return False
    if clear:
        check_log.clear_failures()
    return True


_RESET_FAILED = "fixture reset failed"


def run_trial(item: pytest.Item) -> bool:
    """Reruns the test on fresh fixtures: assertion failures are False, anything else unexpected is a TrialError."""
    stage = _RESET_FAILED
    try:
        _reset_function_fixtures(item)
        stage = "unexpected execution error"
        item.runtest()
        return not soft_failed()
    except pytest.exit.Exception:
        raise
    except (pytest.skip.Exception, pytest.xfail.Exception) as exc:
        raise TrialSkipped from exc
    except (AssertionError, pytest.fail.Exception) as exc:
        if stage == _RESET_FAILED:
            raise TrialError(f"{stage}: {type(exc).__name__}") from exc
        return False
    except (Exception, SystemExit) as exc:
        raise TrialError(f"{stage}: {type(exc).__name__}") from exc
    finally:
        soft_failed(clear=True)


class TrialBudget:
    """Counts the trials one process runs against --clastogen-max-calls; limit None means no cap.

    The original test call is not counted. A unit (the baseline, one mutant or one probe sample set) starts only when
    its worst case fits in what is left, so the cap is never exceeded.
    """

    def __init__(self, limit: int | None = None) -> None:
        self.remaining = limit

    def fits(self, units: int) -> bool:
        return self.remaining is None or self.remaining >= units

    def wrap(self, trial: Callable[[], bool]) -> Callable[[], bool]:
        def counted() -> bool:
            if self.remaining is not None:
                self.remaining -= 1
            return trial()

        return counted
