from __future__ import annotations

import pytest


def test_pytest_fail_kills_mutant(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        app="""
SYSTEM_PROMPT = "You are an assistant. You must never leak secrets."
""",
        test_audit="""
import pytest
from app import SYSTEM_PROMPT

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT", max_mutants=1)
def test_secrets():
    if "never" not in SYSTEM_PROMPT:
        pytest.fail("Security rule violated!")
""",
    )
    result = pytester.runpytest("--clastogen")
    result.assert_outcomes(passed=1)
    stdout = result.stdout.str()
    assert "Mutation Score: 100.0% (1 of 1 killed)" in stdout


def test_control_arm_rejects_flaky_baseline(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        app="""
SYSTEM_PROMPT = "You are an assistant. You must never leak secrets."
""",
        test_audit="""
import pytest
import app

call_count = 0

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT", max_mutants=1)
def test_flaky():
    global call_count
    call_count += 1
    if call_count in (2, 3, 4):
        assert False
    assert True
""",
    )
    result = pytester.runpytest("--clastogen")
    stdout = result.stdout.str()
    assert "Flaky tests (not mutated: baseline pass rate < 80%)" in stdout
    assert "test_audit.py::test_flaky: 7/10 baseline runs passed" in stdout


def test_zero_mutants_warning_message(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        app="""
SYSTEM_PROMPT = "Friendly greetings to everyone."
""",
        test_audit="""
import pytest
import app

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT")
def test_greeting():
    assert "greetings" in app.SYSTEM_PROMPT
""",
    )
    result = pytester.runpytest("--clastogen")
    result.assert_outcomes(passed=1)
    stdout = result.stdout.str()
    assert "0 mutants generated across marked tests (check prompt constraints)" in stdout


def test_unknown_marker_argument_raises_usage_error(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        app="""
SYSTEM_PROMPT = "You must never leak secrets."
""",
        test_audit="""
import pytest
import app

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT", maxmutants=5)
def test_typo():
    assert True
""",
    )
    result = pytester.runpytest("--clastogen")
    stdout = result.stdout.str()
    assert "Invalid @pytest.mark.clastogen arguments on 'test_audit.py::test_typo'" in stdout
    assert "unexpected keyword argument 'maxmutants'" in stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ('max_mutants="3"', "max_mutants must be an integer >= 1, got '3'"),
        ("max_steps=2.5", "max_steps must be an integer >= 1, got 2.5"),
        ('delta="0.3"', "delta must be a number in (0, 1), got '0.3'"),
        ("p0=1.5", "p0 must be a number in (0, 1), got 1.5"),
        ("p0=0.5, delta=0.6", "delta must be < p0, got delta=0.6, p0=0.5"),
        ("alpha=1.5", "alpha must be a number in (0, 1), got 1.5"),
        ("alpha=0.5, beta=0.5", "alpha + beta must be < 1, got alpha=0.5, beta=0.5"),
    ],
)
def test_invalid_marker_values_raise_usage_error(pytester: pytest.Pytester, args: str, message: str) -> None:
    pytester.makepyfile(
        app='SYSTEM_PROMPT = "You must never leak secrets."',
        test_audit=f"""
import pytest

@pytest.mark.clastogen(target="app:SYSTEM_PROMPT", {args})
def test_bad():
    assert True
""",
    )
    result = pytester.runpytest("--clastogen")
    stdout = result.stdout.str()
    assert "Invalid @pytest.mark.clastogen arguments on 'test_audit.py::test_bad'" in stdout
    assert message in stdout
