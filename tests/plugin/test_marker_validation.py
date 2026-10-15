from __future__ import annotations

import pytest


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
        ("oracle_samples=20, alpha=0.5", "alpha must be < 0.5 for a probe, got 0.5"),
        *[(f"oracle_samples={s}", "oracle_samples must be an integer >= 5") for s in (0, 4, True, 5.0)],
        ("maxmutants=5", "unexpected keyword argument 'maxmutants'"),
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
