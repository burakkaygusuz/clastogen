import pytest

from clastogen.stats import compute_wilson_interval
from clastogen.stats.assertions import _binom_upper_tail, _fisher_upper_tail

# Reference values computed with scipy.stats (binomtest / fisher_exact).


@pytest.mark.parametrize(
    ("k", "n", "conf", "low", "high"),
    [
        (0, 10, 0.95, 0.0, 0.27753279986288926),
        (7, 10, 0.95, 0.39677814746114526, 0.892208732593699),
        (10, 10, 0.9, 0.787058029916593, 1.0),
        (45, 50, 0.99, 0.7402692104043159, 0.9660091162215817),
    ],
)
def test_wilson_matches_reference(k: int, n: int, conf: float, low: float, high: float) -> None:
    assert compute_wilson_interval(k, n, conf) == pytest.approx((low, high), abs=1e-12)


@pytest.mark.parametrize(("b", "n", "p"), [(8, 10, 0.0546875), (5, 9, 0.5), (12, 12, 0.000244140625)])
def test_binom_upper_tail_matches_reference(b: int, n: int, p: float) -> None:
    assert _binom_upper_tail(b, n) == pytest.approx(p, abs=1e-12)


@pytest.mark.parametrize(
    ("a", "c", "n", "p"),
    [(18, 10, 20, 0.006907073925983827), (9, 9, 15, 0.6448091743601884), (30, 22, 30, 0.0022875311953459835)],
)
def test_fisher_upper_tail_matches_reference(a: int, c: int, n: int, p: float) -> None:
    assert _fisher_upper_tail(a, c, n) == pytest.approx(p, abs=1e-12)
