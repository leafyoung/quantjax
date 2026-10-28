"""Port of quantburn/src/structural_model.rs (originally
credit_risk/structural_model.py, Merton structural model).

Equity is priced as a Black-Scholes call option on the firm's assets:
``E = V * N(d1) - D * exp(-r*T) * N(d2)`` with the usual d1/d2.
"""

import math

_SQRT_2 = math.sqrt(2.0)
_SQRT_PI = 1.7724538509055159
_A = 0.5  # incomplete-gamma shape parameter
_GLN = 0.5723649429247001  # ln Gamma(0.5) = 0.5 * ln(pi)


def _erfc_pos(z: float) -> float:
    """erfc for z >= 0, accurate to ~1e-15."""
    if z < 1.0:
        # erf Taylor series (alternating, fast for |z| < 1).
        x2 = z * z
        term = z
        s = z
        for n in range(1, 80):
            term *= -x2 / n
            add = term / (2 * n + 1)
            s += add
            if abs(add) < 1e-18 * abs(s):
                break
        return 1.0 - s * 2.0 / _SQRT_PI
    xx = z * z
    if xx < _A + 1.0:
        # Regularized lower incomplete gamma P(0.5, z^2) by series.
        ap = _A
        s = 1.0 / _A
        de = s
        for _ in range(1000):
            ap += 1.0
            de *= xx / ap
            s += de
            if abs(de) < abs(s) * 1e-16:
                break
        return 1.0 - s * math.exp(-xx + _A * math.log(xx) - _GLN)
    # Regularized upper incomplete gamma Q(0.5, z^2) by the modified Lentz
    # continued fraction (Numerical Recipes gcf).
    fmin = 1e-300
    b = xx + 1.0 - _A
    c = 1.0 / fmin
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - _A)
        b += 2.0
        d = an * d + b
        if abs(d) < fmin:
            d = fmin
        c = b + an / c
        if abs(c) < fmin:
            c = fmin
        d = 1.0 / d
        dele = d * c
        h *= dele
        if abs(dele - 1.0) < 1e-16:
            break
    return math.exp(-xx + _A * math.log(xx) - _GLN) * h


def erfc(x: float) -> float:
    """Complementary error function."""
    q = _erfc_pos(abs(x))
    if x >= 0.0:
        return q
    return 2.0 - q


def norm_cdf(x: float) -> float:
    """Standard normal CDF, computed from a self-contained double-precision
    ``erfc`` (series for small args, incomplete-gamma series/continued
    fraction for the tails; verified to ~1e-15 against Python math.erfc)."""
    return 0.5 * erfc(-x / _SQRT_2)


def merton_model(
    asset_value: float, debt: float, volatility: float, rate: float, expiry: float
) -> float:
    """Merton structural model: equity value of the firm."""
    sqrt_t = math.sqrt(expiry)
    d1 = (
        math.log(asset_value / debt) + (rate + 0.5 * volatility * volatility) * expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return asset_value * norm_cdf(d1) - debt * math.exp(-rate * expiry) * norm_cdf(d2)


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn/src/structural_model.rs)
# ---------------------------------------------------------------------------


def test_matches_python_reference():
    # Reference: torch (float64) ``merton_model`` from the Python source,
    # e.g. merton_model(tensor(100.0), tensor(80.0), tensor(0.25),
    # tensor(0.05), tensor(1.0)).item() == 25.412511998314308.
    cases = [
        (100.0, 80.0, 0.25, 0.05, 1.0, 25.412511998314308),
        (100.0, 120.0, 0.3, 0.03, 2.0, 12.113095785829557),
        (50.0, 40.0, 0.2, 0.0, 0.5, 10.154557237945262),
    ]
    for v, d, vol, r, t, expected in cases:
        got = merton_model(v, d, vol, r, t)
        assert abs(got - expected) <= 1e-8 * abs(expected), (
            f"got {got}, expected {expected}"
        )


def test_erfc_accuracy():
    # erfc accuracy sanity vs math.erfc reference values.
    cases = [
        (0.0, 1.0),
        (0.5, 0.4795001221869535),
        (1.0, 0.15729920705028513),
        (1.5, 0.033894853524689274),
        (2.0, 0.004677734981047266),
        (3.0, 2.209049699858544e-5),
        (-1.2, 1.9103139782296354),
    ]
    for x, expected in cases:
        assert abs(erfc(x) - expected) <= 1e-12 * max(abs(expected), 1.0), f"erfc({x})"
