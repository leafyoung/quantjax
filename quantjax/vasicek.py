"""Port of quantburn/src/vasicek.rs (originally
torchquantlib/models/interest_rate/vasicek.py).

Closed-form zero-coupon bond price/yield plus the terminal short-rate
distribution sampler used by `simulate` (exact Gaussian transition over the
full horizon). Module RNG seed: 2002. All math is scalar float64 (plain
Python floats); normals come from quantjax.rng.Xorshift64Star (bit-identical
to the Rust sampler).
"""

import math

from quantjax.rng import Xorshift64Star


def zero_coupon_bond_price(kappa: float, theta: float, sigma: float, r0: float, maturity: float) -> float:
    """Python `Vasicek.zero_coupon_bond_price` with the same internal clamps."""
    if maturity <= 0.0:
        return 1.0
    kappa = max(kappa, 1e-12)
    sigma = max(sigma, 0.0)
    b_term = (1.0 - math.exp(-kappa * maturity)) / kappa
    a_term = math.exp(
        (theta - sigma * sigma / (2.0 * kappa * kappa)) * (b_term - maturity)
        - sigma * sigma * b_term * b_term / (4.0 * kappa)
    )
    return a_term * math.exp(-b_term * r0)


def zero_coupon_yield(kappa: float, theta: float, sigma: float, r0: float, maturity: float) -> float:
    """Python `Vasicek.zero_coupon_yield`."""
    price = zero_coupon_bond_price(kappa, theta, sigma, r0, maturity)
    return -math.log(max(price, 1e-12)) / max(maturity, 1e-12)


def simulate(kappa: float, theta: float, sigma: float, r0: float, t: float, n: int, seed: int) -> list:
    """Python `Vasicek.simulate`: N draws from the exact terminal short-rate
    distribution r(T) ~ N(mean, variance), with the Python's special cases
    (kappa*T >= 1 collapses the mean to theta; sigma == 0 is deterministic).
    Normals come from `quantjax.rng.Xorshift64Star` with `seed` (2002 in tests).
    """
    kappa = max(kappa, 0.0)
    sigma = max(sigma, 0.0)
    mean_reversion = math.exp(-kappa * t)
    mean = theta + (r0 - theta) * mean_reversion
    if kappa * t >= 1.0:
        mean = theta
    if sigma == 0.0:
        return [mean] * n
    variance = sigma * sigma * (1.0 - math.exp(-2.0 * kappa * t)) / max(2.0 * kappa, 1e-12)
    std = math.sqrt(max(variance, 0.0))
    rng = Xorshift64Star(seed)
    return [mean + std * rng.next_std_normal() for _ in range(n)]


# ---------------------------------------------------------------------------
# Embedded parity tests (expected literals from quantburn/src/vasicek.rs)
# ---------------------------------------------------------------------------

KAPPA = 0.15
THETA = 0.05
SIGMA = 0.02
R0 = 0.03


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def test_bond_price_closed_form():
    p1 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 1.0)
    p5 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 5.0)
    p10 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 10.0)
    y1 = zero_coupon_yield(KAPPA, THETA, SIGMA, R0, 1.0)
    assert _rel(p1, 0.9691188082746262) < 1e-12, f"p1={p1}"
    assert _rel(p5, 0.8396917673474292) < 1e-12, f"p5={p5}"
    assert _rel(p10, 0.6897352359349652) < 1e-12, f"p10={p10}"
    assert _rel(y1, 0.03136806544871844) < 1e-11, f"y1={y1}"
    assert zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 0.0) == 1.0


def test_simulate_matches_reference():
    r = simulate(KAPPA, THETA, SIGMA, R0, 2.0, 8, 2002)
    expected = [
        0.02613496077664776,
        -0.012153713624497658,
        0.03323136734270169,
        0.02382083956028588,
        0.02579654700137396,
        0.041537151891176356,
        0.059147298795526104,
        0.025514955997793482,
    ]
    for got, want in zip(r, expected):
        assert _rel(got, want) < 1e-12, f"got={got} want={want}"


def test_simulate_sigma_zero_is_degenerate():
    r = simulate(KAPPA, THETA, 0.0, R0, 0.5, 4, 2002)
    # kappa*T = 0.075 < 1, mean = theta + (r0-theta)*exp(-kappa T).
    mean = THETA + (R0 - THETA) * math.exp(-KAPPA * 0.5)
    for v in r:
        assert abs(v - mean) < 1e-15


def test_simulate_long_horizon_mean_collapses_to_theta():
    r = simulate(KAPPA, THETA, 0.0, R0, 10.0, 4, 2002)
    # kappa*T = 1.5 >= 1 -> mean = theta.
    for v in r:
        assert abs(v - THETA) < 1e-15
