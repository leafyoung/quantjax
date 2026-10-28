"""Port of quantburn/src/cir.rs (originally
torchquantlib/models/interest_rate/cir.py).

Closed-form zero-coupon bond price/yield plus the Gaussian
stationary-approximation terminal sampler used by `simulate` (the Python
matches first two moments of the exact CIR transition and applies a ReLU
floor). Module RNG seed: 2003. All math is scalar float64; normals come from
quantjax.rng.Xorshift64Star (bit-identical to the Rust sampler).
"""

import math

from quantjax.rng import Xorshift64Star


def zero_coupon_bond_price(kappa: float, theta: float, sigma: float, r0: float, maturity: float) -> float:
    """Python `CIR.zero_coupon_bond_price` with the same internal clamps."""
    if maturity <= 0.0:
        return 1.0
    kappa = max(kappa, 1e-12)
    theta = max(theta, 0.0)
    sigma = max(sigma, 1e-12)
    short_rate = max(r0, 0.0)

    gamma = math.sqrt(kappa * kappa + 2.0 * sigma * sigma)
    exp_gamma_t = math.exp(gamma * maturity)
    denominator = (gamma + kappa) * (exp_gamma_t - 1.0) + 2.0 * gamma
    denom = max(denominator, 1e-12)
    b_term = 2.0 * (exp_gamma_t - 1.0) / denom
    a_term = max(
        2.0 * gamma * math.exp((kappa + gamma) * maturity / 2.0) / denom,
        1e-12,
    ) ** (2.0 * kappa * theta / (sigma * sigma))
    return a_term * math.exp(-b_term * short_rate)


def zero_coupon_yield(kappa: float, theta: float, sigma: float, r0: float, maturity: float) -> float:
    """Python `CIR.zero_coupon_yield`."""
    price = zero_coupon_bond_price(kappa, theta, sigma, r0, maturity)
    return -math.log(max(price, 1e-12)) / max(maturity, 1e-12)


def simulate(kappa: float, theta: float, sigma: float, r0: float, t: float, n: int, seed: int) -> list:
    """Python `CIR.simulate` (including the Feller condition adjustment the
    constructor applies when `2*kappa*theta <= sigma^2`: theta is bumped to
    `sigma^2/(2*kappa) + 1e-6`). Normals from `quantjax.rng.Xorshift64Star`
    with `seed` (2003 in tests).
    """
    # Constructor-time Feller fix (matches __init__ in the Python).
    theta = max(theta, 0.0)
    kappa_c = max(kappa, 1e-12)
    feller_gap = 2.0 * kappa_c * theta - sigma * sigma
    if feller_gap <= 0.0:
        theta = sigma * sigma / max(kappa_c, 1e-12) / 2.0 + 1e-6

    kappa = max(kappa, 0.0)
    sigma = max(sigma, 0.0)
    r0 = max(r0, 0.0)
    mean_reversion = math.exp(-kappa * t)
    mean = theta + (r0 - theta) * mean_reversion
    if kappa * t >= 1.0:
        mean = theta
    if sigma == 0.0:
        return [max(mean, 0.0)] * n
    stationary_variance = sigma * sigma * theta / max(2.0 * kappa, 1e-12)
    variance = stationary_variance * (1.0 - math.exp(-2.0 * kappa * t))
    std = math.sqrt(max(variance, 0.0))
    rng = Xorshift64Star(seed)
    return [max(mean + std * rng.next_std_normal(), 0.0) for _ in range(n)]


# ---------------------------------------------------------------------------
# Embedded parity tests (expected literals from quantburn/src/cir.rs)
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
    y5 = zero_coupon_yield(KAPPA, THETA, SIGMA, R0, 5.0)
    assert _rel(p1, 0.9690627652294498) < 1e-11, f"p1={p1}"
    assert _rel(p5, 0.8357016192777666) < 1e-11, f"p5={p5}"
    assert _rel(p10, 0.6733398275379303) < 1e-11, f"p10={p10}"
    assert _rel(y5, 0.035896728869234854) < 1e-10, f"y5={y5}"
    assert zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 0.0) == 1.0


def test_simulate_matches_reference():
    r = simulate(KAPPA, THETA, SIGMA, R0, 2.0, 8, 2003)
    expected = [
        0.027491557755358782,
        0.03929694074081416,
        0.038447389418432246,
        0.04658625647593316,
        0.036818660016410695,
        0.042299964951907305,
        0.03919273215255722,
        0.021893950585852563,
    ]
    for got, want in zip(r, expected):
        assert _rel(got, want) < 1e-12, f"got={got} want={want}"


def test_feller_violation_bumps_theta():
    # 2*kappa*theta = 0.02 < sigma^2 = 0.09 -> theta bumped to
    # sigma^2/(2*kappa) + 1e-6, which inflates the stationary variance.
    # Reference (t=1.0) from scripts/gen_cir.py, seed 2003.
    r = simulate(0.1, 0.1, 0.3, 0.03, 1.0, 4, 2003)
    expected = [0.0, 0.21366040800835318, 0.18398263109246554, 0.4683015115614581]
    for got, want in zip(r, expected):
        assert abs(got - want) < 1e-15, f"got={got} want={want}"


def test_simulate_relu_floor():
    # Large sigma with short horizon: draws below zero must be floored.
    r = simulate(0.1, 0.001, 0.5, 0.001, 0.1, 64, 2003)
    assert all(v >= 0.0 for v in r)
