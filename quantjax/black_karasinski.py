"""Port of quantburn/src/black_karasinski.rs (originally
torchquantlib/models/interest_rate/black_karasinski.py).

Black-Karasinski short-rate model (lognormal Ornstein-Uhlenbeck). Includes
the initial-discount-curve interpolation, the trapezoidal zero-coupon bond
price over the expected short-rate path, and the terminal lognormal sampler
used by `simulate`. Module RNG seed: 2005. All math is scalar float64;
normals come from quantjax.rng.Xorshift64Star (bit-identical to the Rust
sampler).
"""

import bisect
import math
from typing import List, Optional, Sequence

from quantjax.rng import Xorshift64Star


def linear_interpolate(x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
    """Python `_linear_interpolate` (same as the Hull-White file's copy)."""
    if len(xp) == 1:
        return fp[0]
    clamped = min(max(x, xp[0]), xp[len(xp) - 1])
    upper = bisect.bisect_left(xp, clamped)
    upper = min(max(upper, 1), len(xp) - 1)
    lower = upper - 1
    weight = (clamped - xp[lower]) / max(xp[upper] - xp[lower], 1e-12)
    return fp[lower] + weight * (fp[upper] - fp[lower])


class BlackKarasinski:
    """Black-Karasinski model state: speed `a`, volatility `sigma`, initial
    short rate `r0`, and an optional fitted initial discount curve."""

    def __init__(self, a: float, sigma: float, r0: float):
        self.a = a
        self.sigma = sigma
        self.r0 = r0
        self.curve_maturities: Optional[List[float]] = None
        self.curve_discount_factors: Optional[List[float]] = None

    def fit_initial_discount_curve(self, maturities: List[float], discount_factors: List[float]) -> None:
        """Python `fit_initial_discount_curve`."""
        if len(maturities) != len(discount_factors):
            raise ValueError(
                "maturities and discount_factors must be one-dimensional tensors of the same shape."
            )
        if len(maturities) == 0:
            raise ValueError("maturities and discount_factors must be non-empty.")
        self.curve_maturities = list(maturities)
        self.curve_discount_factors = list(discount_factors)

    def initial_discount_factor(self, maturity: float) -> float:
        """Python `initial_discount_factor` (note the r0 clamp at 1e-8)."""
        if self.curve_maturities is not None and self.curve_discount_factors is not None:
            return linear_interpolate(maturity, self.curve_maturities, self.curve_discount_factors)
        return math.exp(-max(self.r0, 1e-8) * maturity)

    def instantaneous_forward_rate(self, maturity: float) -> float:
        """Python `instantaneous_forward_rate` (central difference, bump 1e-4)."""
        bump = 1e-4
        t_up = maturity + bump
        t_down = max(maturity - bump, 0.0)
        log_up = -math.log(max(self.initial_discount_factor(t_up), 1e-12))
        log_down = -math.log(max(self.initial_discount_factor(t_down), 1e-12))
        return (log_up - log_down) / max(t_up - t_down, 1e-12)

    def zero_coupon_bond_price(self, maturity: float, r_t: Optional[float] = None, integration_steps: int = 256) -> float:
        """Python `zero_coupon_bond_price`: exp(-trapz E[r_t] dt) with
        E[r_t] = exp(log(r0) e^{-a t} + variance(t)/2) over `integration_steps`
        points (default 256). When no explicit `r_t` is given and a curve is
        fitted, the interpolated discount factor is returned instead.
        """
        if maturity <= 0.0:
            return 1.0
        if r_t is None and self.curve_maturities is not None:
            return self.initial_discount_factor(maturity)
        a = max(self.a, 1e-12)
        sigma = max(self.sigma, 0.0)
        short_rate = max(self.r0 if r_t is None else r_t, 1e-8)

        n = max(int(integration_steps), 2)
        dt = maturity / (n - 1)
        # torch.linspace over [0, maturity], trapezoid rule.
        integral = 0.0
        for i in range(n):
            t = dt * i
            mean_log_r = math.log(short_rate) * math.exp(-a * t)
            variance_log_r = sigma * sigma * (1.0 - math.exp(-2.0 * a * t)) / max(2.0 * a, 1e-12)
            expected_short_rate = math.exp(mean_log_r + 0.5 * variance_log_r)
            w = 0.5 if (i == 0 or i == n - 1) else 1.0
            integral += w * expected_short_rate * dt
        return math.exp(-integral)

    def zero_coupon_yield(self, maturity: float, r_t: Optional[float] = None, integration_steps: int = 256) -> float:
        """Python `zero_coupon_yield`."""
        price = self.zero_coupon_bond_price(maturity, r_t, integration_steps)
        return -math.log(max(price, 1e-12)) / max(maturity, 1e-12)

    def simulate(self, t: float, n: int, seed: int) -> List[float]:
        """Python `simulate`: N draws of exp(mean + sqrt(var) z) where the
        log-rate is Gaussian. Normals from `quantjax.rng.Xorshift64Star` with
        `seed` (2005 in tests).
        """
        a = max(self.a, 0.0)
        sigma = max(self.sigma, 0.0)
        r0 = max(self.r0, 1e-8)
        log_r0 = math.log(r0)
        mean = log_r0 * math.exp(-a * t)
        if sigma == 0.0:
            return [math.exp(mean)] * n
        variance = sigma * sigma * (1.0 - math.exp(-2.0 * a * t)) / max(2.0 * a, 1e-12)
        std = math.sqrt(max(variance, 0.0))
        rng = Xorshift64Star(seed)
        return [math.exp(mean + std * rng.next_std_normal()) for _ in range(n)]


# ---------------------------------------------------------------------------
# Embedded parity tests (expected literals from quantburn/src/black_karasinski.rs)
# ---------------------------------------------------------------------------


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def test_bond_price_matches_reference():
    bk = BlackKarasinski(0.1, 0.01, 0.03)
    p1 = bk.zero_coupon_bond_price(1.0, None, 256)
    p5 = bk.zero_coupon_bond_price(5.0, None, 256)
    p10 = bk.zero_coupon_bond_price(10.0, None, 256)
    y1 = bk.zero_coupon_yield(1.0, None, 256)
    assert _rel(p1, 0.9649184371120622) < 1e-11, f"p1={p1}"
    assert _rel(p5, 0.7105610577190904) < 1e-11, f"p5={p5}"
    assert _rel(p10, 0.27069253076250477) < 1e-11, f"p10={p10}"
    assert _rel(y1, 0.03571170234265963) < 1e-10, f"y1={y1}"
    assert bk.zero_coupon_bond_price(0.0, None, 256) == 1.0


def test_bond_price_uses_fitted_curve():
    bk = BlackKarasinski(0.1, 0.01, 0.03)
    bk.fit_initial_discount_curve([1.0, 5.0], [0.97, 0.86])
    assert _rel(bk.zero_coupon_bond_price(3.0, None, 256), 0.915) < 1e-12


def test_simulate_matches_reference():
    bk = BlackKarasinski(0.1, 0.01, 0.03)
    r = bk.simulate(2.0, 8, 2005)
    expected = [
        0.05636124422394425,
        0.05628683234533506,
        0.05687669858060467,
        0.056972978601515185,
        0.05591941959828365,
        0.05674199927477411,
        0.05616423664753451,
        0.05642739581981856,
    ]
    for got, want in zip(r, expected):
        assert _rel(got, want) < 1e-12, f"got={got} want={want}"


def test_simulate_sigma_zero_is_degenerate():
    bk = BlackKarasinski(0.1, 0.0, 0.03)
    r = bk.simulate(2.0, 4, 2005)
    expected = 0.05664640303882101
    for v in r:
        assert abs(v - expected) < 1e-15
