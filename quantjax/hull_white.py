"""Port of quantburn/src/hull_white.rs (originally
torchquantlib/models/interest_rate/hull_white.py, curve-consistent
Brigo-Mercurio (3.39) update of 2026-10-27).

Includes the initial-discount-curve interpolation machinery, the numerical
instantaneous forward rate and theta(t), the closed-form zero-coupon bond
price, and the terminal Gaussian sampler used by `simulate`. Module RNG
seed: 2004. All math is scalar float64; normals come from
quantjax.rng.Xorshift64Star (bit-identical to the Rust sampler).
"""

import bisect
import math
from typing import List, Optional, Sequence

from quantjax.rng import Xorshift64Star


def linear_interpolate(x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
    """Python `_linear_interpolate` (clamp to endpoints, searchsorted mid-rule)."""
    if len(xp) == 1:
        return fp[0]
    clamped = min(max(x, xp[0]), xp[len(xp) - 1])
    # torch.searchsorted(xp, clamped, right=False): first index with
    # xp[idx] >= clamped, clamped into [1, len-1].
    upper = bisect.bisect_left(xp, clamped)
    upper = min(max(upper, 1), len(xp) - 1)
    lower = upper - 1
    weight = (clamped - xp[lower]) / max(xp[upper] - xp[lower], 1e-12)
    return fp[lower] + weight * (fp[upper] - fp[lower])


class HullWhiteModel:
    """Hull-White model state: mean-reversion speed `a`, volatility `sigma`,
    initial short rate `r0`, and an optional fitted initial discount curve."""

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
        """Python `initial_discount_factor`."""
        if self.curve_maturities is not None and self.curve_discount_factors is not None:
            return linear_interpolate(maturity, self.curve_maturities, self.curve_discount_factors)
        return math.exp(-self.r0 * maturity)

    def instantaneous_forward_rate(self, maturity: float) -> float:
        """Python `instantaneous_forward_rate` (central difference, bump 1e-4)."""
        bump = 1e-4
        t_up = maturity + bump
        t_down = max(maturity - bump, 0.0)
        log_up = -math.log(max(self.initial_discount_factor(t_up), 1e-12))
        log_down = -math.log(max(self.initial_discount_factor(t_down), 1e-12))
        return (log_up - log_down) / max(t_up - t_down, 1e-12)

    def theta(self, maturity: float) -> float:
        """Python `theta(t)`."""
        a = max(self.a, 1e-12)
        sigma = max(self.sigma, 0.0)
        bump = 1e-4
        fwd_up = self.instantaneous_forward_rate(maturity + bump)
        fwd_down = self.instantaneous_forward_rate(max(maturity - bump, 0.0))
        dfdt = (fwd_up - fwd_down) / max(2.0 * bump, 1e-12)
        forward = self.instantaneous_forward_rate(maturity)
        return dfdt + a * forward + (sigma * sigma / (2.0 * a)) * (1.0 - math.exp(-2.0 * a * maturity))

    def zero_coupon_bond_price(self, maturity: float, r_t: Optional[float] = None) -> float:
        """Python `zero_coupon_bond_price` — curve-consistent Hull-White ZCB
        (Brigo-Mercurio (3.39)) at t=0: P = P^M(0,T)·exp(B·(f^M(0,0) − r_t))
        with B = (1−e^{−aT})/a and f^M(0,0) the one-sided finite-difference
        initial forward (bump 1e-4). Replaces the previous zero-drift closed
        form, which did not reproduce the initial curve. When no explicit
        `r_t` is given and a curve has been fitted, the interpolated discount
        factor is returned.
        """
        if maturity <= 0.0:
            return 1.0
        if r_t is None and self.curve_maturities is not None:
            return self.initial_discount_factor(maturity)
        a = max(self.a, 1e-12)
        short_rate = self.r0 if r_t is None else r_t
        b_term = (1.0 - math.exp(-a * maturity)) / a
        market_price = self.initial_discount_factor(maturity)
        bump = 1e-4
        f0 = (
            -math.log(max(self.initial_discount_factor(bump), 1e-12))
            + math.log(max(self.initial_discount_factor(0.0), 1e-12))
        ) / bump
        a_term = market_price * math.exp(b_term * f0)
        return a_term * math.exp(-b_term * short_rate)

    def zero_coupon_yield(self, maturity: float, r_t: Optional[float] = None) -> float:
        """Python `zero_coupon_yield`."""
        price = self.zero_coupon_bond_price(maturity, r_t)
        return -math.log(max(price, 1e-12)) / max(maturity, 1e-12)

    def simulate(self, t: float, n: int, seed: int) -> List[float]:
        """Python `simulate`: N draws from the exact terminal Gaussian of the
        Ornstein-Uhlenbeck short rate. Normals from
        `quantjax.rng.Xorshift64Star` with `seed` (2004 in tests).
        """
        a = max(self.a, 0.0)
        sigma = max(self.sigma, 0.0)
        mean = self.r0 * math.exp(-a * t)
        if sigma == 0.0:
            return [mean] * n
        variance = sigma * sigma * (1.0 - math.exp(-2.0 * a * t)) / max(2.0 * a, 1e-12)
        std = math.sqrt(max(variance, 0.0))
        rng = Xorshift64Star(seed)
        return [mean + std * rng.next_std_normal() for _ in range(n)]


# ---------------------------------------------------------------------------
# Embedded parity tests (expected literals from quantburn/src/hull_white.rs)
# ---------------------------------------------------------------------------


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def test_linear_interpolate_matches_reference():
    xp = [0.5, 1.0, 2.0, 5.0]
    fp = [0.9753, 0.9512, 0.9048, 0.7788]
    assert _rel(linear_interpolate(1.25, xp, fp), 0.9396) < 1e-12
    assert _rel(linear_interpolate(0.2, xp, fp), 0.9753) < 1e-12  # clamped low
    assert _rel(linear_interpolate(7.0, xp, fp), 0.7788) < 1e-12  # clamped high
    assert _rel(linear_interpolate(3.7, xp, fp), 0.8334) < 1e-12


def test_bond_price_closed_form():
    hw = HullWhiteModel(0.1, 0.01, 0.03)
    p1 = hw.zero_coupon_bond_price(1.0)
    p5 = hw.zero_coupon_bond_price(5.0)
    p10 = hw.zero_coupon_bond_price(10.0)
    assert _rel(p1, 0.9704455335486447) < 1e-12, f"p1={p1}"
    assert _rel(p5, 0.8607079764255583) < 1e-12, f"p5={p5}"
    assert _rel(p10, 0.7408182206824101) < 1e-12, f"p10={p10}"
    # sanity: curve-consistent means flat curve -> exp(-r0*T)
    assert _rel(p1, math.exp(-0.03)) < 1e-11, f"p1 vs exp(-r0T)={p1}"


def test_bond_price_uses_fitted_curve():
    hw = HullWhiteModel(0.1, 0.01, 0.03)
    hw.fit_initial_discount_curve([1.0, 5.0], [0.97, 0.86])
    assert _rel(hw.zero_coupon_bond_price(2.5), 0.92875) < 1e-12


def test_forward_and_theta_flat_curve():
    hw = HullWhiteModel(0.1, 0.01, 0.03)
    f = hw.instantaneous_forward_rate(2.0)
    th = hw.theta(2.0)
    assert _rel(f, 0.03) < 1e-9, f"f={f}"
    # Reference from scripts/gen_hull_white.py (same finite-difference
    # recipe); tiny deviation from the analytic value comes from the
    # bump arithmetic.
    assert _rel(th, 0.0031648418990315058) < 1e-12, f"th={th}"


def test_simulate_matches_reference():
    hw = HullWhiteModel(0.1, 0.01, 0.03)
    r = hw.simulate(2.0, 8, 2004)
    expected = [
        0.01823480828541723,
        0.04722897441969502,
        0.030845099769675633,
        0.040971256455761565,
        0.06182941469944912,
        -0.0029258789730068797,
        0.019132912054679173,
        0.03848256979958703,
    ]
    for got, want in zip(r, expected):
        assert _rel(got, want) < 1e-12, f"got={got} want={want}"
