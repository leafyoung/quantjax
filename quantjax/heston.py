"""Heston stochastic volatility model — port of quantburn/src/heston.rs
(originally torchquantlib/models/stochastic_volatility/heston.py).

- `simulate_terminal_paths`: full-truncation Euler, terminal prices only.
  Scalar f64 reference in plain Python floats; `simulate_terminal_paths_tensor`
  runs the same scheme with jax.numpy ops (float64, x64 enabled in the
  package __init__). Normals are drawn host-side from
  `quantjax.rng.Xorshift64Star` in both cases (module seed 2007), per step as
  a block of `base_count` spot normals followed by `base_count` aux normals.
- `characteristic_function_price`: Lewis-style trapezoid integration of the
  log-price characteristic function over [1e-5, integration_limit],
  n = max(256, integration_points).
- `monte_carlo_option_price`: antithetic MC with martingale normalization.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp

from quantjax.rng import Xorshift64Star

EPSILON = 1e-8


@dataclass
class HestonParams:
    kappa: float = 1.0
    theta: float = 0.04
    sigma_v: float = 0.5
    rho: float = -0.5
    v0: float = 0.04
    mu: float = 0.0


class C64:
    """Minimal f64 complex number (mirrors the Rust C64 helper exactly)."""

    __slots__ = ("re", "im")

    def __init__(self, re: float, im: float):
        self.re = re
        self.im = im

    def sub(self, o):
        return C64(self.re - o.re, self.im - o.im)

    def add(self, o):
        return C64(self.re + o.re, self.im + o.im)

    def mul(self, o):
        return C64(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    def div(self, o):
        d = o.re * o.re + o.im * o.im
        return C64(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )

    def scale(self, s):
        return C64(self.re * s, self.im * s)

    def sqrt(self):
        r = math.hypot(self.re, self.im)
        m = math.sqrt(r)
        th = math.atan2(self.im, self.re) / 2.0
        return C64(m * math.cos(th), m * math.sin(th))

    def ln(self):
        return C64(math.log(math.hypot(self.re, self.im)), math.atan2(self.im, self.re))

    def exp(self):
        e = math.exp(self.re)
        return C64(e * math.cos(self.im), e * math.sin(self.im))

    def powi2(self):
        return self.mul(self)


def _antithetic_pad(z_raw, n):
    """Mirror of `tensor.repeat(2)[:n]`: [z, -z] truncated to n."""
    z = z_raw + [-v for v in z_raw]
    return z[:n]


def simulate_terminal_paths(params, s0, t, n, steps, drift=None, antithetic=False,
                            normalize_martingale=False, seed=2007):
    """Scalar f64 reference (plain Python floats + Xorshift64Star)."""
    spot = max(s0, EPSILON)
    if t <= 0.0:
        return [spot] * n
    steps = max(steps, 1)
    dt = t / steps
    sqrt_dt = math.sqrt(dt)

    model_drift = params.mu if drift is None else drift
    base_count = (n + 1) // 2 if antithetic else n

    rho_residual = math.sqrt(min(max(1.0 - params.rho * params.rho, EPSILON), 1.7976931348623157e308))

    rng = Xorshift64Star(seed)

    # Antithetic padding mirrors `tensor.repeat(2)[:num_paths]`; since the
    # initial arrays are constant, a full-length constant vector is identical.
    prices = [spot] * n
    variance = [params.v0] * n

    for _ in range(steps):
        z_spot_raw = [rng.next_std_normal() for _ in range(base_count)]
        z_aux_raw = [rng.next_std_normal() for _ in range(base_count)]
        z_spot = _antithetic_pad(z_spot_raw, n) if antithetic else z_spot_raw
        z_aux = _antithetic_pad(z_aux_raw, n) if antithetic else z_aux_raw

        for i in range(n):
            z_var = params.rho * z_spot[i] + rho_residual * z_aux[i]
            variance_pos = max(variance[i], 0.0)
            new_variance = max(
                variance[i]
                + params.kappa * (params.theta - variance_pos) * dt
                + params.sigma_v * math.sqrt(max(variance_pos, EPSILON)) * sqrt_dt * z_var,
                0.0,
            )
            prices[i] *= math.exp(
                (model_drift - 0.5 * variance_pos) * dt
                + math.sqrt(max(variance_pos, EPSILON)) * sqrt_dt * z_spot[i]
            )
            variance[i] = new_variance

    if normalize_martingale:
        target_forward = spot * math.exp(model_drift * t)
        mean = sum(prices) / len(prices)
        scale = target_forward / max(mean, EPSILON)
        prices = [p * scale for p in prices]
    return prices


def simulate_terminal_paths_tensor(params, s0, t, n, steps, drift=None, antithetic=False,
                                   normalize_martingale=False, seed=2007):
    """The same scheme as `simulate_terminal_paths`, but the per-step arithmetic
    (max(variance, 0), exp(log-ret) updates, martingale normalization) runs as
    jax.numpy float64 array ops. Normals are still drawn host-side from
    Xorshift64Star (identical draw order) and uploaded via jnp.array."""
    spot = max(s0, EPSILON)
    if t <= 0.0:
        return jnp.full((n,), spot, dtype=jnp.float64)
    steps = max(steps, 1)
    dt = t / steps
    sqrt_dt = math.sqrt(dt)

    model_drift = params.mu if drift is None else drift
    base_count = (n + 1) // 2 if antithetic else n

    rho_residual = math.sqrt(min(max(1.0 - params.rho * params.rho, EPSILON), 1.7976931348623157e308))

    rng = Xorshift64Star(seed)

    prices = jnp.full((n,), spot, dtype=jnp.float64)
    variance = jnp.full((n,), params.v0, dtype=jnp.float64)

    kappa = params.kappa
    theta = params.theta
    sigma_v = params.sigma_v
    rho = params.rho

    for _ in range(steps):
        z_spot_raw = [rng.next_std_normal() for _ in range(base_count)]
        z_aux_raw = [rng.next_std_normal() for _ in range(base_count)]
        z_spot = jnp.asarray(
            _antithetic_pad(z_spot_raw, n) if antithetic else z_spot_raw, dtype=jnp.float64
        )
        z_aux = jnp.asarray(
            _antithetic_pad(z_aux_raw, n) if antithetic else z_aux_raw, dtype=jnp.float64
        )

        z_var = rho * z_spot + rho_residual * z_aux
        variance_pos = jnp.maximum(variance, 0.0)
        new_variance = jnp.maximum(
            variance
            + kappa * (theta - variance_pos) * dt
            + sigma_v * jnp.sqrt(jnp.maximum(variance_pos, EPSILON)) * sqrt_dt * z_var,
            0.0,
        )
        log_ret = (
            (model_drift - 0.5 * variance_pos) * dt
            + jnp.sqrt(jnp.maximum(variance_pos, EPSILON)) * sqrt_dt * z_spot
        )
        prices = prices * jnp.exp(log_ret)
        variance = new_variance

    if normalize_martingale:
        target_forward = spot * math.exp(model_drift * t)
        mean = jnp.mean(prices)
        scale = target_forward / jnp.maximum(mean, EPSILON)
        prices = prices * scale
    return prices


def simulate(params, s0, t, n, steps, seed):
    """Plain Euler, no antithetic / normalization."""
    return simulate_terminal_paths(params, s0, t, n, steps, None, False, False, seed)


def log_price_characteristic_function(params, u, spot, expiry, rate, dividend):
    kappa, theta, sigma_v, rho, v0 = params.kappa, params.theta, params.sigma_v, params.rho, params.v0

    a = kappa * theta
    i = C64(0.0, 1.0)
    iu = i.mul(u)
    d = (
        iu.scale(rho * sigma_v).sub(C64(kappa, 0.0)).powi2()
        .add(C64(sigma_v * sigma_v, 0.0).mul(u.powi2().add(iu)))
        .sqrt()
    )
    g = C64(kappa, 0.0).sub(iu.scale(rho * sigma_v)).sub(d).div(
        C64(kappa, 0.0).sub(iu.scale(rho * sigma_v)).add(d)
    )
    exp_neg_d_t = d.scale(-expiry).exp()

    c_term = C64((rate - dividend) * expiry, 0.0).mul(iu).add(
        C64(a / (sigma_v * sigma_v), 0.0).mul(
            C64(kappa, 0.0).sub(iu.scale(rho * sigma_v)).sub(d).scale(expiry).sub(
                C64(1.0, 0.0).sub(g.mul(exp_neg_d_t)).div(C64(1.0, 0.0).sub(g)).ln().scale(2.0)
            )
        )
    )
    d_term = C64(kappa, 0.0).sub(iu.scale(rho * sigma_v)).sub(d).scale(1.0 / (sigma_v * sigma_v)).mul(
        C64(1.0, 0.0).sub(exp_neg_d_t).div(C64(1.0, 0.0).sub(g.mul(exp_neg_d_t)))
    )
    return c_term.add(d_term.scale(v0)).add(iu.scale(math.log(spot))).exp()


def characteristic_function_price(params, s0, k, t, r, option_type, q, integration_limit=100.0,
                                  integration_points=2000):
    option_style = option_type.lower()
    if option_style not in ("call", "put"):
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    spot = max(s0, EPSILON)
    strike = max(k, EPSILON)

    if t <= 0.0:
        return max(spot - strike, 0.0) if option_style == "call" else max(strike - spot, 0.0)

    n = max(integration_points, 256)
    u_lo = 1e-5
    du = (integration_limit - u_lo) / (n - 1)
    log_strike = math.log(strike)
    i = C64(0.0, 1.0)
    phi_minus_i = log_price_characteristic_function(params, C64(0.0, -1.0), spot, t, r, q)

    p1_int = 0.0
    p2_int = 0.0
    for idx in range(n):
        u = u_lo + du * idx
        kernel = i.scale(-u * log_strike).exp().div(i.scale(u))
        p1_i = (
            kernel.mul(log_price_characteristic_function(params, C64(u, -1.0), spot, t, r, q))
            .div(phi_minus_i)
            .re
        )
        p2_i = kernel.mul(log_price_characteristic_function(params, C64(u, 0.0), spot, t, r, q)).re
        w = 0.5 if idx == 0 or idx == n - 1 else 1.0
        p1_int += w * p1_i * du
        p2_int += w * p2_i * du
    p1 = 0.5 + p1_int / math.pi
    p2 = 0.5 + p2_int / math.pi
    discount_dividend = math.exp(-q * t)
    discount_rate = math.exp(-r * t)
    call_price = spot * discount_dividend * p1 - strike * discount_rate * p2

    if option_style == "call":
        return max(call_price, 0.0)
    return max(call_price - spot * discount_dividend + strike * discount_rate, 0.0)


def monte_carlo_option_price(params, s0, k, t, r, option_type, q, n, steps, seed):
    option_style = option_type.lower()
    if option_style not in ("call", "put"):
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    terminal_prices = simulate_terminal_paths(
        params, s0, t, n, steps, r - q, True, True, seed
    )
    if option_style == "call":
        payoff = sum(max(s - k, 0.0) for s in terminal_prices)
    else:
        payoff = sum(max(k - s, 0.0) for s in terminal_prices)
    return math.exp(-r * t) * payoff / len(terminal_prices)


# ---------------------------------------------------------------------------
# Tests (same literals as quantburn/src/heston.rs)
# ---------------------------------------------------------------------------


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def _params():
    return HestonParams()


def test_cf_price_matches_reference():
    p = _params()
    call = characteristic_function_price(p, 100.0, 100.0, 1.0, 0.05, "call", 0.0, 100.0, 2000)
    put = characteristic_function_price(p, 100.0, 100.0, 1.0, 0.05, "put", 0.0, 100.0, 2000)
    call_otm = characteristic_function_price(p, 100.0, 120.0, 0.5, 0.03, "call", 0.01, 100.0, 2000)
    assert _rel(call, 9.89702247730795) < 1e-10, call
    assert _rel(put, 5.0199649273793625) < 1e-10, put
    assert _rel(call_otm, 0.3987289754945964) < 1e-10, call_otm
    # Zero expiry -> intrinsic.
    intr = characteristic_function_price(p, 110.0, 100.0, 0.0, 0.05, "call", 0.0, 100.0, 256)
    assert abs(intr - 10.0) < 1e-12


def test_simulate_matches_reference():
    out = simulate(_params(), 100.0, 1.0, 6, 32, 2007)
    expected = [
        65.87079325186761,
        78.20202889639539,
        119.95791378057932,
        110.61440405377486,
        87.21994074853231,
        128.78429604873406,
    ]
    for got, want in zip(out, expected):
        assert _rel(got, want) < 1e-12, (got, want)


def test_monte_carlo_price_matches_reference():
    price = monte_carlo_option_price(_params(), 100.0, 100.0, 1.0, 0.05, "call", 0.0, 64, 32, 2007)
    assert _rel(price, 9.586220714087805) < 1e-9, price
    put = monte_carlo_option_price(_params(), 100.0, 100.0, 1.0, 0.05, "put", 0.0, 64, 32, 2007)
    assert _rel(put, 4.709163164159194) < 1e-9, put


def test_tensor_path_matches_scalar_reference():
    """Tensor path on the same normals/params must agree with the scalar f64
    reference (both float64; scalar summation order == jnp sum order here is
    not guaranteed, hence the 1e-9 relative tolerance)."""
    p = _params()
    # Raw (no normalization) — per-path arithmetic must match bit-near-exactly.
    scalar = simulate_terminal_paths(p, 100.0, 1.0, 64, 32, None, False, False, 2007)
    tensor = simulate_terminal_paths_tensor(p, 100.0, 1.0, 64, 32, None, False, False, 2007)
    assert tensor.dtype == jnp.float64
    for i, (g, w) in enumerate(zip(tensor, scalar)):
        assert _rel(float(g), w) < 1e-9, (i, float(g), w)

    # Antithetic + martingale-normalized variant.
    scalar_an = simulate_terminal_paths(p, 100.0, 1.0, 64, 32, 0.05, True, True, 2007)
    tensor_an = simulate_terminal_paths_tensor(p, 100.0, 1.0, 64, 32, 0.05, True, True, 2007)
    for i, (g, w) in enumerate(zip(tensor_an, scalar_an)):
        assert _rel(float(g), w) < 1e-9, (i, float(g), w)


def test_zero_expiry_intrinsic_terminal_paths():
    out = simulate_terminal_paths(_params(), 110.0, 0.0, 8, 32, None, False, False, 2007)
    assert out == [110.0] * 8
    t = simulate_terminal_paths_tensor(_params(), 110.0, 0.0, 8, 32, None, False, False, 2007)
    assert t.tolist() == [110.0] * 8
