"""SABR model with Hagan asymptotics — f64 port of quantburn/src/sabr.rs
(originally torchquantlib/models/stochastic_volatility/sabr.py).

Pure scalar float64 math; no jax. Module RNG seed: 2008 (normals from
`quantjax.rng.Xorshift64Star`, per step a block of N spot normals followed by
N auxiliary normals).
"""

import math
from dataclasses import dataclass

from quantjax.rng import Xorshift64Star

EPS = 1e-8
ATM_TOL = 1e-7


def erf(x: float) -> float:
    """Gauss error function: Maclaurin series for |x| < 3, DLMF 7.9
    continued fraction beyond."""
    if x == 0.0:
        return 0.0
    ax = abs(x)
    if ax < 3.0:
        x2 = x * x
        term = ax
        s = ax
        n = 1
        while True:
            term *= -x2 / n
            add = term / (2.0 * n + 1.0)
            s += add
            if abs(add) <= 1e-18 * abs(s) or n > 300:
                break
            n += 1
        r = s * 2.0 / math.sqrt(math.pi)
    else:
        r = 1.0 - _erfc(ax)
    return -r if x < 0.0 else r


def _erfc(x: float) -> float:
    """Complementary error function via the Laplace continued fraction
    (DLMF 7.9): sqrt(pi) e^{x^2} erfc(x) = x + (1/2)/(x + 1/(x + (3/2)/(x+...)))."""
    val = x
    for n in range(200, 0, -1):
        val = x + (n / 2.0) / val
    return math.exp(-x * x) / (math.sqrt(math.pi) * val)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black76_price(option_type, forward, strike, expiry, volatility, rate):
    discount_factor = math.exp(-rate * expiry)
    intrinsic_value = (
        max(forward - strike, 0.0) if option_type == "call" else max(strike - forward, 0.0)
    )
    if expiry <= 0.0 or volatility <= 0.0:
        return discount_factor * intrinsic_value
    sqrt_expiry = math.sqrt(expiry)
    volatility_term = volatility * sqrt_expiry
    d1 = (math.log(forward / strike) + 0.5 * volatility * volatility * expiry) / volatility_term
    d2 = d1 - volatility_term
    if option_type == "call":
        return discount_factor * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
    return discount_factor * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))


def bachelier_price(option_type, forward, strike, expiry, volatility, rate):
    discount_factor = math.exp(-rate * expiry)
    intrinsic_value = (
        max(forward - strike, 0.0) if option_type == "call" else max(strike - forward, 0.0)
    )
    if expiry <= 0.0 or volatility <= 0.0:
        return discount_factor * intrinsic_value
    sqrt_expiry = math.sqrt(expiry)
    denominator = max(volatility * sqrt_expiry, EPS)
    d = (forward - strike) / denominator
    call_price = discount_factor * (
        (forward - strike) * normal_cdf(d) + volatility * sqrt_expiry * normal_pdf(d)
    )
    if option_type == "call":
        return call_price
    return call_price - discount_factor * (forward - strike)


@dataclass
class SabrParams:
    alpha: float = 0.2
    beta: float = 0.5
    rho: float = 0.0
    nu: float = 0.3
    f0: float = 100.0
    shift: float = 0.0


def _resolve_shift(params, shift):
    return params.shift if shift is None else shift


def base_forward(params, base_level, expiry, rate, dividend):
    """F0 (or given level), grown by the cost of carry when a rate is supplied."""
    forward = params.f0 if base_level is None else base_level
    if rate is None:
        return forward
    return forward * math.exp((rate - dividend) * expiry)


def atm_lognormal_implied_volatility(params, forward, expiry):
    forward = max(forward, EPS)
    one_minus_beta = 1.0 - params.beta
    forward_power = forward ** one_minus_beta
    correction = (
        ((one_minus_beta * one_minus_beta) / 24.0)
        * (params.alpha * params.alpha / max(forward ** (2.0 * one_minus_beta), EPS))
        + (params.rho * params.beta * params.nu * params.alpha) / (4.0 * max(forward_power, EPS))
        + ((2.0 - 3.0 * params.rho * params.rho) / 24.0) * params.nu * params.nu
    )
    return max((params.alpha / max(forward_power, EPS)) * (1.0 + correction * expiry), EPS)


def hagan_lognormal_implied_volatility(params, strike, expiry, forward=None, shift=None):
    if expiry <= 0.0:
        return 0.0
    bf = base_forward(params, params.f0 if forward is None else forward, expiry, None, 0.0)
    shift_v = _resolve_shift(params, shift)
    shifted_forward = max(bf + shift_v, EPS)
    shifted_strike = max(strike + shift_v, EPS)

    log_fk = math.log(shifted_forward / shifted_strike)
    if abs(log_fk) < ATM_TOL:
        return atm_lognormal_implied_volatility(params, shifted_forward, expiry)

    one_minus_beta = 1.0 - params.beta
    fk_beta = max((shifted_forward * shifted_strike) ** (0.5 * one_minus_beta), EPS)
    log_fk_sq = log_fk * log_fk
    log_fk_4 = log_fk_sq * log_fk_sq

    z = (params.nu / max(params.alpha, EPS)) * fk_beta * log_fk
    sqrt_term = math.sqrt(max(1.0 - 2.0 * params.rho * z + z * z, EPS))
    x_z = math.log(max((sqrt_term + z - params.rho) / (1.0 - params.rho), EPS))

    log_denominator = (
        1.0
        + ((one_minus_beta * one_minus_beta) / 24.0) * log_fk_sq
        + ((one_minus_beta ** 4) / 1920.0) * log_fk_4
    )
    prefactor = params.alpha / max(fk_beta * log_denominator, EPS)
    correction = (
        ((one_minus_beta * one_minus_beta) / 24.0)
        * (params.alpha * params.alpha / max((shifted_forward * shifted_strike) ** one_minus_beta, EPS))
        + (params.rho * params.beta * params.nu * params.alpha) / (4.0 * max(fk_beta, EPS))
        + ((2.0 - 3.0 * params.rho * params.rho) / 24.0) * params.nu * params.nu
    )

    ratio = 1.0 if abs(x_z) < ATM_TOL else z / x_z
    return max(prefactor * ratio * (1.0 + correction * expiry), EPS)


def atm_normal_implied_volatility(params, forward, expiry):
    forward = max(forward, EPS)
    forward_beta = forward ** params.beta
    one_minus_beta = 1.0 - params.beta
    correction = (
        -params.beta * (2.0 - params.beta) * params.alpha * params.alpha
        / (24.0 * max(forward ** (2.0 * one_minus_beta), EPS))
        + (params.rho * params.beta * params.nu * params.alpha)
        / (4.0 * max(forward ** one_minus_beta, EPS))
        + ((2.0 - 3.0 * params.rho * params.rho) / 24.0) * params.nu * params.nu
    )
    return max(params.alpha * forward_beta * (1.0 + correction * expiry), EPS)


def hagan_normal_implied_volatility(params, strike, expiry, forward=None, shift=None):
    if expiry <= 0.0:
        return 0.0
    bf = base_forward(params, params.f0 if forward is None else forward, expiry, None, 0.0)
    shift_v = _resolve_shift(params, shift)
    shifted_forward = bf + shift_v
    shifted_strike = strike + shift_v

    diff = shifted_forward - shifted_strike
    if abs(diff) < ATM_TOL:
        return atm_normal_implied_volatility(params, max(shifted_forward, EPS), expiry)

    fk_beta = max(max(shifted_forward * shifted_strike, EPS) ** (0.5 * params.beta), EPS)
    zeta = (params.nu / max(params.alpha, EPS)) * diff / fk_beta
    sqrt_term = math.sqrt(max(1.0 - 2.0 * params.rho * zeta + zeta * zeta, EPS))
    x_zeta = math.log(max((sqrt_term + zeta - params.rho) / (1.0 - params.rho), EPS))

    prod = max(shifted_forward * shifted_strike, EPS)
    correction = (
        -params.beta * (2.0 - params.beta) * params.alpha * params.alpha
        / (24.0 * max(prod ** (1.0 - params.beta), EPS))
        + (params.rho * params.beta * params.nu * params.alpha)
        / (4.0 * max(prod ** (0.5 * (1.0 - params.beta)), EPS))
        + ((2.0 - 3.0 * params.rho * params.rho) / 24.0) * params.nu * params.nu
    )
    ratio = 1.0 if abs(x_zeta) < ATM_TOL else diff / x_zeta
    return max(params.alpha * ratio * (1.0 + correction * expiry), EPS)


def implied_volatility(params, k, t, r, q, model, shift=None):
    forward = base_forward(params, None, t, r, q)
    m = model.lower()
    if m == "lognormal":
        return hagan_lognormal_implied_volatility(params, k, t, forward, shift)
    if m == "normal":
        return hagan_normal_implied_volatility(params, k, t, forward, shift)
    raise ValueError("model must be 'lognormal' or 'normal'")


def simulate(params, s0, t, n, steps, seed):
    """Euler-style SABR sampler for terminal forwards. Per step draws a block
    of N spot normals followed by N auxiliary normals."""
    steps = max(steps, 1)
    dt = t / steps
    sqrt_dt = math.sqrt(dt)
    rho_residual = math.sqrt(min(max(1.0 - params.rho * params.rho, EPS), 1.7976931348623157e308))

    start = max(base_forward(params, s0, t, None, 0.0), EPS)
    forward = [start] * n
    alpha_t = [params.alpha] * n
    rng = Xorshift64Star(seed)

    for _ in range(steps):
        z1 = [rng.next_std_normal() for _ in range(n)]
        z2 = [rng.next_std_normal() for _ in range(n)]
        for i in range(n):
            z_alpha = params.rho * z1[i] + rho_residual * z2[i]
            forward_base = max(forward[i], EPS)
            diffusion = alpha_t[i] * forward_base ** params.beta * sqrt_dt * z1[i]
            forward[i] = max(forward[i] + diffusion, EPS)
            alpha_t[i] = max(
                alpha_t[i] * math.exp(-0.5 * params.nu * params.nu * dt + params.nu * sqrt_dt * z_alpha),
                EPS,
            )
    return forward


def option_price(params, k, t, r, option_type, q, model, shift=None):
    option_style = option_type.lower()
    if option_style not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    forward = base_forward(params, None, t, r, q)
    m = model.lower()
    if m == "lognormal":
        shift_v = _resolve_shift(params, shift)
        iv = hagan_lognormal_implied_volatility(params, k, t, forward, shift)
        return black76_price(option_style, forward + shift_v, k + shift_v, t, iv, r)
    if m == "normal":
        shift_v = _resolve_shift(params, shift)
        iv = hagan_normal_implied_volatility(params, k, t, forward, shift)
        return bachelier_price(option_style, forward + shift_v, k + shift_v, t, iv, r)
    raise ValueError("model must be 'lognormal' or 'normal'")


# ---------------------------------------------------------------------------
# Tests (same literals as quantburn/src/sabr.rs)
# ---------------------------------------------------------------------------


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def _params():
    return SabrParams()


def test_implied_vols_match_reference():
    p = _params()
    iv_ln = implied_volatility(p, 110.0, 1.0, 0.0, 0.0, "lognormal")
    iv_ln_atm = implied_volatility(p, 100.0, 1.0, 0.0, 0.0, "lognormal")
    iv_n = implied_volatility(p, 110.0, 1.0, 0.0, 0.0, "normal")
    iv_n_atm = implied_volatility(p, 100.0, 1.0, 0.0, 0.0, "normal")
    assert _rel(iv_ln, 0.02452109699988402) < 1e-12, iv_ln
    assert _rel(iv_ln_atm, 0.020150083333333332) < 1e-12, iv_ln_atm
    assert _rel(iv_n, 1.7148575695087824) < 1e-12, iv_n
    assert _rel(iv_n_atm, 2.014975) < 1e-12, iv_n_atm


def test_option_prices_match_reference():
    p = _params()
    call_ln = option_price(p, 105.0, 1.0, 0.03, "call", 0.0, "lognormal")
    put_ln = option_price(p, 95.0, 1.0, 0.03, "put", 0.0, "lognormal")
    call_n = option_price(p, 105.0, 1.0, 0.03, "call", 0.0, "normal")
    shifted = option_price(p, 0.01, 1.0, 0.0, "call", 0.0, "lognormal", shift=0.05)
    assert _rel(call_ln, 0.1890807767303359) < 1e-12, call_ln
    assert _rel(put_ln, 0.00020839856139982785) < 1e-8, put_ln
    assert _rel(call_n, 0.04857832011877204) < 1e-12, call_n
    assert _rel(shifted, 99.99) < 1e-12, shifted


def test_simulate_matches_reference():
    p = _params()
    out = simulate(p, None, 1.0, 6, 32, 2008)
    expected = [
        104.35503149972567,
        99.53200457084024,
        99.22584645372046,
        101.30415922762192,
        99.03245350317181,
        100.59543369067899,
    ]
    for got, want in zip(out, expected):
        assert _rel(got, want) < 1e-11, (got, want)
