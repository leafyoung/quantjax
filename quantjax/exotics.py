"""Port of quantburn/src/exotics.rs (originally torchquantlib
core/asset_pricing/option/exotics.py).

Monte-Carlo pricers for digital, rainbow, barrier, lookback, Asian, quanto
and basket options, plus Curran's Asian approximation and a closed-form
chooser option.

Stochasticity: torch `randn` calls are replaced by
`quantjax.rng.Xorshift64Star` normals with an explicit `seed` parameter per
function. Draw order replicates torch tensor fills: 1-D `(num_paths,)` per
call / per time step, 2-D `(num_paths, n_assets)` row-major. Correlated
draws use the Cholesky factor as `z @ L.T`.

All math is closed-form/scalar float64 in plain Python; panics in the Rust
port (invalid parameters) become ValueError here.
"""

import math
from typing import List, Optional, Sequence

from quantjax.black_scholes_merton import black_scholes_merton, normal_cdf
from quantjax.lmm import cholesky
from quantjax.rng import Xorshift64Star


def _validate_parameters(volatility: float, expiry: float, num_paths: Optional[int]) -> None:
    if volatility <= 0.0 or expiry <= 0.0:
        raise ValueError("Volatility and expiry must be positive.")
    if num_paths is not None and num_paths == 0:
        raise ValueError("Number of paths must be a positive integer.")


def _validate_option_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError("Option type must be either 'call' or 'put'.")


def digital_option(
    option_type: str,
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    rate: float,
    payout: float,
    num_paths: int,
    seed: int,
) -> float:
    """Price a digital (binary cash-or-nothing) option by Monte Carlo."""
    _validate_parameters(volatility, expiry, num_paths)
    _validate_option_type(option_type)

    rng = Xorshift64Star(seed)
    z = rng.normals(num_paths)
    total = 0.0
    for zi in z:
        asset_price = spot * math.exp(
            (rate - 0.5 * volatility * volatility) * expiry
            + volatility * math.sqrt(expiry) * zi
        )
        if (option_type == "call" and asset_price >= strike) or (
            option_type == "put" and asset_price <= strike
        ):
            payoff = payout
        else:
            payoff = 0.0
        total += payoff * math.exp(-rate * expiry)
    return total / num_paths


def rainbow_option(
    spots: Sequence[float],
    weights: Sequence[float],
    strike: float,
    expiry: float,
    volatilities: Sequence[float],
    correlations: Sequence[Sequence[float]],
    rate: float,
    num_paths: int,
    seed: int,
) -> float:
    """Price a rainbow (weighted-basket call) option by Monte Carlo with
    correlated assets via Cholesky."""
    n_assets = len(spots)
    l = cholesky(correlations)
    rng = Xorshift64Star(seed)
    total = 0.0
    for _ in range(num_paths):
        # z row (n_assets), then corr_rand = z @ L.T -> (L @ z)
        z = [rng.next_std_normal() for _ in range(n_assets)]
        prices = []
        for a in range(n_assets):
            corr = sum(l[a][k] * z[k] for k in range(n_assets))
            p = spots[a] * math.exp(
                (rate - 0.5 * volatilities[a] * volatilities[a]) * expiry
                + volatilities[a] * math.sqrt(expiry) * corr
            )
            prices.append(p)
        portfolio = sum(p * w for p, w in zip(prices, weights))
        total += max(portfolio - strike, 0.0) * math.exp(-rate * expiry)
    return total / num_paths


def barrier_option(
    option_type: str,
    barrier_type: str,
    spot: float,
    strike: float,
    barrier: float,
    expiry: float,
    volatility: float,
    rate: float,
    num_paths: int,
    num_steps: int,
    seed: int,
) -> float:
    """Price a barrier option by Monte Carlo (discrete barrier monitoring)."""
    _validate_parameters(volatility, expiry, num_paths)
    _validate_option_type(option_type)
    if barrier_type not in ("up-and-out", "down-and-out", "up-and-in", "down-and-in"):
        raise ValueError("Invalid barrier type")

    dt = expiry / num_steps
    rng = Xorshift64Star(seed)
    drift = (rate - 0.5 * volatility * volatility) * dt
    sig_sqrt_dt = volatility * math.sqrt(dt)
    disc = math.exp(-rate * expiry)

    total = 0.0
    for _ in range(num_paths):
        price = spot
        hit = False
        for _ in range(num_steps):
            z = rng.next_std_normal()
            price *= math.exp(drift + sig_sqrt_dt * z)
            if barrier_type == "up-and-out":
                crossed = price > barrier
            elif barrier_type == "down-and-out":
                crossed = price < barrier
            elif barrier_type == "up-and-in":
                crossed = price >= barrier
            else:  # down-and-in
                crossed = price <= barrier
            if crossed:
                hit = True
        if option_type == "call":
            vanilla = max(price - strike, 0.0)
        else:
            vanilla = max(strike - price, 0.0)
        if barrier_type.endswith("out"):
            payoff = 0.0 if hit else vanilla
        else:
            payoff = vanilla if hit else 0.0
        total += payoff * disc
    return total / num_paths


def lookback_option(
    option_type: str,
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    rate: float,
    num_paths: int,
    strike_type: str,
    seed: int,
) -> float:
    """Price a lookback option by Monte Carlo with daily steps (252 per year,
    matching the Python hard-coded `dt = expiry / 252`)."""
    _validate_parameters(volatility, expiry, num_paths)
    _validate_option_type(option_type)
    if strike_type not in ("fixed", "floating"):
        raise ValueError("Invalid strike type")

    dt = expiry / 252.0
    num_steps = int(expiry / dt)
    rng = Xorshift64Star(seed)
    drift = (rate - 0.5 * volatility * volatility) * dt
    sig_sqrt_dt = volatility * math.sqrt(dt)
    disc = math.exp(-rate * expiry)

    total = 0.0
    for _ in range(num_paths):
        price = spot
        max_p = spot
        min_p = spot
        for _ in range(num_steps):
            z = rng.next_std_normal()
            price *= math.exp(drift + sig_sqrt_dt * z)
            if price > max_p:
                max_p = price
            if price < min_p:
                min_p = price
        if option_type == "call":
            if strike_type == "fixed":
                payoff = max(max_p - strike, 0.0)
            else:
                payoff = max(price - max_p, 0.0)
        else:
            if strike_type == "fixed":
                payoff = max(strike - min_p, 0.0)
            else:
                payoff = max(min_p - price, 0.0)
        total += payoff * disc
    return total / num_paths


def asian_option(
    option_type: str,
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    rate: float,
    num_paths: int,
    num_steps: int,
    average_type: str,
    seed: int,
) -> float:
    """Price an Asian (average-price) option by Monte Carlo. The initial
    spot is included in the average (paths[:, 0] = spot in the Python
    source)."""
    _validate_parameters(volatility, expiry, num_paths)
    _validate_option_type(option_type)
    if average_type not in ("arithmetic", "geometric"):
        raise ValueError("Invalid average type")

    dt = expiry / num_steps
    rng = Xorshift64Star(seed)
    drift = (rate - 0.5 * volatility * volatility) * dt
    sig_sqrt_dt = volatility * math.sqrt(dt)
    disc = math.exp(-rate * expiry)

    total = 0.0
    for _ in range(num_paths):
        price = spot
        log_sum = 0.0
        arith_sum = 0.0
        # Include the initial spot in the average, as paths[:, 0] = spot and
        # the Python mean runs over all num_steps + 1 columns.
        arith_sum += spot
        log_sum += math.log(spot)
        for _ in range(num_steps):
            z = rng.next_std_normal()
            price *= math.exp(drift + sig_sqrt_dt * z)
            arith_sum += price
            log_sum += math.log(price)
        if average_type == "arithmetic":
            avg_price = arith_sum / (num_steps + 1)
        else:
            avg_price = math.exp(log_sum / (num_steps + 1))
        if option_type == "call":
            payoff = max(avg_price - strike, 0.0)
        else:
            payoff = max(strike - avg_price, 0.0)
        total += payoff * disc
    return total / num_paths


def quanto_option(
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    fx_volatility: float,
    correlation: float,
    domestic_rate: float,
    foreign_rate: float,
    fx_rate: float,
    num_paths: int,
    seed: int,
) -> float:
    """Price a quanto option by Monte Carlo (asset in foreign currency,
    payoff converted at simulated FX rate)."""
    _validate_parameters(volatility, expiry, num_paths)

    rng = Xorshift64Star(seed)
    total = 0.0
    for _ in range(num_paths):
        z1 = rng.next_std_normal()
        z2 = correlation * z1 + math.sqrt(1.0 - correlation * correlation) * rng.next_std_normal()
        asset_price = spot * math.exp(
            (foreign_rate - 0.5 * volatility * volatility) * expiry
            + volatility * math.sqrt(expiry) * z1
        )
        fx = fx_rate * math.exp(
            (domestic_rate - foreign_rate - 0.5 * fx_volatility * fx_volatility) * expiry
            + fx_volatility * math.sqrt(expiry) * z2
        )
        total += max(asset_price - strike, 0.0) * fx * math.exp(-domestic_rate * expiry)
    return total / num_paths


def basket_option(
    spots: Sequence[float],
    weights: Sequence[float],
    strike: float,
    expiry: float,
    volatilities: Sequence[float],
    correlations: Sequence[Sequence[float]],
    rate: float,
    option_type: str,
    num_paths: int,
    seed: int,
) -> float:
    """Price a basket option by Monte Carlo with correlated assets."""
    _validate_parameters(min(volatilities), expiry, num_paths)
    _validate_option_type(option_type)

    n_assets = len(spots)
    l = cholesky(correlations)
    rng = Xorshift64Star(seed)
    disc = math.exp(-rate * expiry)
    total = 0.0
    for _ in range(num_paths):
        z = [rng.next_std_normal() for _ in range(n_assets)]
        basket_value = 0.0
        for a in range(n_assets):
            corr = sum(l[a][k] * z[k] for k in range(n_assets))
            p = spots[a] * math.exp(
                (rate - 0.5 * volatilities[a] * volatilities[a]) * expiry
                + volatilities[a] * math.sqrt(expiry) * corr
            )
            basket_value += p * weights[a]
        if option_type == "call":
            payoff = max(basket_value - strike, 0.0)
        else:
            payoff = max(strike - basket_value, 0.0)
        total += payoff * disc
    return total / num_paths


class CurranAsianOption:
    """Curran's approximation for an Asian option (port of
    `CurranAsianOption`)."""

    def __init__(self, option_type: str, average_type: str):
        _validate_option_type(option_type)
        self.option_type = option_type
        self.average_type = average_type

    def forward(
        self,
        spot: float,
        strike: float,
        expiry: float,
        volatility: float,
        rate: float,
        num_steps: int,
    ) -> float:
        _validate_parameters(volatility, expiry, None)
        adjusted_vol = volatility * math.sqrt((num_steps + 1) / (6.0 * num_steps))

        if self.average_type == "arithmetic":
            d1 = (
                math.log(spot / strike)
                + (rate + 0.5 * adjusted_vol * adjusted_vol) * expiry
            ) / (adjusted_vol * math.sqrt(expiry))
            d2 = d1 - adjusted_vol * math.sqrt(expiry)
            if self.option_type == "call":
                return (
                    spot * math.exp(-rate * expiry) * normal_cdf(d1)
                    - strike * math.exp(-rate * expiry) * normal_cdf(d2)
                )
            return (
                strike * math.exp(-rate * expiry) * normal_cdf(-d2)
                - spot * math.exp(-rate * expiry) * normal_cdf(-d1)
            )
        # Analytical solution for the geometric average.
        adjusted_rate = rate - 0.5 * volatility * volatility / 3.0
        return black_scholes_merton(
            self.option_type,
            "european",
            spot,
            strike,
            expiry,
            adjusted_vol,
            adjusted_rate,
            0.0,
        )


def chooser_option(
    spot: float,
    strike: float,
    expiry: float,
    volatility: float,
    rate: float,
    choose_time: float,
) -> float:
    """Price a chooser option with the Black-Scholes max(call, put) formula,
    discounted over the choose time."""
    _validate_parameters(volatility, expiry, None)

    t = expiry - choose_time
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * volatility * volatility) * t
    ) / (volatility * math.sqrt(t))
    d2 = d1 - volatility * math.sqrt(t)

    call_price = spot * normal_cdf(d1) - strike * math.exp(-rate * t) * normal_cdf(d2)
    put_price = strike * math.exp(-rate * t) * normal_cdf(-d2) - spot * normal_cdf(-d1)

    return max(call_price, put_price) * math.exp(-rate * choose_time)


# ---------------------------------------------------------------------------
# Embedded parity tests (expected literals from quantburn/src/exotics.rs,
# generated by scripts/gen_exotics.py with the same seeds)
# ---------------------------------------------------------------------------


def test_digital_call():
    v = digital_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 10.0, 1000, 1234)
    assert abs(v - 5.345909365693983) < 1e-9


def test_rainbow_call():
    spots = [100.0, 90.0, 80.0]
    weights = [0.5, 0.3, 0.2]
    vols = [0.2, 0.25, 0.3]
    corr = [
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ]
    v = rainbow_option(spots, weights, 90.0, 1.0, vols, corr, 0.05, 500, 1235)
    assert abs(v - 11.414732712382342) < 1e-9


def test_barrier_up_and_out_call():
    v = barrier_option(
        "call", "up-and-out", 100.0, 100.0, 120.0, 1.0, 0.2, 0.05, 500, 50, 1236
    )
    assert abs(v - 1.4789899008215022) < 1e-9


def test_barrier_down_and_in_put():
    v = barrier_option(
        "put", "down-and-in", 100.0, 100.0, 85.0, 1.0, 0.2, 0.05, 500, 50, 1237
    )
    assert abs(v - 4.46778044843966) < 1e-9


def test_lookback_fixed_call():
    v = lookback_option("call", 100.0, 100.0, 0.5, 0.2, 0.05, 200, "fixed", 1238)
    assert abs(v - 12.790728299383302) < 1e-9


def test_lookback_fixed_put():
    # Note: the floating-strike branches (both call and put) have payoffs
    # max(S_T - max(S), 0) / max(min(S) - S_T, 0) that are identically 0 —
    # a quirk of the Python source (they use the wrong extremum), which the
    # port preserves. The fixed-strike put is the only other non-degenerate
    # combination, so it replaces the old vacuous floating-put test.
    # Reference: scripts/gen_exotics.py, seed 1239.
    v = lookback_option("put", 100.0, 100.0, 0.5, 0.2, 0.05, 200, "fixed", 1239)
    assert abs(v - 9.323267890110916) < 1e-9


def test_asian_mc_arithmetic():
    v = asian_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 1000, 50, "arithmetic", 1240)
    assert abs(v - 5.455662023386994) < 1e-9


def test_asian_mc_geometric():
    v = asian_option("put", 100.0, 100.0, 1.0, 0.2, 0.05, 1000, 50, "geometric", 1241)
    assert abs(v - 3.222490145778699) < 1e-9


def test_quanto_call():
    v = quanto_option(100.0, 100.0, 1.0, 0.2, 0.1, 0.3, 0.05, 0.02, 7.8, 1000, 1242)
    assert abs(v - 70.22137345027403) < 1e-9


def test_basket_put():
    spots = [100.0, 50.0]
    weights = [0.6, 0.4]
    vols = [0.2, 0.3]
    corr = [[1.0, 0.4], [0.4, 1.0]]
    v = basket_option(spots, weights, 75.0, 1.0, vols, corr, 0.05, "put", 1000, 1243)
    assert abs(v - 2.5309404121395453) < 1e-9


def test_curran_arithmetic_call():
    c = CurranAsianOption("call", "arithmetic")
    v = c.forward(100.0, 100.0, 1.0, 0.2, 0.05, 252)
    assert abs(v - 2.575286024338297) < 1e-9


def test_curran_geometric_put():
    c = CurranAsianOption("put", "geometric")
    v = c.forward(100.0, 100.0, 1.0, 0.2, 0.05, 252)
    assert abs(v - 1.5108036415253139) < 1e-9


def test_chooser_call_dominates():
    v = chooser_option(100.0, 100.0, 1.0, 0.2, 0.05, 0.5)
    assert abs(v - 6.71864526308475) < 1e-9


def test_chooser_put_dominates():
    v = chooser_option(80.0, 100.0, 1.0, 0.2, 0.05, 0.5)
    assert abs(v - 17.54304177655884) < 1e-9
