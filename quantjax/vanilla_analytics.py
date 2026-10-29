"""Analytic Black-Scholes barrier and binary option prices, ported from
tf-quant-finance `black_scholes/vanilla_prices.py` (`barrier_price` and
`binary_price`).

`barrier_price_analytic` is the continuous-monitoring single-barrier closed
form (Reiner-Rubinstein / Merton-style, the Clewlow-Strickland / Haug mask
formulation used by tff): 12 terms, each an algebraic factor times a normal
CDF, selected by barrier type and by whether the strike is above the barrier.
`binary_price` is the cash-or-nothing binary paying one unit of cash, with the
dividend-corrected forward ``F = S * exp(-q*T) / DF`` (the FIXED tff version;
ignoring q mis-prices binaries when q > 0).

Pure float64 scalar math (plain Python floats, no jax) mirroring the tff
formulas branch-for-branch, including the rebate terms and the
``sqrt_var > 0`` clamp of the binary price. Values verified against QuantLib
(AnalyticBarrierEngine on BarrierOption, CashOrNothingPayoff +
AnalyticEuropeanEngine) and against tff itself in float64; embedded literals
below come from those runs.

Rebate conventions: by default (`rebate_paid_at_expiry=False`, tff
semantics) a knock-out rebate is paid at hit and a knock-in rebate at
expiry. With `rebate_paid_at_expiry=True` a knock-out rebate is instead
paid at expiry iff the barrier is never touched (the standard
Reiner-Rubinstein knock-out-rebate-at-expiry term `R * DF * P(survival)`,
obtained by re-weighting the same survival-probability terms the knock-in
rows use); knock-in results never depend on the flag.

QuantLib cross-check (AnalyticBarrierEngine, QL 1.44): QL's knock-in rebate
equals the survival term above to <=2e-15, and QL's knock-out rebate is paid
at hit with the Haug `lam = sqrt(mu^2 + 2r/sigma^2)` parameterization
(matches QL to <=2e-15), which differs from tff's `b`-parameterized
hit-time term (e.g. by +0.0298 on the S=100 K=100 B=85 R=3 vol=25% T=1
case). QL knock-out totals therefore match neither path; the
`rebate_paid_at_expiry` knock-out test literals are composed from pure QL
numbers, `QL KO(R=0) + QL KI rebate term`, which equals main + survival
term exactly (engine agreement <=5.3e-15).
"""

import math

from quantjax.black_scholes_merton import black_scholes_merton, normal_cdf

# Mask rows for strike > barrier, indexed by
# (is_barrier_down << 2) + (is_knock_out << 1) + is_call
# (same rows and order as tff's mask_matrix_greater_strike).
_MASK_GREATER_STRIKE = (
    (1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0),  # up and in put
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),  # up and in call
    (0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1),  # up and out put
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),  # up and out call
    (0, 0, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1),  # down and in put
    (0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),  # down and in call
    (1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1),  # down and out put
    (1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1),  # down and out call
)

# Mask rows for strike <= barrier (tff's mask_matrix_lower_strike).
_MASK_LOWER_STRIKE = (
    (0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),  # up and in put
    (0, 0, 1, 1, -1, -1, 1, 1, 1, 1, 0, 0),  # up and in call
    (1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1),  # up and out put
    (1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1),  # up and out call
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),  # down and in put
    (1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0),  # down and in call
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),  # down and out put
    (0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1),  # down and out call
)


def barrier_price_analytic(
    volatility: float,
    strike: float,
    expiry: float,
    spot: float,
    barrier: float,
    rebate: float,
    discount_rate: float,
    dividend_rate: float,
    is_barrier_down: bool,
    is_knock_out: bool,
    is_call: bool,
    rebate_paid_at_expiry: bool = False,
) -> float:
    """Price a continuously-monitored single-barrier option in closed form
    (Black-Scholes, tff ``barrier_price`` semantics).

    Arguments follow the tff keyword names: ``discount_rate`` is the risk-free
    rate r, ``dividend_rate`` the continuous dividend yield q. The barrier
    must not be breached at inception (down barrier below ``spot``, up barrier
    above ``spot``), matching the formula's domain. ``rebate`` is the fixed
    cash payout if a knock-out barrier is breached (paid at hit) or if a
    knock-in barrier is never breached (paid at expiry).
    ``rebate_paid_at_expiry`` only affects knock-outs: with ``True`` the
    rebate is paid at expiry iff the barrier is never touched instead of at
    hit; knock-in rebates are paid at expiry in both conventions, so knock-in
    results are identical under the flag.
    """
    index = (
        ((1 if is_barrier_down else 0) << 2)
        + ((1 if is_knock_out else 0) << 1)
        + (1 if is_call else 0)
    )
    masks = _MASK_GREATER_STRIKE[index] if strike > barrier else _MASK_LOWER_STRIKE[index]
    if is_knock_out and rebate_paid_at_expiry:
        # Knock-out with the rebate paid at expiry: move the rebate weight
        # from the hit-time terms (columns 11-12) onto the survival
        # probability terms (columns 9-10) that the knock-in rows use.
        masks = masks[:8] + (masks[10], masks[11], 0, 0)

    call_or_put = 1.0 if is_call else -1.0
    below_or_above = 1.0 if is_barrier_down else -1.0

    # Parameters for the six integrals (tff names, minus the unicode).
    sqrt_var = volatility * math.sqrt(expiry)
    mu = (discount_rate - dividend_rate) - (volatility**2) / 2.0
    lamda = 1.0 + mu / (volatility**2)
    x = math.log(spot / strike) / sqrt_var + lamda * sqrt_var
    x1 = math.log(spot / barrier) / sqrt_var + lamda * sqrt_var
    y = math.log(barrier**2 / (spot * strike)) / sqrt_var + lamda * sqrt_var
    y1 = math.log(barrier / spot) / sqrt_var + lamda * sqrt_var
    b = (mu**2 + 2.0 * (volatility**2) * discount_rate) / (volatility**2)
    z = math.log(barrier / spot) / sqrt_var + b * sqrt_var
    a = mu / (volatility**2)

    discount_factors = math.exp(-discount_rate * expiry)
    barriers_ratio = barrier / spot
    spots_term = call_or_put * spot * math.exp(-dividend_rate * expiry)
    strikes_term = call_or_put * strike * discount_factors

    terms = (
        spots_term,
        -strikes_term,
        spots_term,
        -strikes_term,
        spots_term * (barriers_ratio ** (2 * lamda)),
        -strikes_term * (barriers_ratio ** (2 * lamda - 2)),
        spots_term * (barriers_ratio ** (2 * lamda)),
        -strikes_term * (barriers_ratio ** (2 * lamda - 2)),
        rebate * discount_factors,
        -rebate * discount_factors * (barriers_ratio ** (2 * lamda - 2)),
        rebate * (barriers_ratio ** (a + b)),
        rebate * (barriers_ratio ** (a - b)),
    )
    cdfs = (
        normal_cdf(call_or_put * x),
        normal_cdf(call_or_put * (x - sqrt_var)),
        normal_cdf(call_or_put * x1),
        normal_cdf(call_or_put * (x1 - sqrt_var)),
        normal_cdf(below_or_above * y),
        normal_cdf(below_or_above * (y - sqrt_var)),
        normal_cdf(below_or_above * y1),
        normal_cdf(below_or_above * (y1 - sqrt_var)),
        normal_cdf(below_or_above * (x1 - sqrt_var)),
        normal_cdf(below_or_above * (y1 - sqrt_var)),
        normal_cdf(below_or_above * z),
        normal_cdf(below_or_above * (z - 2 * b * sqrt_var)),
    )
    return sum(m * t * c for m, t, c in zip(masks, terms, cdfs))


def binary_price(
    volatility: float,
    strike: float,
    expiry: float,
    spot: float,
    discount_rate: float,
    dividend_rate: float,
    is_call: bool,
) -> float:
    """Price a cash-or-nothing binary option paying one unit of cash at
    expiry (Black-Scholes, tff ``binary_price`` semantics with the
    dividend-corrected forward ``F = S * exp(-q*T) / DF``).

    The call pays 1 if the spot ends above the strike, the put 1 if below, so
    the price is the discounted risk-neutral probability of ending in the
    money. Zero volatility collapses to the discounted step payoff (tff's
    ``sqrt_var > 0`` clamp).
    """
    discount_factors = math.exp(-discount_rate * expiry)
    forwards = spot * math.exp(-dividend_rate * expiry) / discount_factors
    sqrt_var = volatility * math.sqrt(expiry)
    if sqrt_var > 0.0:
        d2 = math.log(forwards / strike) / sqrt_var - sqrt_var / 2.0
        undiscounted_calls = normal_cdf(d2)
    else:
        undiscounted_calls = 1.0 if forwards > strike else 0.0
    if is_call:
        return discount_factors * undiscounted_calls
    return discount_factors * (1.0 - undiscounted_calls)


# --- tests (literals verified against QuantLib AnalyticBarrierEngine /
# CashOrNothingPayoff + AnalyticEuropeanEngine and tff float64; see the
# cross-check harness, 2026-10) ---

_R = 0.05
_Q = 0.02


def test_barrier_golden_values():
    # down-out B=85 K=100 call vol=20% T=1, S=100
    v = barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 85.0, 0.0, _R, _Q, True, True, True)
    assert abs(v - 8.759460458735848) < 1e-10
    # up-out B=125 K=100 put vol=25% T=1
    v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 125.0, 0.0, _R, _Q, False, True, False)
    assert abs(v - 7.902600854338467) < 1e-10
    # down-in B=85 K=110 call vol=25% T=1
    v = barrier_price_analytic(0.25, 110.0, 1.0, 100.0, 85.0, 0.0, _R, _Q, True, False, True)
    assert abs(v - 0.5479573225762513) < 1e-10
    # up-in B=125 K=90 put vol=20% T=1
    v = barrier_price_analytic(0.20, 90.0, 1.0, 100.0, 125.0, 0.0, _R, _Q, False, False, False)
    assert abs(v - 0.013422992543647733) < 1e-12


def test_barrier_extra_ql_grid():
    # down-out call, K > B, vol=30%
    v = barrier_price_analytic(0.30, 95.0, 1.0, 100.0, 90.0, 0.0, _R, _Q, True, True, True)
    assert abs(v - 9.711137714182236) < 1e-10
    # up-in call, deep OTM K=140
    v = barrier_price_analytic(0.25, 140.0, 1.0, 100.0, 120.0, 0.0, _R, _Q, False, False, True)
    assert abs(v - 1.5111745977832367) < 1e-11
    # up-out put, K > B
    v = barrier_price_analytic(0.25, 130.0, 1.0, 100.0, 125.0, 0.0, _R, _Q, False, True, False)
    assert abs(v - 24.5605437578551) < 1e-10
    # down-in put, K < B, T=2, vol=30%
    v = barrier_price_analytic(0.30, 75.0, 2.0, 100.0, 85.0, 0.0, _R, _Q, True, False, False)
    assert abs(v - 3.925672050043495) < 1e-11
    # down-out put, K < B: barrier below both spot and strike region -> 0
    v = barrier_price_analytic(0.25, 80.0, 1.0, 100.0, 85.0, 0.0, _R, _Q, True, True, False)
    assert abs(v) < 1e-12
    # up-in call, S=90, T=2, vol=40%
    v = barrier_price_analytic(0.40, 110.0, 2.0, 90.0, 105.0, 0.0, _R, _Q, False, False, True)
    assert abs(v - 14.800183620042578) < 1e-11


def test_barrier_in_plus_out_equals_vanilla():
    # down barrier, call
    pin = barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 80.0, 0.0, _R, _Q, True, False, True)
    pout = barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 80.0, 0.0, _R, _Q, True, True, True)
    vanilla = black_scholes_merton("call", "european", 100.0, 100.0, 1.0, 0.20, _R, _Q)
    assert abs(pin + pout - vanilla) < 1e-10

    # up barrier, put, T=2
    pin = barrier_price_analytic(0.25, 100.0, 2.0, 100.0, 120.0, 0.0, _R, _Q, False, False, False)
    pout = barrier_price_analytic(0.25, 100.0, 2.0, 100.0, 120.0, 0.0, _R, _Q, False, True, False)
    vanilla = black_scholes_merton("put", "european", 100.0, 100.0, 2.0, 0.25, _R, _Q)
    assert abs(pin + pout - vanilla) < 1e-10


def test_barrier_with_rebate():
    # down-out call, rebate 3 paid at hit (tff semantics)
    v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 85.0, 3.0, _R, _Q, True, True, True)
    assert abs(v - 11.475189709298022) < 1e-10
    # down-in call, rebate 1.5 paid at expiry if never knocked in
    v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 85.0, 1.5, _R, _Q, True, False, True)
    assert abs(v - 1.8890581732854028) < 1e-11


def test_barrier_ko_rebate_paid_at_expiry():
    # Knock-out rebate paid at expiry iff the barrier is never touched:
    # R * DF * P(survival) (standard Reiner-Rubinstein KO-rebate-at-expiry
    # terms). QL's AnalyticBarrierEngine pays KO rebates at hit (Haug lam
    # parameterization), so these literals are composed from pure QL numbers:
    # QL KO(R=0) + QL KI rebate term (== the survival term); the engine
    # reproduces them to <= 5.3e-15.
    # known case: down-out call S=100 K=100 B=85 R=3 vol=25% T=1, r=5% q=2%
    v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 85.0, 3.0, _R, _Q,
                               True, True, True, rebate_paid_at_expiry=True)
    assert abs(v - 11.300833094739227) < 1e-10
    # second QL grid point: up-out put S=100 K=90 B=112 R=4 vol=25% T=0.6,
    # r=3% q=1%
    v = barrier_price_analytic(0.25, 90.0, 0.6, 100.0, 112.0, 4.0, 0.03, 0.01,
                               False, True, False, rebate_paid_at_expiry=True)
    assert abs(v - 4.509289715741195) < 1e-10


def test_barrier_rebate_flag_ki_invariance():
    # Knock-in rebates are paid at expiry in both conventions, so knock-in
    # results are bit-identical under the flag (all eight KI mask rows:
    # up/down in, call/put, K > B and K <= B).
    cases = [
        (0.25, 130.0, 1.0, 100.0, 125.0, 2.0, _R, _Q, False, False, False),   # up-in put, K > B
        (0.25, 140.0, 1.0, 100.0, 120.0, 1.0, _R, _Q, False, False, True),    # up-in call, K > B
        (0.22, 95.0, 0.8, 100.0, 88.0, 2.0, 0.06, 0.01, True, False, False),  # down-in put, K > B
        (0.25, 110.0, 1.0, 100.0, 85.0, 1.5, _R, _Q, True, False, True),      # down-in call, K > B
        (0.25, 90.0, 0.6, 100.0, 112.0, 4.0, 0.03, 0.01, False, False, False),  # up-in put, K < B
        (0.25, 100.0, 0.6, 100.0, 112.0, 4.0, 0.03, 0.01, False, False, True),  # up-in call, K < B
        (0.22, 80.0, 0.8, 100.0, 88.0, 2.0, 0.06, 0.01, True, False, False),  # down-in put, K < B
        (0.25, 80.0, 1.0, 100.0, 85.0, 1.5, _R, _Q, True, False, True),       # down-in call, K < B
    ]
    for args in cases:
        assert barrier_price_analytic(*args, rebate_paid_at_expiry=False) == \
            barrier_price_analytic(*args, rebate_paid_at_expiry=True), args


def test_barrier_rebate_flag_default_unchanged():
    # Default (False) path stays on the tff hit-timing convention; the flag
    # genuinely moves the KO rebate term.
    args = (0.25, 100.0, 1.0, 100.0, 85.0, 3.0, _R, _Q, True, True, True)
    hit = barrier_price_analytic(*args)
    assert abs(hit - 11.475189709298022) < 1e-10
    expiry = barrier_price_analytic(*args, rebate_paid_at_expiry=True)
    assert abs(expiry - 11.300833094739227) < 1e-10
    assert abs(hit - expiry) > 0.1


def test_binary_golden_values():
    # ATM call, K=100, T=1, vol=20%
    v = binary_price(0.20, 100.0, 1.0, 100.0, _R, _Q, True)
    assert abs(v - 0.49458109105322323) < 1e-12
    # put, K=110, T=2, vol=30%
    v = binary_price(0.30, 110.0, 2.0, 100.0, _R, _Q, False)
    assert abs(v - 0.557506659561213) < 1e-12


def test_binary_extra_ql_grid():
    # deep ITM call, K=70
    v = binary_price(0.20, 70.0, 1.0, 100.0, _R, _Q, True)
    assert abs(v - 0.9194836337641996) < 1e-12
    # deep OTM call, K=140
    v = binary_price(0.20, 140.0, 1.0, 100.0, _R, _Q, True)
    assert abs(v - 0.04879969669114354) < 1e-12
    # put, K=70, T=2, vol=30%
    v = binary_price(0.30, 70.0, 2.0, 100.0, _R, _Q, False)
    assert abs(v - 0.1996576342569846) < 1e-12
    # ATM call, vol=50%
    v = binary_price(0.50, 100.0, 1.0, 100.0, _R, _Q, True)
    assert abs(v - 0.4039439177288213) < 1e-12


def test_binary_put_call_sum_and_zero_vol():
    c = binary_price(0.25, 105.0, 1.5, 100.0, _R, _Q, True)
    p = binary_price(0.25, 105.0, 1.5, 100.0, _R, _Q, False)
    assert abs(c + p - math.exp(-_R * 1.5)) < 1e-14
    # forward 100*e^{0.03} > 90 -> call pays DF; < 110 -> put pays DF
    assert abs(binary_price(0.0, 90.0, 1.0, 100.0, _R, _Q, True) - math.exp(-_R)) < 1e-14
    assert abs(binary_price(0.0, 110.0, 1.0, 100.0, _R, _Q, False) - math.exp(-_R)) < 1e-14
    assert abs(binary_price(0.0, 110.0, 1.0, 100.0, _R, _Q, True)) < 1e-14
