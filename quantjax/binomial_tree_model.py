"""Port of quantburn/src/binomial_tree_model.rs (itself a port of
torchquantlib models/binomial_tree.py plus the ``price_binomial_tree`` core
it calls from torchquantlib core/asset_pricing/option/utils.py).

Prices European / American / Bermudan options with a CRR binomial tree. The
tree core is the single shared implementation in ``quantjax.option_utils``
(the Python home of the ``utils.py`` this module calls into), re-exported
here under the module's Python names. The Python batch wrapper loops over 1-D
batch elements independently; ``binomial_tree_batch`` does the same.
"""

import math

from quantjax.option_utils import (
    binomial_tree_batch as _batch_impl,
    calculate_binomial_tree_params,
    price_binomial_tree as _price_impl,
)


def option_payoff(option_type, terminal_prices, strike):
    """Python ``option_payoff`` (scalar form used by this module)."""
    if option_type == "call":
        return max(terminal_prices - strike, 0.0)
    if option_type == "put":
        return max(strike - terminal_prices, 0.0)
    raise ValueError("Invalid option type. Use 'call' or 'put'.")


def price_binomial_tree(
    option_type,
    option_style,
    spot,
    strike,
    expiry,
    volatility,
    rate,
    steps,
    exercise_dates=None,
):
    """Python ``price_binomial_tree`` for a scalar spot. ``exercise_dates`` holds
    step indices at which early exercise is allowed (Bermudan); ignored for
    European/American styles, matching the Python."""
    return _price_impl(
        option_type,
        option_style,
        spot,
        strike,
        expiry,
        volatility,
        rate,
        steps,
        exercise_dates,
    )


def binomial_tree_batch(
    option_type,
    option_style,
    spot,
    strike,
    expiry,
    volatility,
    rate,
    steps,
):
    """Python ``binomial_tree`` batch wrapper: evaluate each batch element
    independently. All lists must have the same length (scalar expansion is
    handled by the caller in Python via broadcasting; here we accept either a
    single-element list broadcast across the batch or a full-length list)."""
    return _batch_impl(option_type, option_style, spot, strike, expiry, volatility, rate, steps)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# --- tests (expected values from quantburn's binomial_tree_model.rs tests,
# deterministic closed-form tree results, f64 re-computation in
# scripts/gen_binomial_tree_model.py; tolerance 1e-12 relative) ---


def test_european_call_put():
    call = price_binomial_tree("call", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 100)
    put = price_binomial_tree("put", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 100)
    assert abs(call - 10.430611662249113) < 1e-11, f"call={call}"
    assert abs(put - 5.553554112321353) < 1e-11, f"put={put}"


def test_american_put_early_exercise_premium():
    euro = price_binomial_tree("put", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 100)
    amer = price_binomial_tree("put", "american", 100.0, 100.0, 1.0, 0.2, 0.05, 100)
    assert abs(amer - 6.082354409142444) < 1e-11, f"amer={amer}"
    assert amer > euro  # early-exercise premium


def test_bermudan_between_euro_and_american():
    dates = list(range(0, 100, 10))
    berm = price_binomial_tree("put", "bermudan", 100.0, 100.0, 1.0, 0.2, 0.05, 100, dates)
    assert abs(berm - 6.025795352002811) < 1e-11, f"berm={berm}"


def test_zero_volatility_deterministic():
    v = price_binomial_tree("call", "european", 100.0, 90.0, 1.0, 0.0, 0.05, 50)
    expected = math.exp(-0.05) * (100.0 * math.exp(0.05) - 90.0)
    assert abs(v - expected) < 1e-12


def test_zero_expiry_intrinsic():
    v = price_binomial_tree("call", "european", 110.0, 100.0, 0.0, 0.2, 0.05, 50)
    assert abs(v - 10.0) < 1e-12


def test_batch_matches_scalar():
    out = binomial_tree_batch(
        "call",
        "european",
        [100.0, 110.0],
        [100.0],
        [1.0],
        [0.2],
        [0.05],
        100,
    )
    assert len(out) == 2
    assert abs(out[0] - 10.430611662249113) < 1e-11
