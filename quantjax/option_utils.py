"""Port of quantburn/src/option_utils.rs (itself a port of torchquantlib
core/asset_pricing/option/utils.py).

Binomial-tree building blocks shared by the American/Bermudan option pricers.
All math in plain Python floats (f64), mirroring the Python numerics (formulas
and tree layout identical: ``price_tree[step][downs]``, backward induction over
steps).
"""

import math


def calculate_binomial_tree_params(expiry, volatility, rate, steps):
    """(dt, u, d, p) parameters of the CRR binomial tree."""
    dt = expiry / steps
    if dt <= 0.0:
        return (dt, 1.0, 1.0, 1.0)
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u
    if abs(u - d) < 1e-12 * max(abs(u), 1.0):
        p = 1.0
    else:
        p = (math.exp(rate * dt) - d) / (u - d)
    return (dt, u, d, p)


def option_payoff(option_type, terminal_prices, strike):
    """Payoff of a call/put at terminal (or any) prices (list form)."""
    if option_type == "call":
        return [max(s - strike, 0.0) for s in terminal_prices]
    if option_type == "put":
        return [max(strike - s, 0.0) for s in terminal_prices]
    raise ValueError("Invalid option type. Use 'call' or 'put'.")


def build_price_tree(spot, u, d, steps):
    """Price tree ``tree[step][down]`` = spot * u^(step-down) * d^down."""
    tree = [[0.0] * (steps + 1) for _ in range(steps + 1)]
    for step in range(steps + 1):
        for down in range(step + 1):
            ups = float(step - down)
            tree[step][down] = spot * u ** ups * d ** down
    return tree


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
    lattice="crr",
):
    """Binomial-tree option price. ``exercise_dates`` are step indices at which
    early exercise is allowed (Bermudan); ignored for European style, and the
    full range 0..steps is used for American style.

    ``lattice`` selects the node convention:
      - ``"crr"`` (default): classic multiplicative CRR tree
        (u = e^(sigma*sqrt(dt)), d = 1/u, p = (e^(r*dt) - d)/(u - d)).
      - ``"drift_centered"``: QuantLib's equal-jumps ``CoxRossRubinstein``
        variant (dx = sigma*sqrt(dt), pu = 0.5 + 0.5*(r - sigma^2/2)*dt/dx,
        nodes S*e^((step - 2*down)*dx)). Both lattices share the discount
        e^(-r*dt), the child indexing (``down`` counts down moves) and the
        exercise-date semantics; only node placement and probabilities differ.
        Degenerate inputs follow the same conventions as ``"crr"``: zero
        volatility returns the deterministic discounted payoff; dx <= 0 or
        pu outside [0, 1] raises (pu is reachable outside [0, 1] for extreme
        rate/volatility combinations)."""
    if option_type != "call" and option_type != "put":
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    if option_style not in ("european", "american", "bermudan"):
        raise ValueError("Invalid option style. Use 'european', 'american', or 'bermudan'.")
    if lattice not in ("crr", "drift_centered"):
        raise ValueError("Invalid lattice. Use 'crr' or 'drift_centered'.")
    if steps == 0:
        raise ValueError("steps must be a positive integer.")
    if expiry <= 0.0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative.")
    if abs(volatility) < 1e-12:
        deterministic_terminal = spot * math.exp(rate * expiry)
        disc = math.exp(-rate * expiry)
        if option_type == "call":
            return disc * max(deterministic_terminal - strike, 0.0)
        return disc * max(strike - deterministic_terminal, 0.0)

    if lattice == "drift_centered":
        dt = expiry / steps
        dx = volatility * math.sqrt(dt)
        if dx <= 0.0:  # defensive: unreachable after the zero-vol return above
            raise ValueError("drift-centered lattice requires dx = volatility*sqrt(dt) > 0.")
        pu = 0.5 + 0.5 * (rate - 0.5 * volatility * volatility) * dt / dx
        if not (0.0 <= pu <= 1.0):
            raise ValueError(f"drift-centered lattice requires pu in [0, 1], got {pu}.")
        p = pu
        discount = math.exp(-rate * dt)
        price_tree = [
            [spot * math.exp((step - 2 * down) * dx) for down in range(step + 1)]
            for step in range(steps + 1)
        ]
    else:
        dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)
        discount = math.exp(-rate * dt)
        price_tree = build_price_tree(spot, u, d, steps)
    value_tree = [[0.0] * (steps + 1) for _ in range(steps + 1)]
    for down in range(steps + 1):
        if option_type == "call":
            value_tree[steps][down] = max(price_tree[steps][down] - strike, 0.0)
        else:
            value_tree[steps][down] = max(strike - price_tree[steps][down], 0.0)

    if option_style == "american":
        exercise_steps = list(range(steps))
    elif option_style == "bermudan":
        exercise_steps = [s for s in (exercise_dates or []) if s < steps]
    else:
        exercise_steps = []

    for step in range(steps - 1, -1, -1):
        can_exercise = step in exercise_steps
        for down in range(step + 1):
            continuation = (
                p * value_tree[step + 1][down] + (1.0 - p) * value_tree[step + 1][down + 1]
            ) * discount
            if can_exercise:
                if option_type == "call":
                    exercise = max(price_tree[step][down] - strike, 0.0)
                else:
                    exercise = max(strike - price_tree[step][down], 0.0)
                value_tree[step][down] = max(exercise, continuation)
            else:
                value_tree[step][down] = continuation

    return value_tree[0][0]


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
    """Batch wrapper: evaluate each batch element independently. A single-element
    list is broadcast across the batch (Python broadcasting semantics);
    otherwise lists must be full-length."""
    n = len(spot)

    def pick(v, i):
        return v[0] if len(v) == 1 else v[i]

    return [
        price_binomial_tree(
            option_type,
            option_style,
            spot[i],
            pick(strike, i),
            pick(expiry, i),
            pick(volatility, i),
            pick(rate, i),
            steps,
            None,
        )
        for i in range(n)
    ]


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# --- tests (expected values from quantburn's option_utils.rs tests,
# generated by scripts/gen_option_utils.py, pure f64) ---


def test_european_call_tree():
    v = price_binomial_tree("call", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 5)
    assert abs(v - 10.805933920410084) < 1e-9


def test_american_put_tree():
    v = price_binomial_tree("put", "american", 100.0, 100.0, 1.0, 0.2, 0.05, 10)
    assert abs(v - 6.0042590201829515) < 1e-9


def test_bermudan_put_tree():
    v = price_binomial_tree("put", "bermudan", 100.0, 100.0, 1.0, 0.2, 0.05, 10, [2, 5, 8])
    assert abs(v - 5.856695576439613) < 1e-9


def test_zero_volatility():
    v = price_binomial_tree("call", "european", 100.0, 90.0, 1.0, 0.0, 0.05, 5)
    assert abs(v - 14.38935179493575) < 1e-9


def test_zero_expiry_returns_intrinsic():
    v = price_binomial_tree("put", "european", 95.0, 100.0, 0.0, 0.2, 0.05, 5)
    assert abs(v - 5.0) < 1e-12


def test_params_match_python():
    dt, u, d, p = calculate_binomial_tree_params(1.0, 0.2, 0.05, 4)
    assert abs(dt - 0.25) < 1e-12
    assert abs(u - 1.1051709180756477) < 1e-12
    assert abs(d - 0.9048374180359595) < 1e-12
    assert abs(p - 0.537808371956414) < 1e-12


def test_invalid_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        price_binomial_tree("straddle", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 5)
    with pytest.raises(ValueError):
        price_binomial_tree("call", "asian", 100.0, 100.0, 1.0, 0.2, 0.05, 5)
    with pytest.raises(ValueError):
        price_binomial_tree("call", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 0)
    with pytest.raises(ValueError):
        price_binomial_tree("call", "european", 100.0, 100.0, 1.0, -0.2, 0.05, 5)
    with pytest.raises(ValueError):
        price_binomial_tree("call", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 5, lattice="trinomial")


# --- drift-centered lattice tests (QuantLib `CoxRossRubinstein` convention).
# Reference literals verified against QuantLib's
# BinomialVanillaEngine(process, "crr", 100 steps) for S=K=100, r=5%, q=0,
# vol=20%, T=1 (see ../cross-check README, "Remaining documented difference":
# classic multiplicative CRR vs drift-centered equal-jumps). ---


def test_drift_centered_european_call_matches_quantlib():
    v = price_binomial_tree(
        "call", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 100, lattice="drift_centered"
    )
    assert abs(v - 10.429985954340486) < 1e-10


def test_drift_centered_american_put_matches_quantlib():
    v = price_binomial_tree(
        "put", "american", 100.0, 100.0, 1.0, 0.2, 0.05, 100, lattice="drift_centered"
    )
    assert abs(v - 6.0826182179030805) < 1e-10


def test_drift_centered_zero_volatility_is_deterministic():
    # Same degenerate convention as the classic tree: sigma = 0 collapses to
    # the deterministic discounted payoff (dx <= 0 never reached).
    v_dc = price_binomial_tree(
        "call", "european", 100.0, 90.0, 1.0, 0.0, 0.05, 5, lattice="drift_centered"
    )
    v_default = price_binomial_tree("call", "european", 100.0, 90.0, 1.0, 0.0, 0.05, 5)
    assert abs(v_dc - 14.38935179493575) < 1e-12
    assert v_dc == v_default


def test_drift_centered_rejects_probability_out_of_range():
    import pytest

    # r = 100% with vol = 10% pushes pu far above 1.
    with pytest.raises(ValueError):
        price_binomial_tree(
            "call", "european", 100.0, 100.0, 1.0, 0.1, 1.0, 4, lattice="drift_centered"
        )
