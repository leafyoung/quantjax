"""Port of quantburn/src/expected_shortfall.rs (originally
market_risk/expected_shortfall.py).

Expected Shortfall / CVaR: mean of losses at or beyond historical VaR (plus a
tiny 1e-6 * VaR stabilizer exactly as in the Python source); zero when no
loss reaches VaR.
"""

from typing import List, Sequence

from quantjax.var import calculate_var, validate_confidence_level, validate_returns


def calculate_es(returns: Sequence[float], confidence_level: float) -> float:
    """Expected Shortfall for a 1-D return series."""
    validate_confidence_level(confidence_level)
    validate_returns(returns)
    losses = [max(-r, 0.0) for r in returns]
    var = calculate_var(returns, confidence_level)
    tail = [l for l in losses if l >= var]
    if not tail:
        return 0.0
    mean = sum(tail) / len(tail)
    return mean + 1e-6 * var


def calculate_es_2d(
    returns: Sequence[Sequence[float]], confidence_level: float
) -> List[float]:
    """Expected Shortfall along the last dimension of a 2-D return matrix."""
    validate_confidence_level(confidence_level)
    return [calculate_es(row, confidence_level) for row in returns]


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn/src/expected_shortfall.rs)
# ---------------------------------------------------------------------------


def test_matches_python_reference():
    # Reference: torch float64 ``calculate_es`` from the Python source.
    r = [
        0.01, -0.02, 0.015, -0.005, 0.03, -0.012, 0.008, -0.025, 0.001, -0.018,
    ]
    cases = [
        (0.90, 0.0250000205),
        (0.95, 0.02500002275),
        (0.99, 0.025000024550000003),
    ]
    for q, expected in cases:
        got = calculate_es(r, q)
        assert abs(got - expected) < 1e-14, f"q={q}: {got} vs {expected}"
    r2 = [
        [0.01, -0.05, 0.02, -0.03],
        [0.04, -0.01, -0.02, 0.005],
    ]
    got = calculate_es_2d(r2, 0.75)
    assert abs(got[0] - 0.050000035000000005) < 1e-14
    assert abs(got[1] - 0.0200000125) < 1e-14


def test_positive_returns_zero_es():
    # No losses at/above VaR -> zero (Python returns a zero scalar).
    got = calculate_es([0.01, 0.02, 0.005], 0.95)
    assert got == 0.0
