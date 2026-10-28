"""Port of quantburn/src/var.rs (originally market_risk/var.py).

Historical-simulation VaR: losses = max(-returns, 0), then a linearly
interpolated (torch.quantile default) quantile over the last dimension.
"""

from typing import List, Sequence


def validate_confidence_level(confidence_level: float) -> None:
    """Validates the confidence level is strictly between 0 and 1."""
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("Confidence level must be between 0 and 1")


def validate_returns(returns: Sequence[float]) -> None:
    """Validates a non-empty return series."""
    if len(returns) == 0:
        raise ValueError("returns must not be empty.")


def _quantile_linear(xs: List[float], q: float) -> float:
    """torch.quantile default ("linear") interpolation on ascending input:
    position q*(n-1) with linear interpolation between neighbors."""
    xs = sorted(xs)
    n = len(xs)
    pos = q * (n - 1)
    lo = int(pos // 1)
    hi = int(-(-pos // 1))  # ceil
    frac = pos - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def calculate_var(returns: Sequence[float], confidence_level: float) -> float:
    """Historical VaR for a 1-D return series."""
    validate_confidence_level(confidence_level)
    validate_returns(returns)
    losses = [max(-r, 0.0) for r in returns]
    return _quantile_linear(losses, confidence_level)


def calculate_var_2d(
    returns: Sequence[Sequence[float]], confidence_level: float
) -> List[float]:
    """Historical VaR along the last dimension of a 2-D return matrix."""
    validate_confidence_level(confidence_level)
    return [calculate_var(row, confidence_level) for row in returns]


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn/src/var.rs)
# ---------------------------------------------------------------------------


def test_matches_python_reference():
    # Reference: torch float64 ``calculate_var`` from the Python source.
    r = [
        0.01, -0.02, 0.015, -0.005, 0.03, -0.012, 0.008, -0.025, 0.001, -0.018,
    ]
    cases = [
        (0.90, 0.020499999999999997),
        (0.95, 0.022749999999999996),
        (0.99, 0.024550000000000002),
    ]
    for q, expected in cases:
        got = calculate_var(r, q)
        assert abs(got - expected) < 1e-12, f"q={q}: {got} vs {expected}"
    r2 = [
        [0.01, -0.05, 0.02, -0.03],
        [0.04, -0.01, -0.02, 0.005],
    ]
    got = calculate_var_2d(r2, 0.75)
    assert abs(got[0] - 0.035) < 1e-12
    assert abs(got[1] - 0.0125) < 1e-12


def test_positive_returns_zero_var():
    # Positive-return-only samples map to zero VaR.
    got = calculate_var([0.01, 0.02, 0.005], 0.95)
    assert abs(got) < 1e-15


def test_validation():
    # Validation errors mirror the Python checks.
    import pytest

    with pytest.raises(ValueError):
        calculate_var([0.1], 1.0)
    with pytest.raises(ValueError):
        calculate_var([0.1], 0.0)
    with pytest.raises(ValueError):
        calculate_var([], 0.95)
