"""Port of quantburn/src/scenario_analysis.rs (originally
market_risk/scenario_analysis.py).

Applies scenario multipliers to a portfolio value (elementwise product).
"""

from typing import List, Sequence


def scenario_analysis(portfolio_value: float, scenario: float) -> float:
    """Single portfolio value times a single scenario multiplier."""
    return portfolio_value * scenario


def scenario_analysis_vec(
    portfolio_values: Sequence[float], scenarios: Sequence[float]
) -> List[float]:
    """A vector of portfolio values times a vector of scenario multipliers
    (elementwise product, as torch broadcasting over equal shapes)."""
    return [v * s for v, s in zip(portfolio_values, scenarios)]


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn/src/scenario_analysis.rs)
# ---------------------------------------------------------------------------


def test_matches_python_reference():
    # Reference: torch float64 ``scenario_analysis`` from the Python source.
    got = scenario_analysis_vec([1e6, 2.5e5, 4e5], [0.1, -0.2, 0.05])
    assert abs(got[0] - 100000.0) < 1e-8
    assert abs(got[1] - (-50000.0)) < 1e-8
    assert abs(got[2] - 20000.0) < 1e-8
    got = scenario_analysis(1e6, -0.15)
    assert abs(got - (-150000.0)) < 1e-6
