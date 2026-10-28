"""Port of quantburn/src/stress_testing.rs (originally
market_risk/stress_testing.py).

Applies percentage stress scenarios: portfolio * (1 + scenario).
"""

from typing import List, Sequence


def stress_test(portfolio_value: float, stress_scenario: float) -> float:
    """Single portfolio value under a percentage stress scenario."""
    return portfolio_value * (1.0 + stress_scenario)


def stress_test_vec(
    portfolio_values: Sequence[float], stress_scenarios: Sequence[float]
) -> List[float]:
    """A vector of portfolio values under a vector of percentage scenarios."""
    return [v * (1.0 + s) for v, s in zip(portfolio_values, stress_scenarios)]


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn/src/stress_testing.rs)
# ---------------------------------------------------------------------------


def test_matches_python_reference():
    # Reference: torch float64 ``stress_test`` from the Python source.
    got = stress_test(1e6, -0.35)
    assert abs(got - 650000.0) < 1e-6
    got = stress_test_vec([1e6, 2e6], [0.2, -0.3])
    assert abs(got[0] - 1200000.0) < 1e-6
    assert abs(got[1] - 1400000.0) < 1e-6
