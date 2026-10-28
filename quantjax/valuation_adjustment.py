"""Port of quantburn/src/valuation_adjustment.rs (originally
valuation_adjustment/valuation_adjustment.py).

X-value adjustments (CVA/DVA/FVA/MVA), all closed-form products of their
inputs, with the same non-negative / bounded validation as the Python.
"""

from typing import NoReturn


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_recovery_rate(recovery_rate: float) -> None:
    _validate_non_negative("recovery_rate", recovery_rate)
    if recovery_rate > 1.0:
        raise ValueError("recovery_rate must be between 0 and 1.")


def calculate_cva(exposure: float, default_prob: float, recovery_rate: float) -> float:
    """Credit valuation adjustment."""
    _validate_non_negative("exposure", exposure)
    _validate_non_negative("default_prob", default_prob)
    _validate_recovery_rate(recovery_rate)
    return exposure * (1.0 - recovery_rate) * default_prob


def calculate_dva(exposure: float, default_prob: float, recovery_rate: float) -> float:
    """Debit valuation adjustment (same formula as CVA in the Python source)."""
    return calculate_cva(exposure, default_prob, recovery_rate)


def calculate_fva(exposure: float, funding_spread: float, maturity: float) -> float:
    """Funding valuation adjustment."""
    _validate_non_negative("exposure", exposure)
    _validate_non_negative("funding_spread", funding_spread)
    _validate_non_negative("maturity", maturity)
    return exposure * funding_spread * maturity


def calculate_mva(exposure: float, funding_cost: float, maturity: float) -> float:
    """Margin valuation adjustment."""
    _validate_non_negative("exposure", exposure)
    _validate_non_negative("funding_cost", funding_cost)
    _validate_non_negative("maturity", maturity)
    return exposure * funding_cost * maturity


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn/src/valuation_adjustment.rs)
# ---------------------------------------------------------------------------


def test_matches_python_reference():
    # Reference: torch float64 from the Python source.
    got = calculate_cva(1e6, 0.03, 0.4)
    assert abs(got - 18000.0) < 1e-6
    got = calculate_dva(1e6, 0.03, 0.4)
    assert abs(got - 18000.0) < 1e-6
    got = calculate_fva(5e5, 0.012, 3.0)
    assert abs(got - 18000.0) < 1e-6
    got = calculate_mva(2e5, 0.008, 2.5)
    assert abs(got - 4000.0) < 1e-6


def test_validation():
    # Validation errors mirror the Python checks.
    import pytest

    with pytest.raises(ValueError):
        calculate_cva(-1.0, 0.03, 0.4)
    with pytest.raises(ValueError):
        calculate_cva(1e6, -0.03, 0.4)
    with pytest.raises(ValueError):
        calculate_cva(1e6, 0.03, 1.5)
    with pytest.raises(ValueError):
        calculate_fva(1.0, -0.1, 1.0)
    with pytest.raises(ValueError):
        calculate_mva(1.0, 0.1, -1.0)
