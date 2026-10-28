"""Port of quantburn/src/fixed_income_forward.rs (originally
torchquantlib/core/asset_pricing/bond/fixed_income_forward.py).

Forward price of a fixed income security under continuous compounding:
`F = face_value * exp((forward_rate - rate) * time_to_maturity)`.
"""

import math


def fixed_income_forward(
    face_value: float,
    rate: float,
    time_to_maturity: float,
    forward_rate: float,
) -> float:
    """Forward price of a fixed income security (continuous compounding)."""
    return face_value * math.exp((forward_rate - rate) * time_to_maturity)


# ---------------------------------------------------------------------------
# Tests (expected value from quantburn's scripts/gen_misc_pricers.py, float64)
# ---------------------------------------------------------------------------

def test_fixed_income_forward():
    actual = fixed_income_forward(100.0, 0.05, 2.0, 0.03)
    expected = 96.07894391523232
    assert abs(actual - expected) <= 1e-8, f"actual {actual} vs expected {expected}"
