"""Port of quantburn/src/future_pricer.rs (originally
torchquantlib/core/asset_pricing/money/future_pricer.py).

Theoretical futures price via cost of carry:
`F = spot * exp(carry_rate * expiry)` where `carry_rate = rate` unless a
`foreign_rate` is supplied, in which case `carry_rate = rate - foreign_rate`.
"""

import math


def future_pricer(
    spot: float,
    rate: float,
    expiry: float,
    foreign_rate: float | None = None,
) -> float:
    """Theoretical futures price. Optional `foreign_rate` for cost-of-carry
    style use cases (rate minus foreign rate)."""
    carry_rate = rate if foreign_rate is None else rate - foreign_rate
    return spot * math.exp(carry_rate * expiry)


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn's scripts/gen_misc_pricers.py, float64)
# ---------------------------------------------------------------------------

def test_future_pricer_no_foreign():
    actual = future_pricer(100.0, 0.05, 1.0, None)
    expected = 105.12710963760242
    assert abs(actual - expected) <= 1e-8, f"actual {actual} vs expected {expected}"


def test_future_pricer_with_foreign():
    actual = future_pricer(100.0, 0.05, 0.5, 0.02)
    expected = 101.51130646157189
    assert abs(actual - expected) <= 1e-8, f"actual {actual} vs expected {expected}"
