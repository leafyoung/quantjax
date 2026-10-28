"""Port of quantburn/src/yield_curve_construction.rs (originally
torchquantlib/utils/yield_curve_construction.py).

Simple yield bootstrapping from cash-flow totals and market prices, plus a
Nelson-Siegel yield curve with lambda fixed at 1. All math is scalar float64
(plain Python floats). The 2D `(n_instruments, n_cashflows)` and batched 3D
`(batch, n_instruments, n_cashflows)` forms of the Python source are kept as
separate functions. Errors mirror the Python source's `RuntimeError`s as
`ValueError`s.
"""

import math
from typing import List, Sequence


def bootstrap_yield_curve(cash_flows: Sequence[Sequence[float]], prices: Sequence[float]) -> List[float]:
    """Construct simple yields from cash-flow totals and market prices.

    `cash_flows[i]` holds the cash flows of instrument `i`; instrument `i`
    discounts at exponent `1/(i+1)` (periods are 1-based).

    Raises ValueError for non-positive prices or a shape mismatch
    (mirroring the Rust `YieldCurveError` variants).
    """
    if any(p <= 0.0 for p in prices):
        raise ValueError("all market prices must be positive")
    if len(cash_flows) != len(prices):
        raise ValueError("cash_flows and prices must have the same number of instruments")

    yields = []
    for i, flows in enumerate(cash_flows):
        total = 0.0
        for f in flows:
            total += f
        exponent = 1.0 / (i + 1)
        yields.append((total / prices[i]) ** exponent - 1.0)
    return yields


def bootstrap_yield_curve_batched(
    cash_flows: Sequence[Sequence[Sequence[float]]],
    prices: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Batched (3D) form: one curve per batch element, with matching price rows.

    Raises ValueError on shape mismatches or non-positive prices.
    """
    if len(cash_flows) != len(prices) or any(
        len(cf) != len(p) for cf, p in zip(cash_flows, prices)
    ):
        raise ValueError("batch shapes of cash_flows and prices must match")
    return [bootstrap_yield_curve(cf, p) for cf, p in zip(cash_flows, prices)]


def nelson_siegel_yield_curve(tau: Sequence[float], beta0: float, beta1: float, beta2: float) -> List[float]:
    """Nelson-Siegel yield curve with lambda fixed at 1:
    `y(tau) = beta0 + beta1 * (1 - e^-tau)/tau + beta2 * ((1 - e^-tau)/tau - e^-tau)`.
    """
    out = []
    for t in tau:
        exp_term = math.exp(-t)
        factor1 = (1.0 - exp_term) / t
        factor2 = factor1 - exp_term
        out.append(beta0 + beta1 * factor1 + beta2 * factor2)
    return out


# ---------------------------------------------------------------------------
# Embedded parity tests (expected literals from quantburn/src/yield_curve_construction.rs)
# ---------------------------------------------------------------------------


def test_bootstrap_yield_curve_2d():
    cash_flows = [[50.0, 1050.0], [60.0, 1060.0], [1080.0, 0.0]]
    prices = [980.0, 995.0, 900.0]
    actual = bootstrap_yield_curve(cash_flows, prices)
    expected = [0.12244897959183665, 0.0609562388258611, 0.06265856918261115]
    for a, e in zip(actual, expected):
        assert abs(a - e) <= 1e-8 * max(abs(e), 1.0), f"actual {a} vs expected {e}"


def test_bootstrap_yield_curve_batched():
    cash_flows = [
        [[50.0, 1050.0], [60.0, 1060.0]],
        [[40.0, 1040.0], [70.0, 1070.0]],
    ]
    prices = [[975.0, 990.0], [960.0, 1005.0]]
    actual = bootstrap_yield_curve_batched(cash_flows, prices)
    expected = [
        [0.1282051282051282, 0.06363204695662072],
        [0.125, 0.0650485238752998],
    ]
    for batch, exp_batch in zip(actual, expected):
        for a, e in zip(batch, exp_batch):
            assert abs(a - e) <= 1e-8 * max(abs(e), 1.0), f"actual {a} vs expected {e}"


def test_bootstrap_errors():
    import pytest

    cash_flows = [[50.0, 1050.0]]
    with pytest.raises(ValueError):
        bootstrap_yield_curve(cash_flows, [0.0])
    with pytest.raises(ValueError):
        bootstrap_yield_curve(cash_flows, [1.0, 2.0])


def test_nelson_siegel():
    tau = [0.5, 1.0, 2.0, 5.0]
    actual = nelson_siegel_yield_curve(tau, 0.03, -0.02, 0.01)
    expected = [
        0.016065306597126332,
        0.02,
        0.024323323583816937,
        0.027946096424007316,
    ]
    for a, e in zip(actual, expected):
        assert abs(a - e) <= 1e-8, f"actual {a} vs expected {e}"
