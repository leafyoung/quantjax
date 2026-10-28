"""Port of quantburn/src/reduced_form_model.rs (originally
credit_risk/reduced_form_model.py).

Constant-hazard-rate reduced-form expected loss:
survival = exp(-lambda * t);
expected_loss = (1 - recovery) * (1 - survival).
``lambda_0`` is unused in the Python implementation (kept for parity).
"""

import math


def reduced_form_model(
    lambda_0: float, default_intensity: float, recovery_rate: float, time: float
) -> float:
    """Expected loss over ``time`` under a constant default intensity."""
    survival_prob = math.exp(-default_intensity * time)
    return (1.0 - recovery_rate) * (1.0 - survival_prob)


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn/src/reduced_form_model.rs)
# ---------------------------------------------------------------------------


def test_matches_python_reference():
    # Reference: torch float64 from the Python source.
    cases = [
        (0.03, 0.02, 0.4, 5.0, 0.05709754917842429),
        (0.03, 0.05, 0.25, 10.0, 0.29510200521552493),
        (0.03, 0.10, 1.0, 7.0, 0.0),
    ]
    for l0, lam, rec, t, expected in cases:
        got = reduced_form_model(l0, lam, rec, t)
        assert abs(got - expected) <= 1e-12 * max(abs(expected), 1e-300), (
            f"got {got}, expected {expected}"
        )
