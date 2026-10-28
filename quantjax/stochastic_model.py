"""Port of quantburn/src/stochastic_model.rs (originally
torchquantlib/models/stochastic_model.py).

The Python file is a PyTorch base class (`StochasticModel`) providing
parameter storage/movement, device handling, gradient "anchor" hooks and
parameter constraint clamping — training-harness machinery with no reusable
numeric tensor core (rule 7). The numeric core that survives the port is the
parameter-constraint machinery: per-parameter min/max bounds
(`parameter_constraints`), `constrained_param` (clamped read) and
`apply_constraints` (in-place clamp with a nextafter-adjusted lower bound so
a strict `min` bound stays attainable).
"""

import struct
from typing import Dict, Optional


class ParamBounds:
    """Constraint bounds for a single parameter, mirroring the Python
    `{"min": ..., "max": ...}` dicts."""

    __slots__ = ("min", "max")

    def __init__(self, min: Optional[float] = None, max: Optional[float] = None):
        self.min = min
        self.max = max


def _safe_lower_bound(min: float) -> float:
    """`torch.nextafter(min, inf)` in the Python `_safe_lower_bound`: the
    smallest f64 strictly greater than `min`. Implemented with the same bit
    trick as the Rust port (bits + 1 for non-negative, bits - 1 for
    negative), so -0.0 maps to the smallest negative subnormal exactly like
    the Rust code."""
    bits = struct.unpack(">Q", struct.pack(">d", min))[0]
    if min >= 0.0:
        bits += 1
    else:
        bits -= 1
    return struct.unpack(">d", struct.pack(">Q", bits & ((1 << 64) - 1)))[0]


class StochasticModel:
    """Shared base for stochastic simulation models: a named parameter store
    plus per-parameter constraint bounds."""

    def __init__(self):
        self._params: Dict[str, float] = {}
        self._constraints: Dict[str, ParamBounds] = {}

    def set_param(self, name: str, value: float) -> None:
        self._params[name] = value

    def param(self, name: str):
        return self._params.get(name)

    def set_constraints(self, name: str, bounds: ParamBounds) -> None:
        self._constraints[name] = bounds

    def constrained_param(self, name: str) -> float:
        """Python `constrained_param`: clamp the parameter into its declared
        bounds (no-op when unconstrained)."""
        value = self._params.get(name, 0.0)
        bounds = self._constraints.get(name)
        if bounds is None:
            return value
        v = value
        if bounds.min is not None:
            lo = _safe_lower_bound(bounds.min)
            if v < lo:
                v = lo
        if bounds.max is not None:
            if v > bounds.max:
                v = bounds.max
        return v

    def apply_constraints(self) -> None:
        """Python `_apply_constraints`: clamp the stored parameter in place
        (lower bound via `_safe_lower_bound`, upper bound as declared)."""
        for name, bounds in self._constraints.items():
            if bounds.min is None:
                continue
            lower = _safe_lower_bound(bounds.min)
            if name in self._params:
                v = self._params[name]
                if v < lower:
                    v = lower
                if bounds.max is not None and v > bounds.max:
                    v = bounds.max
                self._params[name] = v


# ---------------------------------------------------------------------------
# Embedded parity tests (expected literals from quantburn/src/stochastic_model.rs)
# ---------------------------------------------------------------------------


def test_constrained_param_clamps_to_bounds():
    m = StochasticModel()
    m.set_param("kappa", -0.5)
    m.set_param("rho", 1.5)
    m.set_param("theta", 0.04)
    m.set_constraints("kappa", ParamBounds(min=1e-6))
    m.set_constraints("rho", ParamBounds(min=-0.9989, max=0.9989))

    # Unconstrained passes through.
    assert abs(m.constrained_param("theta") - 0.04) < 1e-12
    # Below min -> clamped to nextafter(min, inf), strictly above min.
    k = m.constrained_param("kappa")
    assert k > 1e-6 and k - 1e-6 < 1e-18
    # Above max -> clamped.
    assert abs(m.constrained_param("rho") - 0.9989) < 1e-12


def test_apply_constraints_clamps_in_place():
    m = StochasticModel()
    m.set_param("sigma", -1.0)
    m.set_param("rho", -2.0)
    m.set_constraints("sigma", ParamBounds(min=1e-6))
    m.set_constraints("rho", ParamBounds(min=-0.9989, max=0.9989))
    m.apply_constraints()
    # Compare exactly against math.nextafter, which equals the Rust next_up.
    import math

    expected_lower = math.nextafter(1e-6, math.inf)
    assert abs(m.param("sigma") - expected_lower) < 1e-24
    assert abs(m.param("rho") - (-0.9989)) < 1e-12


def test_safe_lower_bound_is_nextafter():
    import math

    assert _safe_lower_bound(1e-6) == math.nextafter(1e-6, math.inf)
    assert _safe_lower_bound(0.0) == 5e-324
