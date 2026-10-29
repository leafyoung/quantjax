"""Port of quantburn/src/lmm.rs (originally
torchquantlib/models/interest_rate/lmm.py).

One-step terminal update of forward rates under a correlated Gaussian shock
(Cholesky factor of the correlation matrix). Module RNG seed: 2006. All math
is scalar float64; normals come from quantjax.rng.Xorshift64Star
(bit-identical to the Rust sampler).

`LMM.simulate_tensor` runs the same one-step scenario arithmetic (correlation
via the Cholesky factor, mean-rate scaling / zero-floor branch) as jax.numpy
float64 array ops; the small Cholesky factor stays host-side and the normals
are drawn host-side from Xorshift64Star (identical draw order) and uploaded
via jnp.array.
"""

import math
from typing import List, Sequence

import jax.numpy as jnp

from quantjax.rng import Xorshift64Star


def cholesky(a: Sequence[Sequence[float]]) -> List[List[float]]:
    """Cholesky factor (lower triangular) of a symmetric positive-definite
    matrix, matching `torch.linalg.cholesky`. Raises ValueError on
    non-positive-definiteness. Shared helper (also used by `exotics` and
    `greeks` in the Rust crate)."""
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j]
            for k in range(j):
                s -= l[i][k] * l[j][k]
            if i == j:
                if s < 0.0:
                    raise ValueError("correlation matrix is not positive definite")
                l[i][j] = math.sqrt(s)
            else:
                l[i][j] = s / l[j][j]
    return l


class LMM:
    """Libor Market Model state."""

    def __init__(self, forward_rates: List[float], volatilities: List[float], correlations: List[List[float]]):
        if len(forward_rates) != len(volatilities):
            raise ValueError("forward_rates_init and volatilities_init must have the same length.")
        n = len(forward_rates)
        if len(correlations) != n or (correlations and len(correlations[0]) != n) or (n > 0 and not correlations):
            raise ValueError("correlations_init must be a square matrix matching the number of rates.")
        self.forward_rates = list(forward_rates)
        self.volatilities = list(volatilities)
        self.correlation_matrix = [list(row) for row in correlations]

    def simulate(self, t: float, n: int, seed: int) -> List[List[float]]:
        """Python `LMM.simulate`: N one-step terminal scenarios at horizon `t`.

        - draws N x num_rates iid normals (row-major) from
          `quantjax.rng.Xorshift64Star` with `seed` (2006 in tests);
        - correlates them with the Cholesky factor: noise = z @ L^T;
        - if all forward rates are non-negative, scales the shock by the mean
          forward rate and floors the result at zero; otherwise plain additive.
        """
        num_rates = len(self.forward_rates)
        vols = [max(v, 0.0) for v in self.volatilities]
        chol = cholesky(self.correlation_matrix)
        sqrt_t = math.sqrt(max(t, 0.0))

        all_nonneg = all(r >= 0.0 for r in self.forward_rates)
        base_scale = sum(self.forward_rates) / num_rates if all_nonneg else 0.0

        rng = Xorshift64Star(seed)
        out = []
        for _ in range(n):
            # Draw one row of iid normals, then correlate.
            z = [rng.next_std_normal() for _ in range(num_rates)]
            row = []
            for j in range(num_rates):
                correlated = 0.0
                for k in range(num_rates):
                    correlated += z[k] * chol[j][k]
                scale = (base_scale if all_nonneg else 1.0) * vols[j] * sqrt_t * correlated
                value = self.forward_rates[j] + scale
                row.append(max(value, 0.0) if all_nonneg else value)
            out.append(row)
        return out

    def simulate_tensor(self, t: float, n: int, seed: int) -> "jnp.ndarray":
        """Tensor variant of `simulate`: the correlated one-step scenarios run
        as jax.numpy float64 array ops (`z @ L^T` correlation, broadcast of
        the per-rate volatilities, mean-rate scaling and zero-floor when all
        forward rates are non-negative). The Cholesky factor and the
        all-non-negative branch decision stay host-side; normals are drawn
        host-side from Xorshift64Star in the same row-major `(n, num_rates)`
        order as the scalar loop and uploaded via jnp.array. Returns a
        float64 `(n, num_rates)` array."""
        num_rates = len(self.forward_rates)
        vols = [max(v, 0.0) for v in self.volatilities]
        chol = cholesky(self.correlation_matrix)
        sqrt_t = math.sqrt(max(t, 0.0))

        all_nonneg = all(r >= 0.0 for r in self.forward_rates)
        base_scale = sum(self.forward_rates) / num_rates if all_nonneg else 0.0

        rng = Xorshift64Star(seed)
        z = jnp.asarray(
            [rng.next_std_normal() for _ in range(n * num_rates)], dtype=jnp.float64
        ).reshape(n, num_rates)

        chol_t = jnp.asarray(chol, dtype=jnp.float64).T
        correlated = z @ chol_t
        vols_arr = jnp.asarray(vols, dtype=jnp.float64)
        scale = (base_scale if all_nonneg else 1.0) * vols_arr * sqrt_t * correlated
        fwd_arr = jnp.asarray(self.forward_rates, dtype=jnp.float64)
        values = fwd_arr + scale
        if all_nonneg:
            values = jnp.maximum(values, 0.0)
        return values


# ---------------------------------------------------------------------------
# Embedded parity tests (expected literals from quantburn/src/lmm.rs)
# ---------------------------------------------------------------------------


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def test_cholesky_matches_reference():
    a = [
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ]
    l = cholesky(a)
    expected = [
        [1.0, 0.0, 0.0],
        [0.5, 0.8660254037844386, 0.0],
        [0.2, 0.23094010767585033, 0.9521904571390466],
    ]
    for i in range(3):
        for j in range(3):
            assert _rel(l[i][j], expected[i][j]) < 1e-12, f"l[{i}][{j}]={l[i][j]}"


def test_simulate_matches_reference_nonneg():
    lmm = LMM(
        [0.03, 0.035, 0.04],
        [0.1, 0.15, 0.2],
        [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
    )
    out = lmm.simulate(1.0, 3, 2006)
    expected = [
        [0.039661145010229715, 0.039854038648551325, 0.04403412931163772],
        [0.035034923857970836, 0.03288614717875651, 0.04410208300180206],
        [0.03318410437062085, 0.040648702167959086, 0.04815742006517208],
    ]
    for row, want in zip(out, expected):
        for got, w in zip(row, want):
            assert _rel(got, w) < 1e-12, f"got={got} want={w}"


def test_simulate_negative_branch():
    lmm = LMM(
        [-0.01, 0.02],
        [0.1, 0.2],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    out = lmm.simulate(1.0, 2, 2006)
    # Uncorrelated noise: value = fwd + vol * sqrt(T) * z.
    expected = [
        [0.26603271457799194, -0.08521279876984011],
        [0.0053045047355977665, 0.30770993474119074],
    ]
    for row, want in zip(out, expected):
        for got, w in zip(row, want):
            assert _rel(got, w) < 1e-12, f"got={got} want={w}"


# ---------------------------------------------------------------------------
# Tensor-variant parity tests (same RNG draws, same branch logic as the
# scalar reference; 1e-9 relative tolerance for jnp float64 rounding)
# ---------------------------------------------------------------------------


def test_simulate_tensor_matches_scalar_nonneg():
    lmm = LMM(
        [0.03, 0.035, 0.04],
        [0.1, 0.15, 0.2],
        [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
    )
    scalar = lmm.simulate(1.0, 5, 2006)
    tensor = lmm.simulate_tensor(1.0, 5, 2006)
    assert tensor.shape == (5, 3)
    assert tensor.dtype == jnp.float64
    for i in range(5):
        for j in range(3):
            assert _rel(float(tensor[i, j]), scalar[i][j]) < 1e-9, (i, j, float(tensor[i, j]), scalar[i][j])
    # Zero floor actually applied on the mean-scaled branch.
    assert all(float(v) >= 0.0 for v in tensor.reshape(-1))


def test_simulate_tensor_matches_scalar_negative_branch():
    lmm = LMM(
        [-0.01, 0.02],
        [0.1, 0.2],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    scalar = lmm.simulate(1.0, 4, 2006)
    tensor = lmm.simulate_tensor(1.0, 4, 2006)
    assert tensor.shape == (4, 2)
    assert tensor.dtype == jnp.float64
    for i in range(4):
        for j in range(2):
            assert _rel(float(tensor[i, j]), scalar[i][j]) < 1e-9, (i, j, float(tensor[i, j]), scalar[i][j])
