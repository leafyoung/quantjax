"""Port of quantburn/src/malliavin.rs (which mirrors greeks/malliavin.py, a
thin re-export of `MalliavinGreeks` from greeks.py). Monte-Carlo option
greeks via Malliavin-calculus weights.

Stochasticity: all standard normals come from `quantjax.rng.Xorshift64Star`
(Box-Muller, bit-exact against quantburn's `scripts/refgen.py`). Draw order
replicates the row-major fill of the original torch tensors: for an
(num_paths, num_steps) draw, path-major then step-minor. Tests use seed 3001
(module index 1 of the 3000-block).

Math is done in Python floats. Expected values come from the float64
reference generator (quantburn's scripts/gen_greeks.py, same normals, same
seed); tolerance 1e-6 relative.

The Cholesky factorization needed by `basket_option_greeks` is implemented
locally as `_cholesky`, copying quantburn/src/lmm.rs's `cholesky` algorithm
bit-for-bit (this package does not ship an lmm module).
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from quantjax.rng import Xorshift64Star


class MalliavinGreeks:
    """Mirror of the Python `MalliavinGreeks` class (a name-space handle; all
    numeric work is in the free functions below)."""


# Alias mirroring the Python `Greeks = MalliavinGreeks` re-export.
Greeks = MalliavinGreeks


@dataclass
class EuropeanGreeks:
    """Greeks bundle for European options (Delta, Gamma, Vega, Theta, Rho)."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class Greeks3:
    """Greeks bundle for digital / barrier / basket / Asian options."""
    delta: float
    gamma: float
    vega: float


def _cholesky(a: Sequence[Sequence[float]]) -> List[List[float]]:
    """Cholesky factorization of a symmetric positive-definite matrix,
    matching torch.linalg.cholesky. Copied from quantburn/src/lmm.rs
    (`crate::lmm::cholesky`); raises ValueError where the Rust version
    asserts positive-definiteness."""
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


def european_option_greeks(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    num_paths: int,
    seed: int,
) -> EuropeanGreeks:
    """Malliavin greeks for a European call (single-step GBM, exact terminal
    simulation)."""
    rng = Xorshift64Star(seed)
    sqrt_dt = math.sqrt(t)
    drift = (r - 0.5 * sigma * sigma) * t
    discount = math.exp(-r * t)

    acc = [0.0] * 6  # payoff*Dt, payoff*D2, payoff*Dsigma, payoff*S, payoff*Dr, payoff
    for _ in range(num_paths):
        z = rng.next_std_normal()
        st = s0 * math.exp(drift + sigma * sqrt_dt * z)
        payoff = max(st - k, 0.0)
        dt_st = st * z / sqrt_dt
        d2t_st = dt_st * (z / sqrt_dt - sigma * sqrt_dt)
        d_sigma_st = st * (z * sqrt_dt - sigma * t)
        d_r_st = st * t
        acc[0] += payoff * dt_st
        acc[1] += payoff * d2t_st
        acc[2] += payoff * d_sigma_st
        acc[3] += payoff * st
        acc[4] += payoff * d_r_st
        acc[5] += payoff
    n = float(num_paths)
    return EuropeanGreeks(
        delta=discount * (acc[0] / n) / (sigma * s0),
        gamma=discount * (acc[1] / n) / (sigma * sigma * s0 * s0),
        vega=discount * (acc[2] / n) / sigma,
        theta=-discount * ((r - 0.5 * sigma * sigma) * acc[3] / n + 0.5 * sigma * acc[0] / n) / t,
        rho=discount * (acc[4] / n) - t * discount * (acc[5] / n),
    )


def digital_option_greeks(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
    num_paths: int,
    epsilon: float,
    seed: int,
) -> Greeks3:
    """Malliavin greeks for a digital (cash-or-nothing) call, with the Dirac
    delta approximated by a narrow Gaussian of width `epsilon`."""
    rng = Xorshift64Star(seed)
    sqrt_dt = math.sqrt(t)
    drift = (r - 0.5 * sigma * sigma) * t
    discount = math.exp(-r * t)
    normalizer = epsilon * math.sqrt(2.0 * math.pi)

    acc = [0.0] * 3
    for _ in range(num_paths):
        z = rng.next_std_normal()
        st = s0 * math.exp(drift + sigma * sqrt_dt * z)
        delta_approx = (1.0 / normalizer) * math.exp(-0.5 * ((st - k) / epsilon) ** 2)
        dt_st = st * z / sqrt_dt
        d2t_st = dt_st * (z / sqrt_dt - sigma * sqrt_dt)
        d_sigma_st = st * (z * sqrt_dt - sigma * t)
        w = q * delta_approx
        acc[0] += w * dt_st
        acc[1] += w * d2t_st
        acc[2] += w * d_sigma_st
    n = float(num_paths)
    return Greeks3(
        delta=discount * (acc[0] / n) / (sigma * s0),
        gamma=discount * (acc[1] / n) / (sigma * sigma * s0 * s0),
        vega=discount * (acc[2] / n) / sigma,
    )


def _simulate_paths(
    s0: float,
    t: float,
    r: float,
    sigma: float,
    num_steps: int,
    z: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Simulate GBM paths row-major from an already-drawn normal matrix
    `z[path][step]`; returns S[0..=steps] per path."""
    dt = t / num_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    paths = []
    for row in z:
        path = [s0]
        for zi in row:
            path.append(path[-1] * math.exp(drift + vol * zi))
        paths.append(path)
    return paths


def barrier_option_greeks(
    s0: float,
    k: float,
    h: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str,
    num_paths: int,
    num_steps: int,
    seed: int,
) -> Greeks3:
    """Malliavin greeks for a knock-out barrier call (`up-and-out` or
    `down-and-out`), monitored at each simulation step."""
    if option_type not in ("up-and-out", "down-and-out"):
        raise ValueError("Invalid option_type. Use 'up-and-out' or 'down-and-out'.")
    up = option_type == "up-and-out"
    rng = Xorshift64Star(seed)
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)

    # Draw the (num_paths, num_steps) normal matrix in row-major order.
    z = [[rng.next_std_normal() for _ in range(num_steps)] for _ in range(num_paths)]
    paths = _simulate_paths(s0, t, r, sigma, num_steps, z)

    discount = math.exp(-r * t)
    acc = [0.0] * 3
    for zi, srow in zip(z, paths):
        st = srow[-1]
        hit_barrier = any((s >= h if up else s <= h) for s in srow[1:])
        if hit_barrier:
            continue  # payoff and all Malliavin weights are zeroed by (~hit_barrier)
        payoff = max(st - k, 0.0)
        z_sum = sum(zi)
        dt_st = st * z_sum / sqrt_dt
        d2t_st = dt_st * (z_sum / sqrt_dt - sigma * sqrt_dt)
        z2m1 = sum(v * v - 1.0 for v in zi)
        d_sigma_st = st * (z2m1 * sqrt_dt)
        acc[0] += payoff * dt_st
        acc[1] += payoff * d2t_st
        acc[2] += payoff * d_sigma_st
    n = float(num_paths)
    return Greeks3(
        delta=discount * (acc[0] / n) / (sigma * s0),
        gamma=discount * (acc[1] / n) / (sigma * sigma * s0 * s0),
        vega=discount * (acc[2] / n) / sigma,
    )


def lookback_option_delta(
    s0: float,
    t: float,
    r: float,
    sigma: float,
    num_paths: int,
    num_steps: int,
    seed: int,
) -> float:
    """Malliavin delta for a floating-strike lookback call."""
    rng = Xorshift64Star(seed)
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)
    z = [[rng.next_std_normal() for _ in range(num_steps)] for _ in range(num_paths)]
    paths = _simulate_paths(s0, t, r, sigma, num_steps, z)

    discount = math.exp(-r * t)
    acc = 0.0
    for zi, srow in zip(z, paths):
        st = srow[-1]
        # S_min over columns [0, num_steps) (final column excluded, as in
        # the Python `S[:, :-1]`).
        prefix = srow[:num_steps]
        s_min = min(prefix)
        payoff = max(st - s_min, 0.0)
        dt_st = st * sum(zi) / sqrt_dt
        # indicator: columns equal to the minimum; average their Malliavin
        # derivatives (0 when none — mirrors the Python NaN guard).
        dt_smin = 0.0
        count = 0.0
        for j, s in enumerate(prefix):
            if s == s_min:
                dt_smin += s * zi[j] / sqrt_dt
                count += 1.0
        dt_smin = dt_smin / count if count > 0.0 else 0.0
        if payoff > 0.0:
            acc += (dt_st - dt_smin) / (sigma * s0)
    return discount * acc / num_paths


def basket_option_greeks(
    s0: Sequence[float],
    k: float,
    t: float,
    r: float,
    sigma: Sequence[float],
    rho: Sequence[Sequence[float]],
    weights: Sequence[float],
    num_paths: int,
    seed: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Malliavin greeks for a European basket (arithmetic-weighted) call on
    correlated GBM assets; returns (Delta, Gamma, Vega), one entry per asset."""
    num_assets = len(s0)
    l = _cholesky(rho)
    rng = Xorshift64Star(seed)
    sqrt_dt = math.sqrt(t)
    discount = math.exp(-r * t)

    acc = [[0.0, 0.0, 0.0] for _ in range(num_assets)]

    for _ in range(num_paths):
        # Independent normals, then correlate: zc = z @ L.T.
        z = [rng.next_std_normal() for _ in range(num_assets)]
        st = [
            s0[j] * math.exp(
                (r - 0.5 * sigma[j] * sigma[j]) * t
                + sigma[j] * sqrt_dt * sum(z[kk] * l[j][kk] for kk in range(num_assets))
            )
            for j in range(num_assets)
        ]
        basket = sum(weights[j] * st[j] for j in range(num_assets))
        payoff = max(basket - k, 0.0)
        for j in range(num_assets):
            zc_j = sum(z[kk] * l[j][kk] for kk in range(num_assets))
            dt_st_j = st[j] * zc_j / sqrt_dt
            d2t_st_j = dt_st_j * (zc_j / sqrt_dt - sigma[j] * sqrt_dt)
            d_sigma_st_j = st[j] * (zc_j * sqrt_dt - sigma[j] * t)
            w = payoff * weights[j]
            acc[j][0] += w * dt_st_j
            acc[j][1] += w * d2t_st_j
            acc[j][2] += w * d_sigma_st_j
    n = float(num_paths)
    delta = [discount * (acc[j][0] / n) / (sigma[j] * s0[j]) for j in range(num_assets)]
    gamma = [
        discount * (acc[j][1] / n) / (sigma[j] * sigma[j] * s0[j] * s0[j])
        for j in range(num_assets)
    ]
    vega = [discount * (acc[j][2] / n) / sigma[j] for j in range(num_assets)]
    return delta, gamma, vega


def asian_option_greeks(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    num_paths: int,
    num_steps: int,
    seed: int,
) -> Greeks3:
    """Malliavin greeks for an arithmetic-average Asian call."""
    rng = Xorshift64Star(seed)
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)
    z = [[rng.next_std_normal() for _ in range(num_steps)] for _ in range(num_paths)]
    paths = _simulate_paths(s0, t, r, sigma, num_steps, z)

    discount = math.exp(-r * t)
    acc = [0.0] * 3
    for zi, srow in zip(z, paths):
        # Average over columns [1, num_steps], as the Python `S[:, 1:]`.
        n_st = float(num_steps)
        s_mean = sum(srow[1:]) / n_st
        payoff = max(s_mean - k, 0.0)
        dt_s_mean = 0.0
        d2t_s_mean = 0.0
        d_sigma_s_mean = 0.0
        for j, zj in enumerate(zi):
            sj = srow[j + 1]
            dt_s = sj * zj / sqrt_dt
            d2t_s = dt_s * (zj / sqrt_dt - sigma * sqrt_dt)
            d_sigma_s = sj * (zj * sqrt_dt - sigma * dt)
            dt_s_mean += dt_s
            d2t_s_mean += d2t_s
            d_sigma_s_mean += d_sigma_s
        dt_s_mean /= n_st
        d2t_s_mean /= n_st
        d_sigma_s_mean /= n_st
        acc[0] += payoff * dt_s_mean
        acc[1] += payoff * d2t_s_mean
        acc[2] += payoff * d_sigma_s_mean
    n = float(num_paths)
    return Greeks3(
        delta=discount * (acc[0] / n) / (sigma * s0),
        gamma=discount * (acc[1] / n) / (sigma * sigma * s0 * s0),
        vega=discount * (acc[2] / n) / sigma,
    )


# ---------------------------------------------------------------------------
# Tests (expected values from quantburn's scripts/gen_greeks.py, float64,
# seed 3001)
# ---------------------------------------------------------------------------

SEED = 3001  # 3000 + module index
TOL = 1e-6  # relative


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), 1e-300) or abs(a - b) < 1e-12


def test_european_matches_reference():
    g = european_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 20_000, SEED)
    assert _close(g.delta, 93.97332927208662, TOL), f"delta {g.delta}"
    assert _close(g.gamma, 7.268677237984516, TOL), f"gamma {g.gamma}"
    assert _close(g.vega, 7991.189624942042, TOL), f"vega {g.vega}"
    assert _close(g.theta, -230.13095761217176, TOL), f"theta {g.theta}"
    assert _close(g.rho, 1395.5863679889742, TOL), f"rho {g.rho}"


def test_digital_matches_reference():
    # epsilon=2.0; the Python default epsilon=1e-4 makes every estimator
    # numerically zero
    g = digital_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 10.0, 20_000, 2.0, SEED)
    assert _close(g.delta, -0.14213961143402573, 1e-6), f"delta {g.delta}"
    assert _close(g.gamma, 0.0029487745130569966, 1e-6), f"gamma {g.gamma}"
    assert _close(g.vega, -33.29994785023318, 1e-6), f"vega {g.vega}"


def test_barrier_matches_reference():
    g = barrier_option_greeks(
        100.0, 110.0, 130.0, 1.0, 0.05, 0.2, "up-and-out", 20_000, 25, SEED
    )
    assert _close(g.delta, 169.99793934820894, TOL), f"delta {g.delta}"
    assert _close(g.gamma, 183.63245445936846, TOL), f"gamma {g.gamma}"
    assert _close(g.vega, -174.05787950305043, TOL), f"vega {g.vega}"

    g2 = barrier_option_greeks(
        100.0, 90.0, 80.0, 1.0, 0.05, 0.2, "down-and-out", 20_000, 25, SEED
    )
    assert _close(g2.delta, 2903.5653727366225, TOL), f"delta {g2.delta}"
    assert _close(g2.gamma, 6034.218595945937, TOL), f"gamma {g2.gamma}"
    assert _close(g2.vega, 1552.748714302966, TOL), f"vega {g2.vega}"


def test_barrier_invalid_type():
    try:
        barrier_option_greeks(100.0, 110.0, 130.0, 1.0, 0.05, 0.2, "knock-in", 10, 5, SEED)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_lookback_matches_reference():
    d = lookback_option_delta(100.0, 1.0, 0.05, 0.2, 20_000, 25, SEED)
    assert _close(d, 17.295340929617094, TOL), f"delta {d}"


def test_basket_matches_reference():
    s0 = [100.0, 90.0, 80.0]
    sigma = [0.2, 0.25, 0.3]
    rho = [
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ]
    weights = [0.4, 0.3, 0.3]
    delta, gamma, vega = basket_option_greeks(s0, 95.0, 1.0, 0.05, sigma, rho, weights, 20_000, SEED)
    d_exp = [22.549108947872366, 14.51060287552215, 11.97730651682888]
    g_exp = [1.804147411523855, 0.9926433261158537, 0.765858818081831]
    v_exp = [1882.6283128582918, 1042.089221101072, 718.1441026308789]
    for j in range(3):
        assert _close(delta[j], d_exp[j], TOL), f"delta[{j}] {delta[j]}"
        assert _close(gamma[j], g_exp[j], TOL), f"gamma[{j}] {gamma[j]}"
        assert _close(vega[j], v_exp[j], TOL), f"vega[{j}] {vega[j]}"


def test_asian_matches_reference():
    g = asian_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 20_000, 25, SEED)
    assert _close(g.delta, 41.90190821064969, TOL), f"delta {g.delta}"
    assert _close(g.gamma, 46.23101947253118, TOL), f"gamma {g.gamma}"
    assert _close(g.vega, 139.4347463474697, TOL), f"vega {g.vega}"


def test_cholesky_reconstruction():
    # Cholesky sanity: L @ L.T == input.
    m = [
        [4.0, 2.0, 1.2],
        [2.0, 5.0, 1.5],
        [1.2, 1.5, 6.0],
    ]
    l = _cholesky(m)
    for i in range(3):
        for j in range(3):
            s = sum(l[i][kk] * l[j][kk] for kk in range(3))
            assert abs(s - m[i][j]) < 1e-12
