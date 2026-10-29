"""Port of quantburn/src/greeks.rs (originally greeks/greeks.py,
``MalliavinGreeks``, re-exported by greeks/malliavin.py as ``Greeks``).
Monte-Carlo option greeks via likelihood-ratio (integration-by-parts) /
pathwise weights.

Estimator-correctness note (2026-10-29): ``european_option_greeks`` and
``digital_option_greeks`` were re-derived with proper LR / pathwise weights
and verified against QuantLib 1.44 ``AnalyticEuropeanEngine`` greeks at 500k
paths over three parameter sets (relative errors: delta <= 1.4e-3,
gamma <= 1.0e-2, vega <= 1.5e-2, theta <= 2.8e-3, rho <= 2.9e-3; digital
kernel bias at epsilon = 2.0 by exact quadrature: delta -0.50%, gamma -0.65%,
vega +0.63%). The previous implementation multiplied the payoff by pathwise
derivatives of S_T (e.g. ``S_T*Z/sqrt(T)``) where LR weights are required,
biasing every European greek by 26x-386x (see cross-check
results/risk_golden.json). ``barrier_option_greeks``,
``lookback_option_delta``, ``basket_option_greeks`` and
``asian_option_greeks`` still use the OLD weight construction and are
BIASED / UNDER REVIEW (same extra-S_T-factor defect; left unchanged).

Stochasticity: all standard normals come from quantjax.rng.Xorshift64Star
(Box-Muller, bit-exact against quantburn's scripts/refgen.py). Draw order
replicates the row-major fill of the torch tensors: for an
(num_paths, num_steps) draw, path-major then step-minor. Tests use seed 3001.

Math is done in plain Python floats (float64). Expected values come from the
float64 reference generator quantburn/scripts/gen_greeks.py (same normals,
same seed); tolerance 1e-6 relative (1e-4 for the digital greeks, where the
Gaussian-kernel Dirac approximation makes the estimators large and
ill-conditioned).
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from quantjax.lmm import cholesky
from quantjax.rng import Xorshift64Star


@dataclass
class MalliavinGreeks:
    """Mirror of the Python ``MalliavinGreeks`` class (a name-space handle;
    all numeric work is in the module-level functions below)."""

    @staticmethod
    def european_option_greeks(s0, k, t, r, sigma, num_paths, seed):
        return european_option_greeks(s0, k, t, r, sigma, num_paths, seed)


# Alias mirroring the Python ``Greeks = MalliavinGreeks`` re-export.
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


def european_option_greeks(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    num_paths: int,
    seed: int,
) -> EuropeanGreeks:
    """Greeks for a European call under GBM (single-step exact terminal
    simulation, S_T = S0*exp((r - sigma^2/2)T + sigma*sqrt(T)*Z), q = 0).

    Estimators (QuantLib-verified, see module doc):
      delta (likelihood-ratio): disc * E[payoff * Z] / (sigma*S0*sqrt(T))
      gamma (2nd-order LR):     disc * [E[payoff*(Z^2-1)]
                                        - sigma*sqrt(T)*E[payoff*Z]]
                                / (sigma^2 * T * S0^2)
      vega (pathwise):          disc * E[1{S_T>K} * S_T * (sqrt(T)*Z - sigma*T)]
      theta (BS PDE identity):  r*price - r*S0*delta - sigma^2*S0^2*gamma/2
      rho   (identity, q=0):    T * (S0*delta - price)
    """
    rng = Xorshift64Star(seed)
    sqrt_t = math.sqrt(t)
    drift = (r - 0.5 * sigma * sigma) * t
    discount = math.exp(-r * t)

    # LR / pathwise accumulators (the old S_T-scaled derivatives of S_T
    # biased every greek by 26x-386x -- see module doc).
    s_pz = s_pz2 = s_vega = s_p = 0.0
    for _ in range(num_paths):
        z = rng.next_std_normal()
        st = s0 * math.exp(drift + sigma * sqrt_t * z)
        payoff = max(st - k, 0.0)
        s_pz += payoff * z
        s_pz2 += payoff * (z * z - 1.0)
        if st > k:
            s_vega += st * (sqrt_t * z - sigma * t)
        s_p += payoff
    n = float(num_paths)
    price = discount * s_p / n
    delta = discount * (s_pz / n) / (sigma * s0 * sqrt_t)
    gamma = discount * (s_pz2 / n - sigma * sqrt_t * s_pz / n) / (sigma**2 * t * s0**2)
    vega = discount * s_vega / n
    theta = r * price - r * s0 * delta - 0.5 * sigma**2 * s0**2 * gamma
    rho = t * (s0 * delta - price)
    return EuropeanGreeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
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
    """Greeks for a digital (cash-or-nothing) call under GBM, with the Dirac
    delta approximated by a Gaussian kernel of width ``epsilon``
    (K_eps(x) = phi(x)/epsilon, x = (S_T-K)/epsilon, smoothed payoff
    h(S_T) = Q*Phi((S_T-K)/epsilon)).

    Pathwise-kernel estimators (QuantLib-verified, see module doc; residual
    kernel bias at epsilon = 2.0: delta -0.50%, gamma -0.65%, vega +0.63%):
      delta: disc * Q * E[K_eps * S_T / S0]
      gamma: disc * Q * E[dK_eps/ds * (S_T/S0)^2],
             dK_eps/ds = -(x/epsilon) * K_eps  (h''(s)/Q; note the tempting
             -(x/epsilon^2) form carries a spurious extra 1/epsilon)
      vega:  disc * Q * E[K_eps * S_T * (sqrt(T)*Z - sigma*T)]
    """
    rng = Xorshift64Star(seed)
    sqrt_t = math.sqrt(t)
    drift = (r - 0.5 * sigma * sigma) * t
    discount = math.exp(-r * t)
    normalizer = epsilon * math.sqrt(2.0 * math.pi)

    sd = sg = sv = 0.0
    for _ in range(num_paths):
        z = rng.next_std_normal()
        st = s0 * math.exp(drift + sigma * sqrt_t * z)
        x = (st - k) / epsilon
        k_eps = (1.0 / normalizer) * math.exp(-0.5 * x * x)
        k_eps_s = -(x / epsilon) * k_eps  # dK_eps/ds, see doc
        u = st / s0
        sd += k_eps * u
        sg += k_eps_s * u * u
        sv += k_eps * st * (sqrt_t * z - sigma * t)
    n = float(num_paths)
    return Greeks3(
        delta=discount * q * sd / n,
        gamma=discount * q * sg / n,
        vega=discount * q * sv / n,
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
    ``z[path][step]``; returns S[0..=steps] per path."""
    dt = t / num_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    paths = []
    for row in z:
        path = [s0]
        prev = s0
        for zi in row:
            prev = prev * math.exp(drift + vol * zi)
            path.append(prev)
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
    """Malliavin greeks for a knock-out barrier call (``up-and-out`` or
    ``down-and-out``), monitored at each simulation step.

    .. warning::
        BIASED / UNDER REVIEW (2026-10-29): this estimator still uses the
        old weight construction (payoff * S_T-scaled pathwise derivatives of
        S_T where likelihood-ratio weights are required), the same defect
        that biased ``european_option_greeks`` by 26x-386x. Values are NOT
        trustworthy; left unchanged pending a QuantLib-verified
        re-derivation.
    """
    if option_type != "up-and-out" and option_type != "down-and-out":
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
        hit_barrier = any((s >= h) if up else (s <= h) for s in srow[1:])
        if hit_barrier:
            # payoff and all Malliavin weights are zeroed by (~hit_barrier)
            continue
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
    """Malliavin delta for a floating-strike lookback call.

    .. warning::
        BIASED / UNDER REVIEW (2026-10-29): old weight construction (see the
        module docstring); left unchanged pending a QuantLib-verified
        re-derivation.
    """
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
        # the Python ``S[:, :-1]``).
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
        if count > 0.0:
            dt_smin /= count
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
    correlated GBM assets; returns (Delta, Gamma, Vega), one entry per asset.

    .. warning::
        BIASED / UNDER REVIEW (2026-10-29): old weight construction (see the
        module docstring); left unchanged pending an independent
        (e.g. common-random-number finite-difference) verification.
    """
    num_assets = len(s0)
    l = cholesky(rho)
    rng = Xorshift64Star(seed)
    dt = t
    sqrt_dt = math.sqrt(dt)
    discount = math.exp(-r * t)

    acc = [[0.0] * 3 for _ in range(num_assets)]

    for _ in range(num_paths):
        # Independent normals, then correlate: zc = z @ L.T.
        z = [rng.next_std_normal() for _ in range(num_assets)]
        st = []
        for j in range(num_assets):
            zc = sum(z[kk] * l[j][kk] for kk in range(num_assets))
            st.append(
                s0[j]
                * math.exp(
                    (r - 0.5 * sigma[j] * sigma[j]) * t + sigma[j] * sqrt_dt * zc
                )
            )
        basket = sum(weights[j] * st[j] for j in range(num_assets))
        payoff = max(basket - k, 0.0)
        for j in range(num_assets):
            zc_j = sum(z[kk] * l[j][kk] for kk in range(num_assets))
            dt_st_j = st[j] * zc_j / sqrt_dt
            d2t_st_j = dt_st_j * (zc_j / sqrt_dt - sigma[j] * sqrt_dt)
            d_sigma_st_j = st[j] * (zc_j * sqrt_dt - sigma[j] * dt)
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
    """Malliavin greeks for an arithmetic-average Asian call.

    .. warning::
        BIASED / UNDER REVIEW (2026-10-29): old weight construction (see the
        module docstring); left unchanged pending an independent
        verification.
    """
    rng = Xorshift64Star(seed)
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)
    z = [[rng.next_std_normal() for _ in range(num_steps)] for _ in range(num_paths)]
    paths = _simulate_paths(s0, t, r, sigma, num_steps, z)

    discount = math.exp(-r * t)
    acc = [0.0] * 3
    for zi, srow in zip(z, paths):
        # Average over columns [1, num_steps], as the Python ``S[:, 1:]``.
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
# Tests (expected values from quantburn/src/greeks.rs, scripts/gen_greeks.py)
# ---------------------------------------------------------------------------

_SEED = 3001  # 3000 + module index
_TOL = 1e-6  # relative


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), 1e-300) or abs(a - b) < 1e-12


def test_european_matches_reference():
    g = european_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 20_000, _SEED)
    # gen_greeks.py: european (2026-10-29 LR/pathwise weights; old literals
    # encoded the biased S_T-derivative estimator, e.g. delta 93.97 vs the
    # correct ~0.65)
    assert _close(g.delta, 0.6499984107108641, _TOL), f"delta {g.delta}"
    assert _close(g.gamma, 0.02041962761268718, _TOL), f"gamma {g.gamma}"
    assert _close(g.vega, 38.012287272987834, _TOL), f"vega {g.vega}"
    assert _close(g.theta, -6.8060708622095625, _TOL), f"theta {g.theta}"
    assert _close(g.rho, 54.44290679344252, _TOL), f"rho {g.rho}"


def test_digital_matches_reference():
    # epsilon=2.0, corrected 2026-10-29 pathwise-kernel weights (residual
    # kernel bias at eps=2.0: delta -0.50%, gamma -0.65%, vega +0.63%); the
    # Python default epsilon=1e-4 makes every estimator numerically zero
    # (documented quirk in quantburn's greeks.rs).
    g = digital_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 10.0, 20_000, 2.0, _SEED)
    assert _close(g.delta, 0.19085986706830554, 1e-6), f"delta {g.delta}"
    assert _close(g.gamma, -0.003820124062759267, 1e-6), f"gamma {g.gamma}"
    assert _close(g.vega, -6.659989570046656, 1e-6), f"vega {g.vega}"


def test_barrier_matches_reference():
    g = barrier_option_greeks(
        100.0, 110.0, 130.0, 1.0, 0.05, 0.2, "up-and-out", 20_000, 25, _SEED
    )
    assert _close(g.delta, 169.99793934820894, _TOL), f"delta {g.delta}"
    assert _close(g.gamma, 183.63245445936846, _TOL), f"gamma {g.gamma}"
    assert _close(g.vega, -174.05787950305043, _TOL), f"vega {g.vega}"

    g2 = barrier_option_greeks(
        100.0, 90.0, 80.0, 1.0, 0.05, 0.2, "down-and-out", 20_000, 25, _SEED
    )
    assert _close(g2.delta, 2903.5653727366225, _TOL), f"delta {g2.delta}"
    assert _close(g2.gamma, 6034.218595945937, _TOL), f"gamma {g2.gamma}"
    assert _close(g2.vega, 1552.748714302966, _TOL), f"vega {g2.vega}"


def test_barrier_invalid_type():
    import pytest

    with pytest.raises(ValueError):
        barrier_option_greeks(100.0, 110.0, 130.0, 1.0, 0.05, 0.2, "knock-in", 10, 5, _SEED)


def test_lookback_matches_reference():
    d = lookback_option_delta(100.0, 1.0, 0.05, 0.2, 20_000, 25, _SEED)
    assert _close(d, 17.295340929617094, _TOL), f"delta {d}"


def test_basket_matches_reference():
    s0 = [100.0, 90.0, 80.0]
    sigma = [0.2, 0.25, 0.3]
    rho = [
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ]
    weights = [0.4, 0.3, 0.3]
    delta, gamma, vega = basket_option_greeks(
        s0, 95.0, 1.0, 0.05, sigma, rho, weights, 20_000, _SEED
    )
    d_exp = [22.549108947872366, 14.51060287552215, 11.97730651682888]
    g_exp = [1.804147411523855, 0.9926433261158537, 0.765858818081831]
    v_exp = [1882.6283128582918, 1042.089221101072, 718.1441026308789]
    for j in range(3):
        assert _close(delta[j], d_exp[j], _TOL), f"delta[{j}] {delta[j]}"
        assert _close(gamma[j], g_exp[j], _TOL), f"gamma[{j}] {gamma[j]}"
        assert _close(vega[j], v_exp[j], _TOL), f"vega[{j}] {vega[j]}"


def test_asian_matches_reference():
    g = asian_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 20_000, 25, _SEED)
    assert _close(g.delta, 41.90190821064969, _TOL), f"delta {g.delta}"
    assert _close(g.gamma, 46.23101947253118, _TOL), f"gamma {g.gamma}"
    assert _close(g.vega, 139.4347463474697, _TOL), f"vega {g.vega}"


def test_cholesky_reconstruction():
    # Cholesky sanity: L @ L.T == input.
    m = [
        [4.0, 2.0, 1.2],
        [2.0, 5.0, 1.5],
        [1.2, 1.5, 6.0],
    ]
    l = cholesky(m)
    for i in range(3):
        for j in range(3):
            s = sum(l[i][kk] * l[j][kk] for kk in range(3))
            assert abs(s - m[i][j]) < 1e-12
