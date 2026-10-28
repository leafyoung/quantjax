"""Port of quantburn/src/asian_option.rs (itself a port of torchquantlib
core/asset_pricing/option/asian_option.py).

Analytical approximations for Asian options (geometric closed form and
lognormal arithmetic moment-matching), an ``AsianOption`` class with Monte
Carlo (delegating to the local port of quantburn's ``exotics::asian_option``
with the shared ``quantjax.rng.Xorshift64Star`` normals) and finite-difference
greeks.

Monte-Carlo seeding: the Python code calls ``torch.manual_seed(seed)`` with
``seed = 42`` by default; here the seed is an explicit parameter of
``monte_carlo_price`` (default 42).
"""

import math

from quantjax.black_scholes_merton import normal_cdf
from quantjax.rng import Xorshift64Star

EPSILON = 1e-8


def sampling_times(expiry, steps):
    return [expiry * k / steps for k in range(1, steps + 1)]


def average_forward_mean(spot, expiry, rate, steps):
    times = sampling_times(expiry, steps)
    mean = sum(math.exp(rate * t) for t in times) / steps
    return spot * mean


def discounted_intrinsic(option_type, average_level, strike, expiry, rate):
    if option_type == "call":
        payoff = max(average_level - strike, 0.0)
    else:
        payoff = max(strike - average_level, 0.0)
    return math.exp(-rate * expiry) * payoff


def lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate):
    mean_level = max(mean_level, EPSILON)
    second_moment = max(second_moment, mean_level * mean_level)
    log_variance = math.log(max(second_moment / max(mean_level * mean_level, EPSILON), 1.0))

    if log_variance <= EPSILON:
        return discounted_intrinsic(option_type, mean_level, strike, expiry, rate)

    sqrt_log_variance = math.sqrt(log_variance)
    d1 = (math.log(mean_level / strike) + 0.5 * log_variance) / sqrt_log_variance
    d2 = d1 - sqrt_log_variance
    discount_factor = math.exp(-rate * expiry)

    if option_type == "call":
        return discount_factor * (mean_level * normal_cdf(d1) - strike * normal_cdf(d2))
    return discount_factor * (strike * normal_cdf(-d2) - mean_level * normal_cdf(-d1))


def geometric_asian_price(option_type, spot, strike, expiry, volatility, rate, steps):
    times = sampling_times(expiry, steps)
    n = float(steps)
    mean_t = sum(times) / n
    log_spot_mean = math.log(max(spot, EPSILON)) + (rate - 0.5 * volatility * volatility) * mean_t
    # mean over the full n x n covariance matrix min(t_i, t_j)
    cov_mean = 0.0
    for ti in times:
        for tj in times:
            cov_mean += min(ti, tj)
    cov_mean /= n * n
    log_variance = volatility * volatility * cov_mean
    second_moment = math.exp(2.0 * log_spot_mean + 2.0 * log_variance)
    mean_level = math.exp(log_spot_mean + 0.5 * log_variance)
    return lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate)


def arithmetic_asian_lognormal_price(option_type, spot, strike, expiry, volatility, rate, steps):
    times = sampling_times(expiry, steps)
    n = float(steps)
    mean_level = average_forward_mean(spot, expiry, rate, steps)
    second_moment = 0.0
    for ti in times:
        for tj in times:
            second_moment += math.exp(rate * (ti + tj) + volatility * volatility * min(ti, tj))
    second_moment = spot * spot * second_moment / (n * n)
    return lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate)


# --- Monte Carlo core, ported from quantburn/src/exotics.rs::asian_option ---


def _exotics_asian_option(
    option_type,
    spot,
    strike,
    expiry,
    volatility,
    rate,
    num_paths,
    num_steps,
    average_type,
    seed,
):
    if volatility <= 0.0 or expiry <= 0.0:
        raise ValueError("Volatility and expiry must be positive.")
    if num_paths == 0:
        raise ValueError("Number of paths must be a positive integer.")
    if option_type != "call" and option_type != "put":
        raise ValueError("Option type must be either 'call' or 'put'.")
    if average_type != "arithmetic" and average_type != "geometric":
        raise ValueError("Invalid average type")

    dt = expiry / num_steps
    rng = Xorshift64Star(seed)
    drift = (rate - 0.5 * volatility * volatility) * dt
    sig_sqrt_dt = volatility * math.sqrt(dt)
    disc = math.exp(-rate * expiry)

    total = 0.0
    for _ in range(num_paths):
        price = spot
        log_sum = 0.0
        arith_sum = 0.0
        # Include the initial spot in the average, as paths[:, 0] = spot and
        # the Python mean runs over all num_steps + 1 columns.
        arith_sum += spot
        log_sum += math.log(spot)
        for _ in range(num_steps):
            z = rng.next_std_normal()
            price *= math.exp(drift + sig_sqrt_dt * z)
            arith_sum += price
            log_sum += math.log(price)
        if average_type == "arithmetic":
            avg_price = arith_sum / (num_steps + 1)
        else:
            avg_price = math.exp(log_sum / (num_steps + 1))
        if option_type == "call":
            payoff = max(avg_price - strike, 0.0)
        else:
            payoff = max(strike - avg_price, 0.0)
        total += payoff * disc
    return total / num_paths


class AsianOption:
    """Asian option specification and pricing (port of the ``AsianOption`` dataclass)."""

    def __init__(
        self,
        underlying,
        strike,
        expiry,
        rf_rate,
        volatility,
        option_type,
        averaging_type,
        steps,
    ):
        self.underlying = underlying
        self.strike = max(strike, EPSILON)
        self.expiry = expiry
        self.rf_rate = rf_rate
        self.volatility = volatility
        self.option_type = option_type
        self.averaging_type = averaging_type
        self.steps = steps

    def price(self, method, num_simulations, num_steps=None, seed=42):
        if self.option_type != "call" and self.option_type != "put":
            raise ValueError("Option type must be either 'call' or 'put'.")
        if self.averaging_type != "arithmetic" and self.averaging_type != "geometric":
            raise ValueError("average_type must be either 'arithmetic' or 'geometric'.")
        steps = self.steps if num_steps is None else num_steps
        if method == "monte_carlo":
            return self.monte_carlo_price(num_simulations, steps, seed)
        if method == "analytical_approximation":
            return self.analytical_approximation_price(steps)
        raise ValueError("method must be 'monte_carlo' or 'analytical_approximation'.")

    def monte_carlo_price(self, num_simulations, num_steps=None, seed=42):
        steps = self.steps if num_steps is None else num_steps
        if self.volatility <= 0.0:
            average_level = average_forward_mean(self.underlying, self.expiry, self.rf_rate, steps)
            return discounted_intrinsic(
                self.option_type, average_level, self.strike, self.expiry, self.rf_rate
            )
        return _exotics_asian_option(
            self.option_type,
            self.underlying,
            self.strike,
            self.expiry,
            self.volatility,
            self.rf_rate,
            num_simulations,
            steps,
            self.averaging_type,
            seed,
        )

    def analytical_approximation_price(self, steps=None):
        steps = self.steps if steps is None else steps
        if self.volatility <= 0.0:
            average_level = average_forward_mean(self.underlying, self.expiry, self.rf_rate, steps)
            return discounted_intrinsic(
                self.option_type, average_level, self.strike, self.expiry, self.rf_rate
            )
        if self.averaging_type == "geometric":
            return geometric_asian_price(
                self.option_type,
                self.underlying,
                self.strike,
                self.expiry,
                self.volatility,
                self.rf_rate,
                steps,
            )
        return arithmetic_asian_lognormal_price(
            self.option_type,
            self.underlying,
            self.strike,
            self.expiry,
            self.volatility,
            self.rf_rate,
            steps,
        )

    def _finite_difference_greek(self, field, bump_size, sign):
        base = getattr(self, field)
        up = self._bumped(field, base + bump_size)
        down = self._bumped(field, base - bump_size)
        return sign * (up - down) / (2.0 * bump_size)

    def _bumped(self, field, value):
        o = AsianOption(
            self.underlying,
            self.strike,
            self.expiry,
            self.rf_rate,
            self.volatility,
            self.option_type,
            self.averaging_type,
            self.steps,
        )
        setattr(o, field, value)
        return o.analytical_approximation_price(None)

    def delta(self):
        return self._finite_difference_greek("underlying", 1e-2, 1.0)

    def gamma(self):
        bump = 1e-2
        price_center = self.analytical_approximation_price(None)
        price_up = self._bumped("underlying", self.underlying + bump)
        price_down = self._bumped("underlying", self.underlying - bump)
        return (price_up - 2.0 * price_center + price_down) / (bump * bump)

    def theta(self):
        return self._finite_difference_greek("expiry", 1e-4, -1.0)

    def vega(self):
        return self._finite_difference_greek("volatility", 1e-4, 1.0)

    def rho(self):
        return self._finite_difference_greek("rf_rate", 1e-4, 1.0)


def asian_option(
    option_type,
    spot,
    strike,
    expiry,
    volatility,
    rate,
    steps,
    average_type,
    num_paths,
    seed,
):
    """Free function matching the Python ``asian_option`` entry point:
    ``num_paths = None`` -> analytical approximation, otherwise Monte Carlo."""
    if option_type != "call" and option_type != "put":
        raise ValueError("Option type must be either 'call' or 'put'.")
    option = AsianOption(spot, strike, expiry, rate, volatility, option_type, average_type, steps)
    if num_paths is None:
        return option.analytical_approximation_price(steps)
    return option.monte_carlo_price(num_paths, steps, seed)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# --- tests (expected values from quantburn's asian_option.rs tests,
# generated by scripts/gen_asian_option.py, pure f64; MC uses the shared
# Xorshift64Star normals with the same seeds) ---


def test_analytical_geometric_call():
    o = AsianOption(100.0, 100.0, 1.0, 0.05, 0.2, "call", "geometric", 252)
    assert abs(o.analytical_approximation_price(None) - 5.565508831349255) < 1e-9


def test_analytical_arithmetic_call():
    o = AsianOption(100.0, 100.0, 1.0, 0.05, 0.2, "call", "arithmetic", 252)
    assert abs(o.analytical_approximation_price(None) - 5.801498938949076) < 1e-9


def test_analytical_arithmetic_put():
    o = AsianOption(100.0, 110.0, 1.0, 0.05, 0.2, "put", "arithmetic", 52)
    assert abs(o.analytical_approximation_price(None) - 9.087690947769374) < 1e-9


def test_analytical_zero_volatility():
    o = AsianOption(100.0, 100.0, 1.0, 0.05, 0.0, "call", "arithmetic", 252)
    # mean forward level over times k/252, k=1..252, discounted payoff
    assert abs(o.analytical_approximation_price(None) - 2.4278855699852664) < 1e-9


def test_monte_carlo_call():
    o = AsianOption(100.0, 100.0, 1.0, 0.05, 0.2, "call", "arithmetic", 50)
    assert abs(o.monte_carlo_price(1000, None, 42) - 5.343075879923734) < 1e-9


def test_free_function_entry_points():
    assert (
        abs(asian_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 252, "geometric", None, 42) - 5.565508831349255)
        < 1e-9
    )
    assert (
        abs(asian_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 50, "arithmetic", 1000, 42) - 5.343075879923734)
        < 1e-9
    )


def test_greeks():
    # Finite differences over ~1e-2/1e-4 bumps cancel ~1e-4 of the price,
    # amplifying cross-libm (Python vs Rust exp) ulp differences; the
    # convention's 1e-6 relative tolerance applies here.
    o = AsianOption(100.0, 100.0, 1.0, 0.05, 0.2, "call", "arithmetic", 252)
    assert _rel(o.delta(), 0.5931734274305711) < 1e-6
    assert _rel(o.gamma(), 0.032109755547793384) < 1e-6
    assert _rel(o.theta(), -3.432807050329778) < 1e-6
    assert _rel(o.vega(), 21.947043359307195) < 1e-6
    assert _rel(o.rho(), 24.762053642102266) < 1e-6
