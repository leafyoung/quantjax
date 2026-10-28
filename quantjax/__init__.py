"""quantjax — JAX port of torchquantlib, mirroring quantburn/quantcandle.

Conventions (quantburn's CONVENTIONS.md, adapted):
- closed-form scalar math in float64 (plain Python floats / jnp float64),
  formulas identical to the Python source of torchquantlib;
- stochastic modules draw standard normals from `quantjax.rng.Xorshift64Star`
  (bit-identical to the Rust sampler and its Python reference harness) on the
  host; JAX arrays are used for the vector work (e.g. Heston Euler paths);
- every module carries embedded parity tests whose expected values are the
  same literals as quantburn's tests (same RNG, same f64 math).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

from quantjax import (american_option, asian_option, bermudan_option,
                      binomial_tree_model, black_karasinski,
                      black_scholes_merton, bond_pricer, cir, cir as CIR,
                      dupire_local_volatility, exotics, expected_shortfall,
                      fixed_income_forward, future_pricer, gans, greeks,
                      heston, hull_white, letsberational, lmm, malliavin,
                      option_utils, reduced_form_model, rng, sabr,
                      scenario_analysis, stochastic_model, stress_testing,
                      structural_model, valuation_adjustment, var, vasicek,
                      yield_curve_construction)
