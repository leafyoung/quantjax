# quantjax

JAX port of `torchquantlib`, mirroring `../quantburn` and `../quantcandle`
module-for-module: same 33 modules, same f64 numerics, same bit-exact
`Xorshift64Star` RNG, same embedded expected values. This is the fourth
implementation of the same numerics, so the cross-check suite at
`../cross-check` validates it lib-vs-lib against all the others.

- Closed-form modules: pure float64 Python (precision over framework purity),
  formulas identical to torchquantlib / quantburn.
- `heston.simulate_terminal_paths_tensor`: the genuine JAX path — full-truncation
  Euler as jnp array ops (float64; x64 enabled in `__init__`), normals still
  drawn host-side from the shared RNG so all four libraries see identical draws.
- Parity: 133 embedded tests, expected values are quantburn's test literals.
  Green on CPU jax and ROCM jax (gfx1151; run with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, low `XLA_PYTHON_CLIENT_MEM_FRACTION`).

## Cross-check runner

    quantjax/.venv/bin/python -m quantjax.cli [--backend cpu|rocm] [--cases cases.json]

Reads the runtime-agnostic case JSON shared with `../cross-check`; supports the
same 21 models as quantcandle's bin (the 13 tfq_ analytic cases are tff-only).
