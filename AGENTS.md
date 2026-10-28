# AGENTS.md — quantjax

JAX port of torchquantlib, mirroring `../quantburn` / `../quantcandle`
module-for-module. Read `../quantburn/CONVENTIONS.md` before converting further.

## Commands

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m pytest -q   # 133 tests
.venv/bin/python -m quantjax.cli [--backend cpu|rocm] --cases cases.json
```

## Rules

- Closed-form math in plain float64 — no jax for scalars. jnp only for vector
  work (currently `heston.simulate_terminal_paths_tensor`).
- Stochastic code must use `quantjax.rng.Xorshift64Star` (bit-identical to
  quantburn/quantcandle) — never jax.random (different stream, breaks parity).
- Expected test literals come from quantburn's Rust tests; do not loosen
  tolerances. If Python disagrees with a Rust literal, the port is wrong.
- ROCM runs: `XLA_PYTHON_CLIENT_PREALLOCATE=false` and a LOW
  `XLA_PYTHON_CLIENT_MEM_FRACTION` (0.25) — unified memory box, see
  ../tf-quant-finance/AGENTS.md.
