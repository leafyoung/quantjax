import os
# gfx1151 unified-memory defense (see tf-quant-finance AGENTS.md): never let
# JAX pre-allocate; CPU jax needs no device memory but keep the guard anyway.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
