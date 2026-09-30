"""Linkage-synthesis framework for 2.156 CP1 (the kangaroo curves)."""

import os

# LINKS runs on JAX; pin it to the CPU before anything imports jax
# (there is no usable JAX GPU backend on Apple Silicon or on the CI runners).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
