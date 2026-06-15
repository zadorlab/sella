import os

# Enable JAX persistent compilation cache before importing jax.
# This caches expensive compiled XLA programs to disk, eliminating ~5s of
# JIT compilation overhead on subsequent runs.
#
# The cache location is taken from JAX_COMPILATION_CACHE_DIR if set, otherwise
# it falls back to ~/.cache/sella/jax_cache. On heterogeneous SLURM clusters,
# point this at a path shared by nodes of the same CPU architecture to avoid
# repeated recompilation (and away from slow/quota-limited home filesystems).
_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.expanduser(
    "~/.cache/sella/jax_cache"
)
os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)
# JAX is used only for AD, not linalg; GPU linalg routes through torch
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from .optimize import IRC, Sella
from .internal import Internals, Constraints

jax.config.update("jax_enable_x64", True)

# Only persist compilations that are actually expensive to redo. Anything
# that compiles in <2s is not worth the cache I/O, especially when the cache
# lives on a networked filesystem.
try:
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
except Exception:
    pass  # JAX version may not support these options

__all__ = ['IRC', 'Sella']
