import os

# Enable JAX persistent compilation cache before importing jax
# This caches compiled XLA programs to disk, eliminating ~5s of JIT
# compilation overhead on subsequent runs
_cache_dir = os.path.expanduser("~/.cache/sella/jax_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)

import jax

from .optimize import IRC, Sella
from .internal import Internals, Constraints

jax.config.update("jax_enable_x64", True)

# Enable compilation cache with aggressive settings
# Cache all compilations (no minimum time threshold) for faster startup
try:
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass  # JAX version may not support these options

__all__ = ['IRC', 'Sella']
