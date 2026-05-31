import os

# Enable JAX persistent compilation cache before importing jax
# This caches compiled XLA programs to disk, eliminating ~5s of JIT
# compilation overhead on subsequent runs
# Cache dir is given by an environment variable, or a fallback if not set
# Explicitly setting the environment variable is recommended if the home
# directory is not writable
default_cache_dir = os.path.expanduser("~/.cache/sella/jax_cache")
_cache_dir = os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                                   default_cache_dir)
os.makedirs(_cache_dir, exist_ok=True)
# JAX is used only for AD, not linalg; GPU linalg routes through torch
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
