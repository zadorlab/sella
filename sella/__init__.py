import os

# Enable JAX persistent compilation cache before importing jax
# This caches compiled XLA programs to disk, eliminating ~5s of JIT
# compilation overhead on subsequent runs
# Cache dir is given by an environment variable, or a fallback if not set
# Explicitly setting the environment variable is recommended if the home
# directory is not writable
_cache_dir = os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                                   os.path.expanduser("~/.cache/sella/jax_cache"))
# Guard the mkdir: a read-only or unwritable home must not crash import.
# If the directory can't be created, disable the persistent cache instead.
try:
    os.makedirs(_cache_dir, exist_ok=True)
except OSError:
    _cache_dir = None
# JAX is used only for AD, not linalg; GPU linalg routes through torch
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from .optimize import IRC, Sella
from .internal import Internals, Constraints
from ._threads import configure_compute, set_cpu_threads

jax.config.update("jax_enable_x64", True)

# Enable compilation cache with aggressive settings
# Cache all compilations (no minimum time threshold) for faster startup
if _cache_dir is not None:
    try:
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception:
        pass  # JAX version may not support these options

__all__ = ['IRC', 'Sella', 'configure_compute', 'set_cpu_threads']
