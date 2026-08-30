"""Runtime CPU-thread limiting for co-locating multiple Sella processes.

Sella is heavily CPU-BLAS bound (internal coordinates, geodesic ODE, Hessian
updates), and nothing in the stack caps thread pools by default: numpy/scipy
OpenBLAS and the torch CPU pools each grab as many threads as there are cores.
Running N Sella processes on one machine (e.g. sharing a single GPU card) then
oversubscribes the CPU N-fold, which thrashes and slows every process.

``configure_compute(max_cpu_threads=...)`` lets whatever launches the Sella
processes hand each one a slice of the cores (typically ``cpu_count // N``).

Only CPU threads are handled here. PyTorch/CUDA expose no runtime per-process
cap on GPU *compute* threads, so partitioning a shared GPU is left to the
driver / CUDA MPS / the launcher.
"""

import ctypes
import logging
import os
import re

logger = logging.getLogger(__name__)

# OpenBLAS runtime thread setters. numpy and scipy ship *separate* OpenBLAS
# builds (ILP64 vs LP64) whose symbols are renamed with build-specific
# suffixes, so we scan every loaded OpenBLAS library and try each known name.
_OPENBLAS_SETTERS = (
    "openblas_set_num_threads",
    "openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads",
    "scipy_openblas_set_num_threads64_",
    "goto_set_num_threads",
)

# Env vars covering the common CPU math backends. Setting these is the only
# lever for pools not yet loaded (MKL, OMP regions in scipy/JAX) and for any
# child processes the caller spawns; already-loaded OpenBLAS is handled by the
# runtime setters above.
_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _limit_loaded_openblas(n):
    """Cap every OpenBLAS library already mapped into this process to ``n``.

    Returns the list of library paths that were successfully capped. Relies on
    ``/proc/self/maps`` and is a no-op on platforms without it.
    """
    capped = []
    try:
        with open("/proc/self/maps") as fh:
            maps = fh.read()
    except OSError:
        return capped
    for lib_path in sorted(set(re.findall(r"\S+openblas\S*\.so\S*", maps, re.I))):
        try:
            lib = ctypes.CDLL(lib_path)
        except OSError:
            continue
        for sym in _OPENBLAS_SETTERS:
            setter = getattr(lib, sym, None)
            if setter is None:
                continue
            try:
                setter(ctypes.c_int(n))
                capped.append(lib_path)
                break
            except Exception:
                continue
    return capped


def set_cpu_threads(n):
    """Cap CPU math parallelism (BLAS/LAPACK + torch) to ``n`` threads.

    Applied at runtime, so it takes effect even though numpy/torch are already
    imported by the time Sella runs:

      * OpenBLAS (numpy's and scipy's separate copies) via their runtime setters
      * torch intra-op (and best-effort inter-op) thread pools
      * ``*_NUM_THREADS`` env vars, for pools loaded later and child processes

    ``n`` is clamped to ``>= 1``. Failures are logged, never raised, so a
    thread-control hiccup can't break an optimization. ``None`` is a no-op.
    """
    if n is None:
        return
    n = max(1, int(n))

    for var in _ENV_VARS:
        os.environ[var] = str(n)

    capped = _limit_loaded_openblas(n)
    logger.debug("Capped %d OpenBLAS lib(s) to %d thread(s)", len(capped), n)

    try:
        import torch
    except Exception:
        return
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        # The inter-op pool is fixed once parallel work has started; leaving it
        # is harmless (it only sizes cross-op parallelism, not the BLAS pool).
        pass
    except Exception:
        pass


def configure_compute(max_cpu_threads=None):
    """Configure this process's compute-resource share for Sella.

    Meant to be called once by whatever spins up the Sella processes so each
    one uses only its allotted slice of the cores. Safe before or after building
    a ``Sella`` object; earlier is better (it caps the pools before they work).

    Only CPU threads are handled: PyTorch/CUDA expose no runtime per-process cap
    on GPU compute threads, so partitioning a shared GPU is left to the driver /
    CUDA MPS / the launcher.

    Parameters
    ----------
    max_cpu_threads : int, optional
        Maximum CPU threads for all CPU math (OpenBLAS/LAPACK + torch). Sella is
        CPU-BLAS bound, so when co-locating ``N`` processes on one machine set
        this to roughly ``cpu_count // N`` to avoid oversubscription. ``None``
        (default) leaves library defaults untouched.
    """
    set_cpu_threads(max_cpu_threads)
