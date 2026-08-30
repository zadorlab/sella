"""Tests for runtime CPU-thread capping (sella._threads).

These verify that the thread controls used to keep co-located Sella processes
from oversubscribing the CPU actually take effect at runtime.
"""
import ctypes
import os
import re

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT

import sella
from sella import Sella, configure_compute, set_cpu_threads


def _loaded_openblas_thread_counts():
    """Return {lib_basename: current_thread_count} for loaded OpenBLAS copies.

    Returns an empty dict on platforms without /proc/self/maps or when no
    OpenBLAS get-threads symbol is exposed, so callers can skip gracefully.
    """
    counts = {}
    try:
        with open("/proc/self/maps") as fh:
            maps = fh.read()
    except OSError:
        return counts
    getters = (
        "openblas_get_num_threads",
        "openblas_get_num_threads64_",
        "scipy_openblas_get_num_threads",
        "scipy_openblas_get_num_threads64_",
    )
    for lib_path in sorted(set(re.findall(r"\S+openblas\S*\.so\S*", maps, re.I))):
        try:
            lib = ctypes.CDLL(lib_path)
        except OSError:
            continue
        for sym in getters:
            fn = getattr(lib, sym, None)
            if fn is not None:
                fn.restype = ctypes.c_int
                counts[os.path.basename(lib_path)] = fn()
                break
    return counts


@pytest.fixture(autouse=True)
def _restore_threads():
    """Restore torch's thread count after each test (process-wide state)."""
    try:
        import torch
        original = torch.get_num_threads()
    except Exception:
        torch = None
        original = None
    yield
    if torch is not None and original is not None:
        try:
            torch.set_num_threads(original)
        except Exception:
            pass


def test_set_cpu_threads_caps_torch():
    torch = pytest.importorskip("torch")
    set_cpu_threads(2)
    assert torch.get_num_threads() == 2


def test_set_cpu_threads_sets_env_vars():
    set_cpu_threads(3)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[var] == "3"


def test_set_cpu_threads_caps_loaded_openblas():
    # Force numpy's (and scipy's, if present) OpenBLAS to load.
    np.linalg.svd(np.random.rand(8, 8))
    before = _loaded_openblas_thread_counts()
    if not before:
        pytest.skip("no introspectable OpenBLAS loaded")
    set_cpu_threads(2)
    after = _loaded_openblas_thread_counts()
    for lib, n in after.items():
        assert n == 2, f"{lib} not capped: {n}"


def test_none_is_noop():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(5)
    set_cpu_threads(None)
    assert torch.get_num_threads() == 5


def test_clamped_to_at_least_one():
    torch = pytest.importorskip("torch")
    set_cpu_threads(0)
    assert torch.get_num_threads() == 1


def test_configure_compute_caps_torch():
    torch = pytest.importorskip("torch")
    configure_compute(max_cpu_threads=2)
    assert torch.get_num_threads() == 2


def test_sella_init_accepts_max_cpu_threads():
    torch = pytest.importorskip("torch")
    atoms = molecule("C2H6")
    atoms.calc = EMT()
    Sella(atoms, order=0, internal=True, max_cpu_threads=2, logfile=None)
    assert torch.get_num_threads() == 2
