"""Centralized GPU helpers for Sella.

Provides numpy-in / numpy-out wrappers for eigh and QR that route through
torch+CUDA when a usable GPU is present, with a single shared `_has_torch`
state and a size threshold that gates upload overhead.

GPU usage is opt-out via SELLA_DISABLE_GPU=1 and the size threshold is
tunable via SELLA_GPU_MIN_DIM (default 200). On a torch CUDA OOM the call
falls back to CPU and records the failure so subsequent calls skip the GPU
attempt for that size; this keeps smaller GPUs from thrashing.
"""

import os
import numpy as np
from scipy.linalg import eigh as _cpu_eigh

try:
    import torch
    _has_torch = torch.cuda.is_available()
except Exception:
    torch = None
    _has_torch = False

if os.environ.get("SELLA_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
    _has_torch = False

try:
    _GPU_MIN_DIM = int(os.environ.get("SELLA_GPU_MIN_DIM", "200"))
except ValueError:
    _GPU_MIN_DIM = 200

# After a CUDA OOM at dimension N, refuse subsequent GPU offload for shapes
# >= N. Keeps a single failure from cascading and lets the CPU path take over
# cleanly on small GPUs.
_oom_floor = None


def _gpu_ok(n):
    return _has_torch and n >= _GPU_MIN_DIM and (
        _oom_floor is None or n < _oom_floor
    )


def _record_oom(n):
    global _oom_floor
    if _oom_floor is None or n < _oom_floor:
        _oom_floor = n
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def to_gpu(A):
    """Upload a numpy array to GPU as a contiguous float64 tensor.

    Returns the torch tensor, or None if no GPU is available / upload fails.
    Caller is responsible for size-gating with `_gpu_ok` if desired.
    """
    if not _has_torch:
        return None
    try:
        return torch.from_numpy(np.ascontiguousarray(A)).cuda()
    except (RuntimeError, MemoryError):
        _record_oom(A.shape[0] if A.ndim >= 1 else 0)
        return None


def gpu_eigh(A, A_gpu=None):
    """Eigendecomposition. GPU when beneficial, CPU otherwise.

    If A_gpu (a torch tensor on CUDA) is supplied, the upload is skipped.
    """
    n = A.shape[0]
    if A_gpu is not None or _gpu_ok(n):
        try:
            At = A_gpu if A_gpu is not None else to_gpu(A)
            if At is not None:
                evals_t, evecs_t = torch.linalg.eigh(At)
                return evals_t.cpu().numpy(), evecs_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return _cpu_eigh(A)


def gpu_eigh_t(A_gpu):
    """Eigendecomposition that returns torch tensors (no .cpu() download).

    Caller has already uploaded; returns (evals_t, evecs_t) on GPU, or
    (None, None) on OOM so caller can fall back.
    """
    try:
        return torch.linalg.eigh(A_gpu)
    except (RuntimeError, MemoryError):
        _record_oom(A_gpu.shape[0])
        return None, None


def gpu_qr(A):
    """Economy QR. GPU when beneficial, CPU otherwise."""
    n = A.shape[0]
    if _gpu_ok(n):
        try:
            At = to_gpu(A)
            if At is not None:
                Q_t, R_t = torch.linalg.qr(At, mode='reduced')
                return Q_t.cpu().numpy(), R_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return np.linalg.qr(A, mode='reduced')


def gpu_project(H, U, H_gpu=None):
    """Compute U.T @ H @ U on GPU when beneficial.

    Returns a numpy array. If H_gpu is supplied (cached), the H upload is
    skipped — this is the main motivation for the helper, since the same H
    is read by both the BFGS eigh and the projection in the same step.
    """
    n = H.shape[0]
    if H_gpu is not None or _gpu_ok(n):
        try:
            Ht = H_gpu if H_gpu is not None else to_gpu(H)
            Ut = to_gpu(U)
            if Ht is not None and Ut is not None:
                # Project on device, download only the (k, k) result.
                R_t = Ut.T @ Ht @ Ut
                return R_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return U.T @ H @ U
