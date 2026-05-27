#!/usr/bin/env python

from __future__ import division

import numpy as np

from scipy.linalg import eigh, lstsq, solve

from sella import _gpu as _gpu_mod


def symmetrize_Y2(S, Y):
    _, nvecs = S.shape
    dY = np.zeros_like(Y)
    YTS = Y.T @ S
    dYTS = np.zeros_like(YTS)
    STS = S.T @ S
    for i in range(1, nvecs):
        RHS = np.linalg.lstsq(STS[:i, :i],
                              YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i],
                              rcond=None)[0]
        dY[:, i] = -S[:, :i] @ RHS
        dYTS[i, :] = -STS[:, :i] @ RHS
    return dY


def symmetrize_Y(S, Y, symm):
    if symm is None or S.shape[1] == 1:
        return Y
    elif symm == 0:
        return Y + S @ lstsq(S.T @ S, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
    elif symm == 1:
        return Y + Y @ lstsq(S.T @ Y, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
    elif symm == 2:
        return Y + symmetrize_Y2(S, Y)
    else:  # pragma: no cover
        raise ValueError("Unknown symmetrization method {}".format(symm))


def update_H(B, S, Y, method='TS-BFGS', symm=2, lams=None, vecs=None,
             B_gpu=None, evals_gpu=None, evecs_gpu=None):
    """Quasi-Newton update.

    Optional GPU-resident path: when B_gpu (torch CUDA tensor) is supplied,
    the TS-BFGS update runs on device and returns (Bplus_numpy, Bplus_gpu)
    so the caller can refresh its GPU cache without re-uploading. Falls
    back to numpy if GPU unavailable or for other update methods.
    """
    if len(S.shape) == 1:
        if np.linalg.norm(S) < 1e-8:
            return B
        S = S[:, np.newaxis]
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]

    Ytilde = symmetrize_Y(S, Y, symm)

    if B is None:
        # Approximate B as a scaled identity matrix, where the
        # scalar is the average Ritz value from S.T @ Y
        thetas, _ = eigh(S.T @ Ytilde)
        # Guard against zero eigenvalues which would give log(0) = -Inf
        thetas_abs = np.abs(thetas)
        thetas_abs = np.maximum(thetas_abs, 1e-12)
        lam0 = np.exp(np.average(np.log(thetas_abs)))
        d, _ = S.shape
        B = lam0 * np.eye(d)

    # GPU-resident TS-BFGS path: requires B_gpu and (evals_gpu, evecs_gpu).
    if (method == 'TS-BFGS' and B_gpu is not None
            and evals_gpu is not None and evecs_gpu is not None):
        result = _gpu_update_TS_BFGS(B_gpu, S, Ytilde, evals_gpu,
                                     evecs_gpu)
        if result is not None:
            return result  # (Bplus_numpy, Bplus_gpu)

    if lams is None or vecs is None:
        lams, vecs = eigh(B)

    if method == 'BFGS_auto':
        # Default to TS-BFGS, and only use BFGS if B and S.T @ Y are
        # both positive definite
        method = 'TS-BFGS'
        if lams is not None and np.all(lams > 0):
            lams_STY, vecs_STY = eigh(S.T @ Ytilde, S.T @ S)
            if np.all(lams_STY > 0):
                method = 'BFGS'

    if method == 'BFGS':
        Bplus = _MS_BFGS(B, S, Ytilde)
    elif method == 'TS-BFGS':
        Bplus = _MS_TS_BFGS(B, S, Ytilde, lams, vecs)
    elif method == 'PSB':
        Bplus = _MS_PSB(B, S, Ytilde)
    elif method == 'DFP':
        Bplus = _MS_DFP(B, S, Ytilde)
    elif method == 'SR1':
        Bplus = _MS_SR1(B, S, Ytilde)
    elif method == 'Greenstadt':
        Bplus = _MS_Greenstadt(B, S, Ytilde)
    else:  # pragma: no cover
        raise ValueError('Unknown update method {}'.format(method))

    Bplus += B
    # Symmetrize to clean up floating-point roundoff. The MS_* updates above
    # are mathematically symmetric, so any asymmetry is at machine precision;
    # (B + B.T) / 2 is faster than the tril-based approach and gives the same
    # result up to ~1e-16.
    Bplus = (Bplus + Bplus.T) * 0.5

    return Bplus


def _MS_BFGS(B, S, Y):
    return Y @ solve(Y.T @ S, Y.T) - B @ S @ solve(S.T @ B @ S, S.T @ B)


def _MS_TS_BFGS(B, S, Y, lams, vecs):
    J = Y - B @ S
    X1 = S.T @ Y @ Y.T
    absBS = vecs @ (np.abs(lams[:, np.newaxis]) * (vecs.T @ S))
    X2 = S.T @ absBS @ absBS.T
    U = lstsq((X1 + X2) @ S, X1 + X2)[0].T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T


def _MS_PSB(B, S, Y):
    J = Y - B @ S
    U = solve(S.T @ S, S.T).T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T


def _MS_DFP(B, S, Y):
    J = Y - B @ S
    U = solve(S.T @ Y, Y.T).T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T


def _MS_SR1(B, S, Y):
    YBS = Y - B @ S
    return YBS @ solve(YBS.T @ S, YBS.T)


def _MS_Greenstadt(B, S, Y):
    J = Y - B @ S
    MS = B @ S
    U = solve(S.T @ MS, MS.T).T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T


# Not a symmetric update, so not available my default
def _MS_Powell(B, S, Y):  # pragma: no cover
    return (Y - B @ S) @ S.T


def _gpu_update_TS_BFGS(B_gpu, S, Y, evals_gpu, evecs_gpu):
    """GPU-resident TS-BFGS update + symmetrize, returning numpy + torch.

    Mirrors `_MS_TS_BFGS` but does the heavy matmuls and lstsq on device,
    so the (N,N) eigvecs never need to .cpu()-bounce. Returns
    (Bplus_numpy, Bplus_gpu) on success, or None to signal the caller to
    fall back to the numpy path.
    """
    torch = _gpu_mod.torch
    if torch is None:
        return None
    try:
        # S, Y are small (N, k) with k typically 1. Upload once.
        S_t = torch.from_numpy(np.ascontiguousarray(S)).cuda()
        Y_t = torch.from_numpy(np.ascontiguousarray(Y)).cuda()

        J_t = Y_t - B_gpu @ S_t
        X1_t = S_t.T @ Y_t @ Y_t.T  # (k, N)
        # |B| @ S = vecs @ (|lams| * (vecs.T @ S))
        absBS_t = evecs_gpu @ (
            evals_gpu.abs().unsqueeze(1) * (evecs_gpu.T @ S_t)
        )
        X2_t = S_t.T @ absBS_t @ absBS_t.T  # (k, N)
        XS_t = X1_t + X2_t  # (k, N)

        # Solve (XS @ S) U.T = XS  →  U = ((XS @ S)^{-1} @ XS).T
        # XS_S is (k, k) and tiny; use solve.
        XS_S_t = XS_t @ S_t  # (k, k)
        # Use lstsq for parity with numpy path (handles k=1 trivially).
        U_t = torch.linalg.lstsq(XS_S_t, XS_t).solution.T

        UJT_t = U_t @ J_t.T
        delta_t = (UJT_t + UJT_t.T) - U_t @ (J_t.T @ S_t) @ U_t.T

        Bplus_t = B_gpu + delta_t
        # Symmetrize on device.
        Bplus_t = 0.5 * (Bplus_t + Bplus_t.T)

        # Single download of the new B.
        Bplus = Bplus_t.cpu().numpy()
        return Bplus, Bplus_t
    except (RuntimeError, MemoryError):
        _gpu_mod._record_oom(B_gpu.shape[0])
        return None
