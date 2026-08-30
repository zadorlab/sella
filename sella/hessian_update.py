#!/usr/bin/env python

import logging
import numpy as np

from scipy.linalg import eigh, lstsq, solve

from sella import _gpu as _gpu_mod

logger = logging.getLogger(__name__)

from sella._constants import _LSTSQ_RCOND


def symmetrize_Y2(S, Y):
    _, nvecs = S.shape
    dY = np.zeros_like(Y)
    YTS = Y.T @ S
    dYTS = np.zeros_like(YTS)
    STS = S.T @ S
    for i in range(1, nvecs):
        RHS = np.linalg.lstsq(STS[:i, :i],
                              YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i],
                              rcond=_LSTSQ_RCOND)[0]
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


def _as_column_matrix(arr):
    if len(arr.shape) == 1:
        return arr[:, np.newaxis]
    return arr


def _download_gpu_hessian(B_gpu):
    try:
        return B_gpu.cpu().numpy()
    except (RuntimeError, MemoryError):
        _gpu_mod._record_oom(B_gpu.shape[0])
        raise


def _zero_step_update(B, S, B_gpu, download_numpy):
    if B_gpu is None:
        if B is None:
            return np.eye(S.shape[0], dtype=np.float64)
        return B
    if not download_numpy:
        return B, B_gpu
    if B is None:
        B = _download_gpu_hessian(B_gpu)
    return B, B_gpu


def _initial_hessian_from_secant(S, Y):
    # Approximate B as a scaled identity matrix, where the scalar is the
    # average Ritz value from S.T @ Y.
    thetas, _ = eigh(S.T @ Y)
    thetas_abs = np.maximum(np.abs(thetas), 1e-12)
    lam0 = np.exp(np.average(np.log(thetas_abs)))
    d, _ = S.shape
    return lam0 * np.eye(d)


def _can_gpu_ts_bfgs(method, B_gpu, evals_gpu, evecs_gpu):
    return (
        method == 'TS-BFGS' and B_gpu is not None
        and evals_gpu is not None and evecs_gpu is not None
    )


def _resolve_update_method(method, S, Y, lams):
    if method != 'BFGS_auto':
        return method

    # Default to TS-BFGS, and only use BFGS if B and S.T @ Y are both positive
    # definite.
    if lams is not None and np.all(lams > 0):
        lams_STY, _ = eigh(S.T @ Y, S.T @ S)
        if np.all(lams_STY > 0):
            return 'BFGS'
    return 'TS-BFGS'


def _update_delta(method, B, S, Y, lams, vecs):
    if method == 'TS-BFGS':
        return _MS_TS_BFGS(B, S, Y, lams, vecs)
    try:
        updater = _MS_UPDATE_METHODS[method]
    except KeyError:  # pragma: no cover
        raise ValueError('Unknown update method {}'.format(method))
    return updater(B, S, Y)


def update_H(B, S, Y, method='TS-BFGS', symm=2, lams=None, vecs=None,
             B_gpu=None, evals_gpu=None, evecs_gpu=None,
             download_numpy=True):
    """Quasi-Newton update.

    Optional GPU-resident path: when B_gpu (torch CUDA tensor) is supplied,
    the TS-BFGS update runs on device and returns (Bplus_numpy, Bplus_gpu)
    so the caller can refresh its GPU cache without re-uploading. Falls
    back to numpy if GPU unavailable or for other update methods.
    """
    if np.linalg.norm(S) < 1e-8:
        return _zero_step_update(B, S, B_gpu, download_numpy)

    S = _as_column_matrix(S)
    Y = _as_column_matrix(Y)
    Ytilde = symmetrize_Y(S, Y, symm)
    gpu_ts_bfgs = _can_gpu_ts_bfgs(method, B_gpu, evals_gpu, evecs_gpu)

    if B is None and not gpu_ts_bfgs:
        B = _initial_hessian_from_secant(S, Ytilde)

    # GPU-resident TS-BFGS path: requires B_gpu and (evals_gpu, evecs_gpu).
    if gpu_ts_bfgs:
        result = _gpu_update_TS_BFGS(B_gpu, S, Ytilde, evals_gpu,
                                     evecs_gpu,
                                     download_numpy=download_numpy)
        if result is not None:
            return result  # (Bplus_numpy, Bplus_gpu)
        if B is None:
            # Rare GPU fallback path. Download the current Hessian only when
            # needed so the normal GPU path can keep B device-resident.
            B = _download_gpu_hessian(B_gpu)

    if lams is None or vecs is None:
        lams, vecs = eigh(B)

    method = _resolve_update_method(method, S, Ytilde, lams)
    Bplus = _update_delta(method, B, S, Ytilde, lams, vecs)

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
    if S.shape[1] == 1:
        s = S[:, 0]
        y = Y[:, 0]
        J = y - B @ s
        absBS = vecs @ (np.abs(lams) * (vecs.T @ s))
        X = (s @ y) * y + (s @ absBS) * absBS
        XS = X @ s
        if XS == 0.0:
            logger.debug("TS-BFGS singular rank-one update, falling back to PSB")
            return _MS_PSB(B, S, Y)
        U = X / XS
        JTS = J @ s
        return (np.outer(U, J) + np.outer(J, U)
                - JTS * np.outer(U, U))

    J = Y - B @ S
    X1 = S.T @ Y @ Y.T
    absBS = vecs @ (np.abs(lams[:, np.newaxis]) * (vecs.T @ S))
    X2 = S.T @ absBS @ absBS.T
    XS = (X1 + X2) @ S
    cond_XS = np.linalg.cond(XS)
    if cond_XS > 1e12:
        logger.debug("TS-BFGS ill-conditioned (cond=%.2e), falling back to PSB",
                     cond_XS)
        return _MS_PSB(B, S, Y)
    U = lstsq(XS, X1 + X2)[0].T
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


_MS_UPDATE_METHODS = {
    'BFGS': _MS_BFGS,
    'PSB': _MS_PSB,
    'DFP': _MS_DFP,
    'SR1': _MS_SR1,
    'Greenstadt': _MS_Greenstadt,
}


# Not a symmetric update, so not available my default
def _MS_Powell(B, S, Y):  # pragma: no cover
    return (Y - B @ S) @ S.T


def _gpu_update_TS_BFGS(B_gpu, S, Y, evals_gpu, evecs_gpu,
                        download_numpy=True):
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
        # Check conditioning on CPU (tiny matrix, negligible cost)
        if torch.linalg.cond(XS_S_t).item() > 1e12:
            return None  # fall back to CPU PSB path
        U_t = torch.linalg.lstsq(XS_S_t, XS_t).solution.T

        UJT_t = U_t @ J_t.T
        delta_t = (UJT_t + UJT_t.T) - U_t @ (J_t.T @ S_t) @ U_t.T

        Bplus_t = B_gpu + delta_t
        # Symmetrize on device.
        Bplus_t = 0.5 * (Bplus_t + Bplus_t.T)

        # Download only when the caller needs an immediate CPU copy.  The
        # optimizer's hot path can use Bplus_t for projection, eigensolve, and
        # Hessian-vector products, and materialize numpy lazily if requested.
        Bplus = Bplus_t.cpu().numpy() if download_numpy else None
        return Bplus, Bplus_t
    except (RuntimeError, MemoryError):
        _gpu_mod._record_oom(B_gpu.shape[0])
        return None
