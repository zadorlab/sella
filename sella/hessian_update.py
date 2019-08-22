#!/usr/bin/env python

from __future__ import division

import numpy as np

from scipy.linalg import eigh, lstsq, solve

from .cython_routines import symmetrize_Y2


def symmetrize_Y(S, Y, symm):
    if symm is None or S.shape[1] == 1:
        return Y
    elif symm == 0:
        return Y + S @ lstsq(S.T @ S, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
    elif symm == 1:
        return Y + Y @ lstsq(S.T @ Y, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
    elif symm == 2:
        return Y + symmetrize_Y2(S, Y)
    else:
        raise ValueError("Unknown symmetrization method {}".format(symm))


def update_H(B, S, Y, method='TS-BFGS', symm=2, lams=None, vecs=None):
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
        thetas, _ = eigh(S.T @ Ytilde, S.T @ S)
        lam0 = np.average(np.abs(thetas))
        d, _ = S.shape
        B = lam0 * np.eye(d)

    if lams is None or vecs is None:
        lams, vecs = eigh(B)

    if method == 'BFGS_auto':
        # Default to TS-BFGS, and only use BFGS if B and S.T @ Y are
        # both positive definite
        method = 'TS-BFGS'
        if np.all(lams > 0):
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
    elif method == 'TS-DFP':
        Bplus = _MS_TS_DFP(B, S, Ytilde, lams, vecs)
    elif method == 'SR1':
        Bplus = _MS_SR1(B, S, Ytilde)
    elif method == 'Greenstadt':
        Bplus = _MS_Greenstadt(B, S, Ytilde)
    elif method == 'TS-Greenstadt':
        Bplus = _MS_TS_Greenstadt(B, S, Ytilde)
    elif method == 'test':
        Bplus = _MS_test(B, S, Ytilde, lams, vecs)
    else:
        raise ValueError('Unknown update method {}'.format(method))

    Bplus += B
    Bplus -= np.tril(Bplus.T - Bplus, -1).T

    return Bplus


def _MS_BFGS(B, S, Y):
    return Y @ solve(Y.T @ S, Y.T) - B @ S @ solve(S.T @ B @ S, S.T @ B)


def _MS_TS_BFGS(B, S, Y, lams, vecs):
    J = Y - B @ S
    X1 = S.T @ Y @ Y.T
    absBS = vecs @ (np.abs(lams[:, np.newaxis]) * (vecs.T @ S))
    X2 = S.T @ absBS @ absBS.T
    U = solve((X1 + X2) @ S, X1 + X2).T
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


def _MS_TS_DFP(B, S, Y, lams, vecs):
    J = Y - B @ S
    MS = Y @ Y.T @ S
    U = solve(S.T @ MS, MS.T).T
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


def _MS_TS_Greenstadt(B, S, Y):
    J = Y - B @ S
    lams, vecs = eigh(B)
    MS = vecs @ (np.abs(lams)[:, np.newaxis] * vecs.T @ S)
    U = solve(S.T @ MS, MS.T).T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T


def _MS_test(B, S, Y, lams, vecs):
    Binv = np.linalg.inv(B)
    Binvlams, _ = eigh(Binv)
    J = Binv @ Y - S
    MY = S
    Ylams, _ = eigh(S.T @ Y, S.T @ S)
    shift = min(0, sorted(1 / lams)[0])
    MY -= 2.0 * shift * Y
    U = solve(Y.T @ MY, MY.T).T
    UJT = U @ J.T
    Binvplus = Binv - (UJT + UJT.T) + U @ (J.T @ Y) @ U.T
    return np.linalg.inv(Binvplus) - B


# Not a symmetric update, so not available my default
def _MS_Powell(B, S, Y):
    return (Y - B @ S) @ S.T
