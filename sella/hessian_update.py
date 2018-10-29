#!/usr/bin/env python

from __future__ import division

import numpy as np

from scipy.linalg import eigh, lstsq

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

def update_H(B, S, Y, method='TS-BFGS', symm=2):
    if len(S.shape) == 1:
        if np.linalg.norm(S) < 1e-8:
            return B
        S = S[:, np.newaxis]
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]

    Ytilde = symmetrize_Y(S, Y, symm)

    if method == 'BFGS':
        Bplus = B + _MS_BFGS(B, S, Ytilde)
    elif method == 'TS-BFGS':
        Bplus = B + _MS_TS_BFGS(B, S, Ytilde)
    elif method == 'PSB':
        Bplus = B + _MS_TS_BFGS(B, S, Ytilde)
    elif method == 'SR1':
        Bplus = B + _MS_SR1(B, S, Ytilde)
    else:
        raise ValueError('Unknown update method {}'.format(method))
    
    Bplus -= np.tril(Bplus.T - Bplus, -1).T

    return Bplus

def _MS_BFGS(B, S, Y):
    return Y @ lstsq(Y.T @ S, Y.T)[0] - B @ S @ lstsq(S.T @ B @ S, S.T @ B)[0]

def _MS_TS_BFGS(B, S, Y):
    lams_B, vecs_B = eigh(B)
    J = Y - B @ S
    X1 = S.T @ Y @ Y.T
    absBS = vecs_B @ (np.abs(lams_B[:, np.newaxis]) * (vecs_B.T @ S))
    X2 = S.T @ absBS @ absBS.T
    U = lstsq((X1 + X2) @ S, X1 + X2)[0].T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T

def _MS_PSB(B, S, Y):
    J = Y - B @ S
    U = lstsq(S.T @ S, S.T)[0].T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ J.T

def _MS_SR1(B, S, Y):
    YBS = Y - B @ S
    return YBS @ lstsq(YBS.T @ S, YBS.T)[0]

# Not a symmetric update, so not available my default
def _MS_Powell(B, S, Y):
    return (Y - B @ S) @ S.T
