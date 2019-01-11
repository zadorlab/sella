cimport cython
from cython.view cimport array as cvarray

from scipy.linalg.cython_blas cimport ddot, dgemv, dnrm2, dcopy, daxpy, dscal
from scipy.linalg.cython_lapack cimport dgels

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cimport numpy as np

import numpy as np

cdef int ZERO = 0
cdef int ONE = 1

cdef double DZERO = 0.
cdef double DONE = 1.
cdef double DNONE = -1.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double inner(double* M, double* x, double* y, double* Mx, int n, int xinc, int yinc, int Mxinc):
    dgemv('N', &n, &n, &DONE, M, &n, x, &xinc, &DZERO, Mx, &Mxinc)
    return ddot(&n, y, &yinc, Mx, &Mxinc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ortho(X, Y, M=None, double eps=1e-15):
    # Original Python code, in full:
    # X = X.copy()
    # Y = Y.copy()
    #
    # if len(X.shape) == 1:
    #     X = np.array([X]).T
    #
    # n, nx = X.shape
    # m, ny = Y.shape
    # assert n == m
    #
    # if nx + ny > n:
    #     raise RuntimeError
    #
    # if ny == m:
    #     return np.empty((0, m))
    #
    # if M is None:
    #     M = np.eye(n)
    #
    # X /= np.linalg.norm(X, axis=0)[np.newaxis, :]
    #
    # MY = M @ Y
    # MX = M @ X
    # if np.iscomplexobj(MY) or np.iscomplexobj(MX):
    #     X = np.array(X, dtype=complex)
    #
    # Xout = np.zeros_like(X)
    # nout = 0
    #
    # YTMY = Y.T @ MY
    #
    # for i in range(nx):
    #     X[:, i] /= X[:, i].T @ MX[:, i]
    #     while True:
    #         for j in range(ny):
    #             MXi = M @ X[:, i]
    #             X[:, i] -= Y[:, j] * (Y[:, j].conj().T @ MXi) / np.sqrt(YTMY[j, j] * (X.T)[i] @ MXi)
    #         for j in range(nout):
    #             MXi = M @ X[:, i]
    #             X[:, i] -= Xout[:, j] * (Xout[:, j].conj().T @ MXi) / np.sqrt((Xout.T)[j] @ M @ Xout[:, j] * (X.T)[i] @ MXi)
    #         xinorm = np.sqrt(X[:, i].conj().T @ M @ X[:, i])
    #         X[:, i] /= xinorm
    #         MX[:, i] = M @ X[:, i]
    #         if abs(1 - xinorm.real) < 1e-15 and abs(xinorm.imag) < 1e-15:
    #             Xout[:, nout] = X[:, i]
    #             nout += 1
    #             break
    #         elif abs(xinorm) < eps:
    #             break
    #     else:
    #         raise RuntimeError
    # return Xout[:, :nout]

    if len(X.shape) == 1:
        X_local = np.array(X[:, np.newaxis], order='C')
    else:
        X_local = np.array(X, order='C')
    Y_local = np.array(Y, order='C')

    cdef int n
    cdef int nx
    cdef int ny

    n, nx = X_local.shape
    _, ny = Y_local.shape
    if nx + ny > n:
        ny = n - nx
        #Y_local = np.empty((n, ny), order='C')
        Y_local = Y_local[:, :ny]

    if ny == n:
        return np.empty((0, n))

    if M is None:
        return simple_ortho(X, Y, eps)

    MX = M @ X_local
    MY = M @ Y_local
    MXi = np.zeros(n)
    YTMY = np.sqrt(np.diag(Y_local.T @ MY))
    Xout = np.zeros_like(X_local)
    MXout = np.zeros_like(MX)
    XoutTMXout = np.diag(np.zeros((nx, nx))).copy()

    cdef double[:, :] X_mv = memoryview(X_local)
    cdef double[:, :] Y_mv = memoryview(Y_local)
    cdef double[:, :] M_mv = memoryview(M)
    cdef double[:, :] MX_mv = memoryview(MX)
    cdef double[:, :] MY_mv = memoryview(MY)
    cdef double[:] MXi_mv = memoryview(MXi)
    cdef double[:] YTMY_mv = memoryview(YTMY)
    cdef double[:, :] Xout_mv = memoryview(Xout)
    cdef double[:, :] MXout_mv = memoryview(MXout)
    cdef double[:] XoutTMXout_mv = memoryview(XoutTMXout)

    cdef int i
    cdef int j
    cdef int k
    cdef int nout = 0

    cdef double XiTMXi
    cdef double YjTMXi
    cdef double XoutjTMXi
    cdef double scale
    
    for i in range(nx):
        while True:
            for j in range(ny):
                XiTMXi = inner(&M_mv[0, 0], &X_mv[0, i], &X_mv[0, i], &MXi_mv[0], n, nx, nx, 1)
                YjTMXi = ddot(&n, &Y_mv[0, j], &ny, &MXi_mv[0], &ONE)
                scale = -YjTMXi / (YTMY_mv[j] * sqrt(XiTMXi))
                daxpy(&n, &scale, &Y_mv[0, j], &ny, &X_mv[0, i], &nx)
            for j in range(nout):
                XiTMXi = inner(&M_mv[0, 0], &X_mv[0, i], &X_mv[0, i], &MXi_mv[0], n, nx, nx, 1)
                XoutjTMXi = ddot(&n, &Xout_mv[0, j], &nx, &MXi_mv[0], &ONE)
                scale = -XoutjTMXi / (XoutTMXout_mv[j] * sqrt(XiTMXi))
                daxpy(&n, &scale, &Xout_mv[0, j], &nx, &X_mv[0, i], &nx)

            XiTMXi = sqrt(inner(&M_mv[0, 0], &X_mv[0, i], &X_mv[0, i], &MXi_mv[0], n, nx, nx, 1))
            scale = 1. / XiTMXi
            dscal(&n, &scale, &X_mv[0, i], &nx)
            if abs(1 - XiTMXi) < eps:
                dcopy(&n, &X_mv[0, i], &nx, &Xout_mv[0, nout], &nx)
                XoutTMXout_mv[nout] = sqrt(inner(&M_mv[0, 0], &Xout_mv[0, nout], &Xout_mv[0, nout], &MXout_mv[0, nout], n, nx, nx, 1))
                nout += 1
                break
            elif abs(XiTMXi) < eps:
                break

    return Xout[:, :nout]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simple_ortho(X, Y, double eps=1e-15):
    # A lot of code overlap with ortho. Can these be rolled into a
    # single function without using a lot of conditionals?
    if len(X.shape) == 1:
        X_local = np.array(X[:, np.newaxis], order='C')
    else:
        X_local = np.array(X, order='C')
    Y_local = np.array(Y, order='C')

    cdef int n
    cdef int nx
    cdef int ny

    n, nx = X_local.shape
    _, ny = Y_local.shape
    if nx + ny > n:
        ny = n - nx
        Y_local = Y_local[:, :ny]

    if ny == n:
        return np.empty((0, n))

    YTY = np.sqrt(np.diag(Y_local.T @ Y))
    Xout = np.zeros_like(X_local)
    XoutTXout = np.sqrt(np.diag(np.zeros((nx, nx))))

    cdef double[:, :] X_mv = memoryview(X_local)
    cdef double[:, :] Y_mv = memoryview(Y_local)
    cdef double[:] YTY_mv = memoryview(YTY)
    cdef double[:, :] Xout_mv = memoryview(Xout)
    cdef double[:] XoutTXout_mv = memoryview(XoutTXout)

    cdef int i
    cdef int j
    cdef int k
    cdef int nout

    cdef double XiTXi
    cdef double YjTXi
    cdef double XoutjTXi
    cdef double scale

    nout = 0
    
    for i in range(nx):
        while True:
            for j in range(ny):
                scale = 1. / dnrm2(&n, &X_mv[0, i], &nx)
                dscal(&n, &scale, &X_mv[0, i], &nx)
                YjTXi = ddot(&n, &Y_mv[0, j], &ny, &X_mv[0, i], &nx)
                scale = -YjTXi / YTY_mv[j]
                daxpy(&n, &scale, &Y_mv[0, j], &ny, &X_mv[0, i], &nx)
            for j in range(nout):
                scale = 1. / dnrm2(&n, &X_mv[0, i], &nx)
                dscal(&n, &scale, &X_mv[0, i], &nx)
                XoutjTXi = ddot(&n, &Xout_mv[0, j], &nx, &X_mv[0, i], &nx)
                scale = -XoutjTXi / XoutTXout_mv[j]
                daxpy(&n, &scale, &Xout_mv[0, j], &nx, &X_mv[0, i], &nx)
            XiTXi = dnrm2(&n, &X_mv[0, i], &nx)
            scale = 1. / XiTXi
            dscal(&n, &scale, &X_mv[0, i], &nx)
            if abs(1 - XiTXi) < eps:
                dcopy(&n, &X_mv[0, i], &nx, &Xout_mv[0, nout], &nx)
                XoutTXout_mv[nout] = XiTXi
                nout += 1
                break
            elif abs(XiTXi) < eps:
                break

    return Xout[:, :nout]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def symmetrize_Y2(S, Y):
    # Original Python code, in full:
    # _, nvecs = S.shape
    # dY = np.zeros_like(Y)
    # YTS = Y.T @ S
    # dYTS = np.zeros_like(YTS)
    # STS = S.T @ S
    # for i in range(1, nvecs):
    #     RHS = np.linalg.lstsq(STS[:i, :i], YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i], rcond=None)[0]
    #     dY[:, i] = -S[:, :i] @ RHS
    #     dYTS[i, :] = -STS[:, :i] @ RHS
    # return dY

    cdef int n
    cdef int nvecs
    cdef int nvecs2
    cdef int i
    cdef int j
    cdef int k
    
    cdef int lwork
    cdef int info


    S_local = np.array(S, order='C')
    Y_local = np.array(Y, order='C')

    n, nvecs = S_local.shape
    nvecs2 = nvecs * nvecs
    dY = np.zeros_like(Y_local, order='C')
    YTS = np.array(Y_local.T @ S_local, order='C')
    dYTS = np.zeros_like(YTS, order='C')
    STS = np.array(S_local.T @ S_local, order='C')
    RHS = np.zeros(nvecs, order='C')

    cdef double[:, :] S_mv = memoryview(S_local)
    cdef double[:, :] STS_mv = memoryview(STS)
    cdef double[:, :] dY_mv = memoryview(dY)
    cdef double[:, :] YTS_mv = memoryview(YTS)
    cdef double[:, :] dYTS_mv = memoryview(dYTS)
    cdef double[:] RHS_mv = memoryview(RHS)

    cdef double* dgels_A = <double*> malloc(sizeof(double) * nvecs * nvecs)

    cdef double* work_tmp = <double*> malloc(sizeof(double) * 1)
    lwork = -1
    dgels('N', &nvecs, &nvecs, &ONE, &STS_mv[0, 0], &nvecs, &RHS_mv[0], &nvecs, work_tmp, &lwork, &info)
    lwork = int(work_tmp[0])
    free(work_tmp)
    cdef double* work = <double*> malloc(sizeof(double) * lwork)

    # Original Python code, in full:

    for i in range(1, nvecs):
        # RHS = YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i]
        dcopy(&i, &YTS_mv[i, 0], &ONE, &RHS_mv[0], &ONE)
        daxpy(&i, &DNONE, &YTS_mv[0, i], &nvecs, &RHS_mv[0], &ONE)
        daxpy(&i, &DNONE, &dYTS_mv[0, i], &nvecs, &RHS_mv[0], &ONE)

        # copy contents of STS into separate array for DGELS
        dcopy(&nvecs2, &STS_mv[0, 0], &ONE, dgels_A, &ONE)

        # LHS = (STS[:i, :i])^-1 @ RHS
        dgels('N', &i, &i, &ONE, dgels_A, &nvecs, &RHS_mv[0], &i, work, &lwork, &info)

        # dY = S @ LHS
        dgemv('T', &i, &n, &DNONE, &S_mv[0, 0], &nvecs, &RHS_mv[0], &ONE, &DZERO, &dY_mv[0, i], &nvecs)
        
        # dYTS = dY.T @ S (or rather, dYTS.T = S.T @ dY)
        dgemv('N', &nvecs, &i, &DNONE, &STS_mv[0, 0], &nvecs, &RHS_mv[0], &ONE, &DZERO, &dYTS_mv[i, 0], &ONE)

    free(dgels_A)
    free(work)
    return dY

