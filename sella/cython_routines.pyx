# cython: language_level=3

cimport cython
from cython.view cimport array as cvarray

from scipy.linalg.cython_blas cimport ddot, dgemv, dnrm2, dcopy, daxpy, dscal
from scipy.linalg.cython_lapack cimport dgels, dgesvd
from scipy.linalg import null_space

from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cimport numpy as np

import numpy as np

cdef extern from "math.h":
    double INFINITY

cdef int ZERO = 0
cdef int ONE = 1
cdef int NONE = -1

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
def ortho(X, Y=None, M=None, double eps=1e-15):
    if M is None:
        return simple_ortho(X, Y, eps)

    if len(X.shape) == 1:
        X_local = np.array(X[:, np.newaxis], order='C')
    else:
        X_local = np.array(X, order='C')

    cdef int n
    cdef int nx
    cdef int ny

    n, nx = X_local.shape
    if Y is None:
        Y_local = np.empty((n, 0), dtype=X.dtype, order='C')
    else:
        Y_local = np.array(Y, order='C')

    _, ny = Y_local.shape
    if nx + ny > n:
        ny = n - nx
        #Y_local = np.empty((n, ny), order='C')
        Y_local = Y_local[:, :ny]

    if ny == n:
        return np.empty((0, n))

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
def modified_gram_schmidt(X_np, double eps=1e-15):
    X_local = X_np.copy()

    cdef double[:, :] X = memoryview(X_local)

    cdef int d = X.shape[0]
    cdef int n = X.shape[1]
    cdef int i
    cdef int j
    cdef double scale

    cdef int sd = X.strides[0] // 8
    
    with nogil:
        for i in range(n):
            while True:
                for j in range(i - 1):
                    scale = -ddot(&d, &X[0, j], &sd, &X[0, i], &sd)
                    daxpy(&d, &scale, &X[0, j], &sd, &X[0, i], &sd)
                scale = dnrm2(&d, &X[0, i], &sd)
                for j in range(d):
                    X[j, i] /= scale
                if fabs(1 - scale) < eps:
                    break

    return X_local

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simple_ortho(X, Y=None, double eps=1e-15):
    # A lot of code overlap with ortho. Can these be rolled into a
    # single function without using a lot of conditionals?
    if len(X.shape) == 1:
        X_local = np.array(X[:, np.newaxis], order='C')
    else:
        X_local = np.array(X, order='C')

    cdef int n
    cdef int nx
    cdef int ny

    cdef int i
    cdef int j
    cdef int k
    cdef int nxout
    cdef int nyout

    n, nx = X_local.shape

    if Y is None:
        Y_local = np.empty((n, 0), dtype=X.dtype, order='C')
    else:
        Y_local = np.array(Y, order='C')
    _, ny = Y_local.shape

    nxout = nx
    nyout = ny

    cdef double[:, :] X_mv = memoryview(X_local)
    cdef double[:, :] Y_mv = memoryview(Y_local)
    cdef double scale
    cdef double eps2 = sqrt(eps)

    # Initialize SVD
    # Note: all of my matrices are C-ordered (row-major) for consistency,
    # but this means the Fortran LAPACK routines see the matrices as their
    # transpose. So, this code asks DGESVD for the right singular vectors,
    # even though what we really want is the left singular vectors.
    #
    # It would not be unreasonable to make the arrays Fortran-ordered
    # (column-major) to make the DGESVD calls clearer.

    cdef int lwork = -1
    cdef int info
    cdef double* S = <double*> malloc(sizeof(double) * max(nx, ny))
    cdef double* work = <double*> malloc(sizeof(double) * 1)
    cdef double U
    cdef double VT
    if nx >= ny:
        dgesvd('N', 'O', &nx, &n, &X_mv[0, 0], &nx, S, &U, &ONE, &VT, &ONE, work, &lwork, &info)
    else:
        dgesvd('N', 'O', &ny, &n, &Y_mv[0, 0], &ny, S, &U, &ONE, &VT, &ONE, work, &lwork, &info)
    lwork = int(work[0])
    free(work)
    work = <double*> malloc(sizeof(double) * lwork)

    if ny > 1:
        dgesvd('N', 'O', &ny, &n, &Y_mv[0, 0], &ny, S, &U, &ONE, &VT, &ONE, work, &lwork, &info)
        for i in range(ny):
            if abs(S[i]) < eps2:
                nyout -= 1

    if nx + nyout >= n:
        free(S)
        free(work)
        if nyout == 0:
            return np.eye(n)
        else:
            return null_space(Y_local.T)

    cdef double err

    while True:
        # Modified Gram-Schmidt to orthogonalize X against Y,
        # renormalizing the columns of X after every iteration
        for i in range(nxout):
            for j in range(nyout):
                scale = -ddot(&n, &Y_mv[0, j], &ny, &X_mv[0, i], &nx)
                daxpy(&n, &scale, &Y_mv[0, j], &ny, &X_mv[0, i], &nx)
                scale = dnrm2(&n, &X_mv[0, i], &nx)
                for k in range(n):
                    X_mv[k, i] /= scale
        # Use SVD to make X orthonormal
        dgesvd('N', 'O', &nxout, &n, &X_mv[0, 0], &nx, S, &U, &ONE, &VT, &ONE, work, &lwork, &info)
        # Singular values are ranked greatest to least; truncate small singular
        # values, because that means there is a high degree of linear dependence
        for i in range(nxout):
            if abs(S[i]) < eps2:
                nxout -= 1
        # Ensure columns of X are orthogonal to the columns of Y.
        # This is a bit redundant with the first step of the next iteration.
        for i in range(nxout):
            for j in range(nyout):
                err = ddot(&n, &Y_mv[0, j], &ny, &X_mv[0, i], &nx)
                if err > eps:
                    break
            if err > eps:
                break
        else:
            free(S)
            free(work)
            return X_local[:, :nxout]

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

cdef double ahbisect(double[:] d, double[:] z, double alpha, double lower, double upper, double middle, double eps=1e-15) nogil:
    cdef int n = len(d)

    if (upper - middle)**2 < eps or (middle - lower)**2 < eps:
        return middle

    cdef double f, df, mid1, dmid1

    cdef int k = 0

    cdef int i

    while k < 100:
        k += 1
        f = 0.
        df = 0.
        for i in range(n):
            f -= z[i]**2 / (d[i] - middle)
            df -= z[i]**2 / (d[i] - middle)**2
        f += alpha - middle
        df -= 1

        if fabs(f) < eps or (lower > -INFINITY and upper < INFINITY and (upper - lower)**2 < eps):
            return middle

        if f < 0:
            upper = middle
        else:
            lower = middle

        mid1 = middle - f / df
        if mid1 - lower < eps or upper - mid1 < eps:
            middle = (upper + lower) / 2.
        else:
            middle = mid1
    return middle

def aheig(np.ndarray[np.float64_t, ndim=1] d_np, np.ndarray[np.float64_t, ndim=1] z_np, double alpha):
    cdef double[:] d = memoryview(d_np)
    cdef double[:] z = memoryview(z_np)

    cdef int n = len(d) + 1

    lams_np = np.zeros(n)
    vecs_np = np.zeros((n, n))

    cdef double[:] lams = memoryview(lams_np)
    cdef double[:, :] vecs = memoryview(vecs_np)

    lower_np = np.zeros(n)
    upper_np = np.zeros(n)
    middle_np = np.zeros(n)

    cdef double[:] lower = memoryview(lower_np)
    cdef double[:] upper = memoryview(upper_np)
    cdef double[:] middle = memoryview(middle_np)

    cdef double avgdiff = (d[n - 2] - d[0]) / (n - 1)
    if abs(avgdiff) < 1e-15:
        avgdiff = 1e-2
    cdef double denom

    cdef size_t i, j, k

    lower[0] = -INFINITY
    upper[0] = d[0]
    middle[0] = d[0] - avgdiff

    for i in range(1, n - 1):
        lower[i] = d[i - 1]
        upper[i] = d[i]
        middle[i] = (d[i - 1] + d[i]) / 2.

    lower[n - 1] = d[n - 2]
    upper[n - 1] = INFINITY
    middle[n - 1] = d[n - 2] + avgdiff

    for i in range(n):
        lams[i] = ahbisect(d, z, alpha, lower[i], upper[i], middle[i])
        for j in range(n - 2):
            if (d[j] - lams[i])**2 < 1e-15:
                denom = 0.
                for k in range(j, i):
                    vecs[k, i] = z[k] * z[i]
                    denom += z[k] * z[k]
                for k in range(j, i):
                    vecs[k, i] /= denom
                vecs[i, i] = -1.
                vecs_np[:, i] /= np.linalg.norm(vecs_np[:, i])
                break
        else:
            for j in range(n - 1):
                vecs[j, i] = z[j] / (d[j] - lams[i])
            vecs[-1, i] = -1.
            vecs_np[:, i] /= np.linalg.norm(vecs_np[:, i])
    return lams_np, vecs_np
