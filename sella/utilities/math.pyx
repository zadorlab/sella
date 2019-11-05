# cython: language_level=3

cimport cython
from cython.view cimport array as cva

from libc.math cimport sqrt, fabs
from scipy.linalg.cython_blas cimport daxpy, dnrm2, dcopy, dgemv, ddot

import numpy as np

cdef double DUNITY = 1.
cdef double DZERO = 0.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int my_daxpy(double scale, double[:] x, double[:] y) nogil:
    """A wrapper for daxpy with size checks and automatic determination
    of stride size"""
    cdef int n = len(x)
    if len(y) != n:
        return -1
    cdef int sdx = x.strides[0] >> 3
    cdef int sdy = y.strides[0] >> 3
    daxpy(&n, &scale, &x[0], &sdx, &y[0], &sdy)
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int normalize(double[:] x) nogil:
    """Normalizes a vector in place"""
    cdef int n = len(x)
    cdef int sdx = x.strides[0] >> 3
    cdef double norm = dnrm2(&n, &x[0], &sdx)
    cdef int i
    for i in range(n):
        x[i] /= norm
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int vec_sum(double[:] x, double[:] y, double[:] z,
                        double scale=1.) nogil:
    """Evaluates z[:] = x[:] + scale * y[:]"""
    cdef int n = len(x)
    if len(y) != n:
        return -1
    if len(z) != n:
        return -1
    cdef int sdx = x.strides[0] >> 3
    cdef int sdy = y.strides[0] >> 3
    cdef int sdz = z.strides[0] >> 3
    dcopy(&n, &x[0], &sdx, &z[0], &sdz)
    daxpy(&n, &scale, &y[0], &sdy, &z[0], &sdz)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void cross(double[:] x, double[:] y, double[:] z) nogil:
    """Evaluates z[:] = x[:] cross y[:]"""
    z[0] = x[1] * y[2] - y[1] * x[2]
    z[1] = x[2] * y[0] - y[2] * x[0]
    z[2] = x[0] * y[1] - y[0] * x[1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void symmetrize(double* X, size_t n, size_t lda) nogil:
    """Symmetrizes matrix X by populating the lower triangle with the
    contents of the upper triangle"""
    cdef size_t i, j
    for i in range(n - 1):
        for j in range(i + 1, n):
            X[j * lda + i] = X[i * lda + j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void skew(double[:] x, double[:, :] Y, double scale=1.) nogil:
    """Fillx matrix Y with the elements of vectors scale * x such that
    Y becomes a skew-symmetric matrix"""
    # Safer than memset, because this might be a slice of a larger array
    cdef int i, j
    for i in range(3):
        for j in range(3):
            Y[i, j] = 0.

    Y[2, 1] = scale * x[0]
    Y[0, 2] = scale * x[1]
    Y[1, 0] = scale * x[2]

    Y[1, 2] = -Y[2, 1]
    Y[2, 0] = -Y[0, 2]
    Y[0, 1] = -Y[1, 0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef (int, double) inner(double[:, :] M, double[:] x, double[:] y,
                         double[:] Mx) nogil:
    """Calculates the inner product y.T @ M @ x"""
    cdef int n = len(x)
    cdef int m = len(y)
    if M.shape[0] != m:
        return (-1, 0.)
    if M.shape[1] != n:
        return (-1, 0.)
    if len(Mx) != m:
        return (-1, 0.)
    cdef int sdx = x.strides[0] >> 3
    cdef int sdM = M.strides[1] >> 3
    cdef int ldM = n * sdM
    cdef int sdy = y.strides[0] >> 3
    cdef int sdMx = Mx.strides[0] >> 3
    dgemv('N', &n, &m, &DUNITY, &M[0, 0], &ldM, &x[0], &sdx, &DZERO,
          &Mx[0], &sdMx)
    return (0, ddot(&n, &y[0], &sdy, &Mx[0], &sdMx))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int mgs(double[:, :] X, double[:, :] Y=None, double eps1=1e-15,
             double eps2=1e-6, int maxiter=100) nogil:
    """Orthonormalizes X in-place against itself and Y. To accomplish this, Y
    is first orthonormalized in-place."""

    cdef int n = X.shape[0]
    cdef int nx = X.shape[1]
    cdef int sdx = X.strides[0] >> 3

    cdef int ny
    cdef int sdy
    if Y is None:
        ny = 0
        sdy = -1
    else:
        if Y.shape[0] != n:
            return -1
        ny = Y.shape[1]
        sdy = Y.strides[0] >> 3

    if ny != 0:
        ny = mgs(Y, None, eps1=eps1, eps2=eps2, maxiter=maxiter)
        if ny < 0:
            return ny

    cdef double norm, normtot, dot

    # Now orthonormalize X
    cdef int m = 0
    cdef int niter
    for i in range(nx):
        if i != m:
            X[:, m] = X[:, i]
            #dcopy(&n, &X[0, i], &sdx, &X[0, m], &sdx)
        norm = dnrm2(&n, &X[0, m], &sdx)
        for k in range(n):
            X[k, m] /= norm
        for niter in range(maxiter):
            normtot = 1.
            for j in range(ny):
                dot = -ddot(&n, &Y[0, j], &sdy, &X[0, m], &sdx)
                daxpy(&n, &dot, &Y[0, j], &sdy, &X[0, m], &sdx)
                norm = dnrm2(&n, &X[0, m], &sdx)
                normtot *= norm
                if normtot < eps2:
                    break
                for k in range(n):
                    X[k, m] /= norm
            if normtot < eps2:
                break
            for j in range(m):
                dot = -ddot(&n, &X[0, j], &sdx, &X[0, m], &sdx)
                daxpy(&n, &dot, &X[0, j], &sdx, &X[0, m], &sdx)
                norm = dnrm2(&n, &X[0, m], &sdx)
                normtot *= norm
                if normtot < eps2:
                    break
                for k in range(n):
                    X[k, m] /= norm
            if normtot < eps2:
                break
            elif 0. <= 1. - normtot <= eps1:
                m += 1
                break
        else:
            return -1

    # Just for good measure, zero out any leftover bits of X
    for i in range(m, nx):
        for k in range(n):
            X[k, i] = 0.

    return m

def modified_gram_schmidt(Xin, Yin=None, eps1=1.e-15, eps2=1.e-6,
                          maxiter=100):
    Xout_np = Xin.copy()
    cdef double[:, :] Xout = memoryview(Xout_np)

    if Yin is None:
        nx = mgs(Xout, None, eps1=eps1, eps2=eps2, maxiter=maxiter)
        if nx < 0:
            raise RuntimeError("MGS failed.")
        return Xout_np[:, :nx]

    Y_np = Yin.copy()
    cdef double[:, :] Y = memoryview(Y_np)

    nx = mgs(Xout, Y, eps1=eps1, eps2=eps2, maxiter=maxiter)
    if nx < 0:
        raise RuntimeError("MGS failed: Mismatched matrix sizes!")
    return Xout_np[:, :nx]