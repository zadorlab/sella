from libc.math cimport sqrt, fabs, INFINITY
from libc.string cimport memset
from scipy.linalg.cython_blas cimport daxpy, dnrm2, dcopy, dgemv, ddot, dger
from scipy.linalg.cython_lapack cimport dgesvd

import numpy as np

cdef int UNITY = 1

cdef double DUNITY = 1.
cdef double DZERO = 0.


cdef inline int normalize(double[:] x) nogil:
    """Normalizes a vector in place"""
    cdef int n = len(x)
    cdef int sdx = x.strides[0] >> 3
    cdef double norm = dnrm2(&n, &x[0], &sdx)
    cdef int i
    for i in range(n):
        x[i] /= norm
    return 0


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


cdef inline void cross(double[:] x, double[:] y, double[:] z) nogil:
    """Evaluates z[:] = x[:] cross y[:]"""
    z[0] = x[1] * y[2] - y[1] * x[2]
    z[1] = x[2] * y[0] - y[2] * x[0]
    z[2] = x[0] * y[1] - y[0] * x[1]


cdef inline void symmetrize(double* X, size_t n, size_t lda) nogil:
    """Symmetrizes matrix X by populating the lower triangle with the
    contents of the upper triangle"""
    cdef size_t i, j
    for i in range(max(0, n - 1)):
        for j in range(i + 1, n):
            X[j * lda + i] = X[i * lda + j]


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

    cdef double norm, normtot, dot

    # Now orthonormalize X
    cdef int m = 0
    cdef int niter
    for i in range(nx):
        if i != m:
            X[:, m] = X[:, i]
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
            return -2

    # Just for good measure, zero out any leftover bits of X
    for i in range(m, nx):
        for k in range(n):
            X[k, i] = 0.

    return m


def modified_gram_schmidt(Xin, Yin=None, eps1=1.e-15, eps2=1.e-6,
                          maxiter=100):
    if Xin.shape[1] == 0:
        return Xin

    if Yin is not None:
        Yout = Yin.copy()
        ny = mgs(Yout, None, eps1=eps1, eps2=eps2, maxiter=maxiter)
        Yout = Yout[:, :ny]
    else:
        Yout = None

    Xout = Xin.copy()
    nx = mgs(Xout, Yout, eps1=eps1, eps2=eps2, maxiter=maxiter)
    if nx < 0:
        raise RuntimeError("MGS failed.")
    return Xout[:, :nx]


cdef int mppi(int n, int m, double[:, :] A, double[:, :] U, double[:, :] VT,
              double[:] s, double[:, :] Ainv, double[:] work,
              double eps=1e-6) nogil:
    """Computes the Moore-Penrose pseudoinverse of A and stores the result in
    Ainv. This is done using singular value decomposition. Additionally, saves
    the right singular values in VT. A is then populated with the columns of
    VT that correspond to the null space of the singular vectors."""
    if A.shape[0] < n:
        return -1
    if A.shape[1] < m:
        return -1

    cdef int minnm = min(n, m)
    cdef int lda = A.strides[0] >> 3
    cdef int ldu = U.strides[0] >> 3
    cdef int ldvt = VT.strides[0] >> 3

    cdef int ns = s.shape[0]
    if ns < minnm:
        return -1

    cdef int lwork = work.shape[0]
    cdef int info

    dgesvd('A', 'S', &m, &n, &A[0, 0], &lda, &s[0], &VT[0, 0], &ldvt,
           &U[0, 0], &ldu, &work[0], &lwork, &info)
    if info != 0:
        return -1

    memset(&Ainv[0, 0], 0, Ainv.shape[0] * Ainv.shape[1] * sizeof(double))

    cdef int i
    cdef double sinv
    cdef int incvt = VT.strides[1] >> 3
    cdef int ldainv = Ainv.strides[0] >> 3
    cdef int nsing = 0

    # Evaluate the pseudo-inverse
    for i in range(minnm):
        if fabs(s[i]) < eps:
            continue
        nsing += 1
        sinv = 1. / s[i]
        dger(&n, &m, &sinv, &U[0, i], &ldu, &VT[i, 0], &incvt,
             &Ainv[0, 0], &ldainv)

    # Populate the basis matrices
    cdef int inca = A.strides[1] >> 3
    for i in range(m):
        dcopy(&m, &VT[i, 0], &incvt, &A[0, i], &lda)

    for i in range(m - nsing):
        dcopy(&m, &A[0, nsing + i], &lda, &VT[0, i], &ldvt)

    return nsing


def pseudo_inverse(double[:, :] A, double eps=1e-6):
    cdef int n, m, minnm, maxnm
    n, m = A.shape[:2]
    minnm = min(n, m)
    maxnm = max(n, m)

    U = np.zeros((n, n), dtype=np.float64)
    VT = np.zeros((m, m), dtype=np.float64)
    s = np.zeros(min(n, m), dtype=np.float64)
    Ainv = np.zeros((m, n), dtype=np.float64)
    work = np.zeros(2 * max(3 * minnm + maxnm, 5 * minnm, 1))

    nsing = mppi(n, m, A, U, VT, s, Ainv, work, eps=eps)

    if nsing == -1:
        raise RuntimeError("mmpi failed!")

    return U, s, VT, Ainv, nsing
