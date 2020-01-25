from libc.math cimport INFINITY
from scipy.linalg.cython_blas cimport (dasum, daxpy, dcopy, ddot, dnrm2,
                                       dscal, dswap, dgemv, dgemm, dger)

cdef double my_dasum(double[:] x) nogil:
    """Returns sum(x)"""
    cdef int n = x.shape[0]
    cdef int sdx = x.strides[0] >> 3
    return dasum(&n, &x[0], &sdx)

cdef int my_daxpy(double scale, double[:] x, double[:] y) nogil except -1:
    """Evaluates y = y + scale * x"""
    cdef int n = len(x)
    if len(y) != n:
        return -1
    cdef int sdx = x.strides[0] >> 3
    cdef int sdy = y.strides[0] >> 3
    daxpy(&n, &scale, &x[0], &sdx, &y[0], &sdy)
    return 0

cdef int my_dcopy(double[:] x, double[:] y) nogil except -1:
    """Copies contents of x into y"""
    cdef int n = x.shape[0]
    if n != y.shape[0]:
        return -1
    cdef int sdx = x.strides[0] >> 3
    cdef int sdy = y.strides[0] >> 3
    dcopy(&n, &x[0], &sdx, &y[0], &sdy)
    return 0

cdef double my_ddot(double[:] x, double[:] y) nogil except INFINITY:
    """Calculators dot(x, y)"""
    cdef int n = x.shape[0]
    if n != y.shape[0]:
        return INFINITY
    cdef int sdx = x.strides[0] >> 3
    cdef int sdy = y.strides[0] >> 3
    return ddot(&n, &x[0], &sdx, &y[0], &sdy)

cdef double my_dnrm2(double[:] x) nogil:
    """Calculators 2-norm of x"""
    cdef int n = x.shape[0]
    cdef int sdx = x.strides[0] >> 3
    return dnrm2(&n, &x[0], &sdx)

cdef void my_dscal(double alpha, double[:] x) nogil:
    """Multiplies x by scalar alpha"""
    cdef int n = x.shape[0]
    cdef int sdx = x.strides[0] >> 3
    dscal(&n, &alpha, &x[0], &sdx)

cdef int my_dswap(double[:] x, double[:] y) nogil except -1:
    """Swaps contents of x and y"""
    cdef int n = x.shape[0]
    if n != y.shape[0]:
        return -1
    cdef int sdx = x.strides[0] >> 3
    cdef int sdy = y.strides[0] >> 3
    dswap(&n, &x[0], &sdx, &y[0], &sdy)
    return 0

cdef int my_dgemv(double[:, :] A, double[:] x, double[:] y,
                  double alpha=1., double beta=1.) nogil except -1:
    """Evaluates y = alpha * A @ x + beta * y"""
    cdef int ma, na, nx, my
    ma, na = A.shape[:2]
    nx = x.shape[0]
    my = y.shape[0]
    if ma != my:
        return -1
    if na != nx:
        return -1
    cdef int lda = A.strides[0] >> 3
    cdef int sda = A.strides[1] >> 3
    cdef int sdx = x.strides[0] >> 3
    cdef int sdy = y.strides[0] >> 3
    cdef char* trans = 'N'
    if lda > sda:
        lda, sda = sda, lda
        ma, na = na, ma
        trans = 'T'
    dgemv(trans, &ma, &na, &alpha, &A[0, 0], &sda, &x[0], &sdx, &beta, &y[0],
          &sdy)
    return 0

cdef int my_dgemm(double[:, :] A, double[:, :] B, double[:, :] C,
                  double alpha=1., double beta=1.) nogil except -1:
    """Evaluates C = alpha * A @ B + beta * C"""
    cdef int ma, na, mb, nb, mc, nc
    ma, na = A.shape[:2]
    mb, nb = B.shape[:2]
    mc, nc = C.shape[:2]
    if na != mb or ma != mc or nb != nc:
        return -1

    cdef int ldc = C.strides[0] >> 3
    cdef int sdc = C.strides[1] >> 3

    if ldc > sdc:
        return my_dgemm(B.T, A.T, C.T, alpha=alpha, beta=beta)

    cdef char* transa = 'N'
    cdef char* transb = 'N'

    cdef int lda = A.strides[0] >> 3
    cdef int sda = A.strides[1] >> 3
    cdef int ldb = B.strides[0] >> 3
    cdef int sdb = B.strides[1] >> 3

    if (lda > sda):
        transa = 'T'
        lda, sda = sda, lda

    if (ldb > sdb):
        transb = 'T'
        ldb, sdb = sdb, ldb

    dgemm(transa, transb, &mc, &nc, &na, &alpha, &A[0, 0], &sda, &B[0, 0],
          &sdb, &beta, &C[0, 0], &sdc)
    return 0

cdef int my_dger(double[:, :] A, double[:] x, double[:] y,
                 double alpha=1.) nogil except -1:
    """Evaluates A = A + alpha * outer(x, y)"""
    cdef int lda = A.strides[0] >> 3
    cdef int sda = A.strides[1] >> 3
    cdef int ma, na
    ma, na = A.shape[:2]
    cdef int mx = x.shape[0]
    cdef int ny = y.shape[0]
    if ma != mx or na != ny:
        return -1

    if lda > sda:
        lda, sda = sda, lda
        x, y = y, x
        ma, na = na, ma

    cdef int sdx = x.strides[0] >> 3
    cdef int sdy = y.strides[0] >> 3


    dger(&ma, &na, &alpha, &x[0], &sdx, &y[0], &sdy, &A[0, 0], &sda)
    return 0
