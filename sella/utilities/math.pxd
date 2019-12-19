# cython: language_level=3

cdef int my_daxpy(double scale, double[:] x, double[:] y) nogil

cdef int normalize(double[:] x) nogil

cdef int vec_sum(double[:] x, double[:] y, double[:] z, double scale=?) nogil

cdef void cross(double[:] x, double[:] y, double[:] z) nogil

cdef void symmetrize(double* X, size_t n, size_t lda) nogil

cdef void skew(double[:] x, double[:, :] Y, double scale=?) nogil

cdef (int, double) inner(double[:, :] M, double[:] x, double[:] y,
                         double[:] Mx) nogil

cdef int mgs(double[:, :] X, double[:, :] Y=?, double eps1=?,
             double eps2=?, int maxiter=?) nogil

cdef int mppi(int n, int m, double[:, :] A, double[:, :] U, double[:, :] VT,
              double[:] s, double[:, :] Ainv, double[:] work,
              double eps=?) nogil
