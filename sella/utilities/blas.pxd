# cython: language_level=3

from libc.math cimport INFINITY

cdef double my_dasum(double[:] x) nogil

cdef int my_dcopy(double[:] x, double[:] y) nogil except -1

cdef double my_ddot(double[:] x, double[:] y) nogil except INFINITY

cdef double my_dnrm2(double[:] x) nogil

cdef void my_dscal(double alpha, double[:] x) nogil

cdef int my_dswap(double[:] x, double[:] y) nogil except -1

cdef int my_daxpy(double scale, double[:] x, double[:] y) nogil except -1

cdef int my_dgemv(double[:, :] A, double[:] x, double[:] y,
                  double alpha=?, double beta=?) nogil except -1

cdef int my_dgemm(double[:, :] A, double[:, :] B, double[:, :] C,
                  double alpha=?, double beta=?) nogil except -1

cdef int my_dger(double[:, :] A, double[:] x, double[:] y,
                 double alpha=?) nogil except -1

