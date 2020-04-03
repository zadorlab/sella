# cython: language_level=3

cdef int cart_to_bond(int a, int b, double[:] dx, double* q, double[:, :] dq,
                      double[:, :, :, :] d2q, bint gradient=?,
                      bint curvature=?) nogil

cdef int cart_to_angle(int a, int b, int c, double[:] dx1, double[:] dx2,
                       double* q, double[:, :] dq, double[:, :, :, :] d2q,
                       double[:, :, :, :] work, bint gradient=?,
                       bint curvature=?) nogil

cdef int cart_to_dihedral(int a, int b, int c, int d, double[:] dx1,
                          double[:] dx2, double[:] dx3, double* q,
                          double[:, :] dq, double[:, :, :, :] d2q,
                          double[:, :, :, :] work, bint gradient=?,
                          bint curvature=?) nogil

cdef int cart_to_dihedral_mod(int a, int b, int c, int d, double[:] dx1,
                              double[:] dx2, double[:] dx3, double* q,
                              double[:, :] dq, double[:, :, :, :] d2q,
                              double[:, :, :, :] work, bint gradient=?,
                              bint curvature=?) nogil
