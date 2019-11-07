# cython: language_level=3

from libc.stdint cimport uint8_t

cdef class CartToInternal:
    cdef bint grad, curv, calc_required
    cdef public int natoms, nreal, ndummies, nbonds, nangles, ndihedrals
    cdef public int nangle_sums, nangle_diffs, ncart
    cdef int nq, nx, nint, next, lwork, nmin, nmax
    cdef public int[:] dinds
    cdef public int[:, :] cart, bonds, angles, dihedrals, angle_sums
    cdef public int[:, :] angle_diffs
    cdef double[:] q1
    cdef double[:, :] pos, work2
    cdef double[:] dx1, dx2, dx3, work3, sing
    cdef double[:, :, :] dq
    cdef double[:, :, :, :] work1
    cdef double[:, :, :, :, :] d2q_bonds, d2q_angles, d2q_dihedrals
    cdef double[:, :, :, :, :] d2q_angle_sums, d2q_angle_diffs
    cdef double[:, :] Uint, Uext, Binv, Usvd
    cdef dict __dict__

    cdef bint geom_changed(CartToInternal self, double[:, :] pos,
                           double[:, :] dummypos=?) nogil

    cdef bint _validate_pos(CartToInternal self, double[:, :] pos,
                            double[:, :] dummypos=?) nogil

    cdef int _update(CartToInternal self, double[:, :] pos,
                     double[:, :] dummypos=?, bint grad=?, bint curv=?,
                     bint force=?) nogil except -1

    cdef int _U_update(CartToInternal self, double[:, :] pos,
                       double[:, :] dummypos=?, bint force=?) nogil

    cdef int _angle_sum_diff(CartToInternal self, int[:] indices, double sign,
                             double* q, double[:, :] dq,
                             double[:, :, :, :] d2q,
                             bint grad, bint curv) nogil

    cdef double _h0_bond(CartToInternal self, int a, int b, double[:, :] rij,
                         double[:] rcov, double conv, double Ab=?,
                         double Bb=?) nogil

    cdef double _h0_angle(CartToInternal self, int a, int b, int c,
                          double[:, :] rij, double[:] rcov, double conv,
                          double Aa=?, double Ba=?, double Ca=?,
                          double Da=?) nogil

    cdef double _h0_dihedral(CartToInternal self, int a, int b, int c, int d,
                             int[:] nbonds, double[:, :] rij, double[:] rcov,
                             double conv, double At=?, double Bt=?,
                             double Ct=?, double Dt=?, double Et=?) nogil

cdef class Constraints(CartToInternal):
    cdef bint proj_trans, proj_rot, calc_res
    cdef int ntrans, nrot, ninternal
    cdef public double[:] res, target
    cdef double[:] center
    cdef double[:, :] Ured
    cdef uint8_t[:] trans_dirs
    cdef double[:, :] rot_axes
    cdef double[:, :] tvecs
    cdef double[:, :] rvecs

    cdef int _update(Constraints self,
                     double[:, :] pos,
                     double[:, :] dummypos=?,
                     bint grad=?,
                     bint curv=?,
                     bint force=?) nogil except -1

    cdef int project_rotation(Constraints self) nogil


cdef class D2q:
    cdef int natoms, ncart, nbonds, nangles, ndihedrals, nangle_sums
    cdef int nangle_diffs
    cdef int nq, nx, sw1, sw2, sw3
    cdef int[:, :] bonds, angles, dihedrals, angle_sums, angle_diffs
    cdef double[:, :, :, :, :] Dbonds, Dangles, Ddihedrals
    cdef double[:, :, :, :, :] Dangle_sums, Dangle_diffs
    cdef double[:, :] work1, work2, work3

    cdef int _ld(D2q self, size_t start, size_t nq, size_t nind,
                 int[:, :] q, double[:, :, :, :, :] D2, double[:] v,
                 double[:, :, :, :] res) nogil

    cdef int _rd(D2q self, size_t start, size_t nq, size_t nind,
                 int[:, :] q, double[:, :, :, :, :] D2, double[:] v,
                 double[:, :, :] res) nogil

    cdef int _dd(D2q self, size_t start, size_t nq, size_t nind,
                 int[:, :] q, double[:, :, :, :, :] D2, double[:] v1,
                 double[:] v2, double[:] res) nogil
