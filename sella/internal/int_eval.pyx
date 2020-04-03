# cimports

from libc.math cimport sqrt, atan2, sin, cos, pi
from libc.string cimport memset

from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dger, dsyr, dsyr2
from scipy.linalg.cython_lapack cimport dlacpy, dlaset, dlascl

from sella.utilities.blas cimport my_daxpy, my_ddot, my_dger
from sella.utilities.math cimport vec_sum, cross, symmetrize, skew

# imports

import numpy as np

# Constants for BLAS/LAPACK calls
cdef double DNTWO = -2.
cdef double DNUNITY = -1.
cdef double DZERO = 0.
cdef double DUNITY = 1.

cdef int UNITY = 1
cdef int THREE = 3
cdef int NINE = 9
cdef int EIGHTYONE = 81

def test_bond(pos):
    cdef double q
    dx = pos[1] - pos[0]
    dq = np.zeros((2, 3), dtype=float)
    d2q = np.zeros((2, 3, 2, 3), dtype=float)
    cdef int err = cart_to_bond(0, 1, dx, &q, dq, d2q, True, True)
    if err != 0:
        raise RuntimeError
    return q, dq, d2q

cdef int cart_to_bond(int a,
                      int b,
                      double[:] dx,
                      double* q,
                      double[:, :] dq,
                      double[:, :, :, :] d2q,
                      bint gradient=False,
                      bint curvature=False) nogil:
    """Calculate the bond distance (in Angstrom) between atoms a and b,
       and optionally determine the first or first and second deriatives
       of this bond distance with respect to Cartesian coordinates."""

    # Temporary variables
    cdef double tmp
    cdef int info, err

    # bits -> bytes
    cdef int sd_dx = dx.strides[0] >> 3
    cdef int sd_dq = dq.strides[1] >> 3
    cdef int sd_d2q = d2q.strides[1] >> 3

    q[0] = dnrm2(&THREE, &dx[0], &sd_dx)

    if not (gradient or curvature):
        return 0

    tmp = 1 / q[0]
    err = my_daxpy(tmp, dx, dq[b])
    if err != 0: return err
    err = my_daxpy(-1., dq[b], dq[a])
    if err != 0: return err

    if not curvature:
        return 0

    dlaset('G', &THREE, &THREE, &DZERO, &tmp, &d2q[0, 0, 0, 0], &sd_d2q)

    tmp /= -q[0] * q[0]
    dger(&THREE, &THREE, &tmp, &dx[0], &sd_dx, &dx[0], &sd_dx,
         &d2q[0, 0, 0, 0], &sd_d2q)
    dlacpy('G', &THREE, &THREE, &d2q[0, 0, 0, 0], &sd_d2q,
           &d2q[1, 0, 1, 0], &sd_d2q)

    dlacpy('G', &THREE, &THREE, &d2q[0, 0, 0, 0], &sd_d2q,
           &d2q[0, 0, 1, 0], &sd_d2q)
    dlascl('G', &UNITY, &UNITY, &DUNITY, &DNUNITY, &THREE, &THREE,
           &d2q[0, 0, 1, 0], &sd_d2q, &info)
    dlacpy('G', &THREE, &THREE, &d2q[0, 0, 1, 0], &sd_d2q,
           &d2q[1, 0, 0, 0], &sd_d2q)
    return info

def get_bond(atoms, int i, int j, grad=False):
    cdef int natoms = len(atoms)
    dx_np = np.zeros(3)
    cdef double[:] dx = memoryview(dx_np)
    cdef double[:, :] pos = memoryview(atoms.positions)
    cdef int err
    err = vec_sum(pos[i], pos[j], dx, -1.)
    if err != 0:
        raise RuntimeError("get_bond failed!")

    cdef double q

    if grad:
        dq_np = np.zeros((natoms, 3))
    else:
        dq_np = np.empty((0, 3))

    cdef double[:, :] dq = memoryview(dq_np)

    d2q_np = np.empty((0, 3, 0, 3))
    cdef double[:, :, :, :] d2q = memoryview(d2q_np)

    cart_to_bond(0, 1, dx, &q, dq, d2q, grad, False)

    if not grad:
        return q
    else:
        return q, dq_np

def test_angle(pos):
    cdef double q
    dx1 = pos[1] - pos[0]
    dx2 = pos[2] - pos[1]
    dq = np.zeros((3, 3), dtype=float)
    d2q = np.zeros((3, 3, 3, 3), dtype=float)
    work = np.zeros((11, 3, 3, 3), dtype=float)
    cdef int err = cart_to_angle(0, 1, 2, dx1, dx2, &q, dq, d2q, work, True, True)
    if err != 0:
        raise RuntimeError
    return q, dq, d2q

cdef int cart_to_angle(int a,
                       int b,
                       int c,
                       double[:] dx1,
                       double[:] dx2,
                       double* q,
                       double[:, :] dq,
                       double[:, :, :, :] d2q,
                       double[:, :, :, :] work,
                       bint gradient=False,
                       bint curvature=False) nogil:
    """Calculate the angle (in radians) defined by the atoms a-b-c,
       and optionally determine the first or first and second deriatives
       of this angle with respect to Cartesian coordinates."""
    cdef int i, j, k, info, err

    cdef int nzero
    if curvature:
        nzero = 81
    elif gradient:
        nzero = 9
    else:
        nzero = 3

    # We are explicitly *not* zeroing the whole work array, because the
    # higher parts of this array are used in angle sum/diff evaluations.
    memset(&work[0, 0, 0, 0], 0, nzero * sizeof(double))

    # temporary arrays
    cdef double[:] dx_cross = work[0, 0, 0, :]

    # Other temporary variables
    cdef double dx_dot, dx_cross_norm, dx_cross_norm2, dx_cross_norm3
    cdef double r1, r2, r12, r22
    cdef double tmp1, tmp2, tmp3, tmp4

    # bits -> bytes
    cdef int sd_dx = dx1.strides[0] >> 3
    cdef int sd_dq = dq.strides[1] >> 3
    cdef int sd_d2q = d2q.strides[1] >> 3

    cdef int sd_dx_cross = dx_cross.strides[0] >> 3

    dx_dot = ddot(&THREE, &dx1[0], &sd_dx, &dx2[0], &sd_dx)
    cross(dx1, dx2, dx_cross)
    dx_cross_norm = dnrm2(&THREE, &dx_cross[0], &sd_dx_cross)
    q[0] = atan2(dx_cross_norm, -dx_dot)

    if not (gradient or curvature):
        return 0

    cdef double[:, :] dq_int = work[0, 0, 1:3, :]
    cdef int sd_dint = dq_int.strides[1] >> 3

    r12 = ddot(&THREE, &dx1[0], &sd_dx, &dx1[0], &sd_dx)
    r22 = ddot(&THREE, &dx2[0], &sd_dx, &dx2[0], &sd_dx)

    r1 = sqrt(r12)
    r2 = sqrt(r22)

    tmp1 = -dx_dot / (r12 * dx_cross_norm)
    tmp2 = -dx_dot / (r22 * dx_cross_norm)
    tmp3 = 1. / dx_cross_norm

    err = my_daxpy(tmp1, dx1, dq_int[0])
    if err != 0: return err
    err = my_daxpy(tmp3, dx2, dq_int[0])
    if err != 0: return err
    err = my_daxpy(tmp2, dx2, dq_int[1])
    if err != 0: return err
    err = my_daxpy(tmp3, dx1, dq_int[1])
    if err != 0: return err

    # a
    err = my_daxpy(-1., dq_int[0], dq[a])
    if err != 0: return err
    # b
    err = vec_sum(dq_int[0], dq_int[1], dq[b], -1.)
    if err != 0: return err
    # c
    err = my_daxpy(1., dq_int[1], dq[c])
    if err != 0: return err

    if not curvature:
        return 0

    cdef double[:, :, :, :] d2q_int = work[1:3, :3, :2, :3]
    cdef int sd_d2int = d2q_int.strides[1] >> 3

    dx_cross_norm2 = dx_cross_norm * dx_cross_norm
    dx_cross_norm3 = dx_cross_norm2 * dx_cross_norm

    tmp1 = -dx_dot / (r12 * dx_cross_norm)
    tmp2 = -r22 / dx_cross_norm3
    tmp3 = dx_dot / dx_cross_norm3
    tmp4 = dx_dot * (2. / r12 + r22 / dx_cross_norm2) / (r12 * dx_cross_norm)

    dlaset('L', &THREE, &THREE, &DZERO, &tmp1,
           &d2q_int[0, 0, 0, 0], &sd_d2int)
    dsyr2('L', &THREE, &tmp2, &dx1[0], &sd_dx, &dx2[0], &sd_dx,
          &d2q_int[0, 0, 0, 0], &sd_d2int)
    dsyr('L', &THREE, &tmp3, &dx2[0], &sd_dx,
         &d2q_int[0, 0, 0, 0], &sd_d2int)
    dsyr('L', &THREE, &tmp4, &dx1[0], &sd_dx,
         &d2q_int[0, 0, 0, 0], &sd_d2int)

    tmp1 = 1 / dx_cross_norm3
    dsyr('L', &THREE, &tmp1, &dx_cross[0], &sd_dx_cross,
         &d2q_int[0, 0, 1, 0], &sd_d2int)
    symmetrize(&d2q_int[0, 0, 1, 0], 3, sd_d2int)

    tmp1 = -dx_dot / (r22 * dx_cross_norm)
    tmp2 = -r12  / dx_cross_norm3
    tmp4 = dx_dot * (2. / r22 + r12 / dx_cross_norm2) / (r22 * dx_cross_norm)

    dlaset('L', &THREE, &THREE, &DZERO, &tmp1,
           &d2q_int[1, 0, 1, 0], &sd_d2int)
    dsyr2('L', &THREE, &tmp2, &dx1[0], &sd_dx, &dx2[0], &sd_dx,
          &d2q_int[1, 0, 1, 0], &sd_d2int)
    dsyr('L', &THREE, &tmp3, &dx1[0], &sd_dx,
         &d2q_int[1, 0, 1, 0], &sd_d2int)
    dsyr('L', &THREE, &tmp4, &dx2[0], &sd_dx,
         &d2q_int[1, 0, 1, 0], &sd_d2int)
    symmetrize(&d2q_int[0, 0, 0, 0], 6, sd_d2int)

    for i in range(3):
        for j in range(i, 3):
            d2q[0, i, 0, j] = d2q[0, j, 0, i] = d2q_int[0, i, 0, j]
            d2q[1, i, 1, j] = d2q[1, j, 1, i] = (d2q_int[1, i, 1, j]
                                                 + d2q_int[0, i, 0, j]
                                                 - d2q_int[0, j, 1, i]
                                                 - d2q_int[0, i, 1, j])
            d2q[2, i, 2, j] = d2q[2, j, 2, i] = d2q_int[1, i, 1, j]
        for j in range(3):
            d2q[0, i, 1, j] = d2q[1, j, 0, i] = (d2q_int[0, i, 1, j]
                                                 - d2q_int[0, i, 0, j])
            d2q[0, i, 2, j] = d2q[2, j, 0, i] = -d2q_int[0, i, 1, j]
            d2q[1, i, 2, j] = d2q[2, j, 1, i] = (d2q_int[0, i, 1, j]
                                                 - d2q_int[1, i, 1, j])
    return 0


def test_dihedral(pos):
    cdef double q
    dx1 = pos[1] - pos[0]
    dx2 = pos[2] - pos[1]
    dx3 = pos[3] - pos[2]
    dq = np.zeros((4, 3), dtype=float)
    d2q = np.zeros((4, 3, 4, 3), dtype=float)
    work = np.zeros((11, 3, 3, 3), dtype=float)
    cdef int err = cart_to_dihedral(0, 1, 2, 3, dx1, dx2, dx3, &q, dq, d2q, work, True, True)
    if err != 0:
        raise RuntimeError
    return q, dq, d2q

cdef int cart_to_dihedral(int a,
                          int b,
                          int c,
                          int d,
                          double[:] dx1,
                          double[:] dx2,
                          double[:] dx3,
                          double* q,
                          double[:, :] dq,
                          double[:, :, :, :] d2q,
                          double[:, :, :, :] work,
                          bint gradient=False,
                          bint curvature=False) nogil:
    """Calculate the dihedral (in radians) defined by the atoms a-b-c-d,
       and optionally determine the first or first and second deriatives
       of this dihedral with respect to Cartesian coordinates."""
    cdef int nzero, err
    if curvature:
        nzero = 297
    elif gradient:
        nzero = 57
    else:
        nzero = 12
    memset(&work[0, 0, 0, 0], 0, nzero * sizeof(double))

    cdef size_t i, j, k
    cdef double r22, r2, numer, denom, tmp1, tmp2, tmp3

    cdef double[:] dx1_cross_dx2 = work[0, 0, 0, :]
    cdef double[:] dx2_cross_dx3 = work[0, 0, 1, :]
    cdef double[:] tmpar1 = work[0, 0, 2, :]
    cdef double[:] dx1_cross_dx3 = work[0, 1, 0, :]

    # bits -> bytes
    cdef int sd_dx = dx1.strides[0] >> 3
    cdef int sd_work = dx1_cross_dx2.strides[0] >> 3
    cdef int sd_nine = 9 * sd_work

    r22 = ddot(&THREE, &dx2[0], &sd_dx, &dx2[0], &sd_dx)
    r2 = sqrt(r22)
    cross(dx1, dx2, dx1_cross_dx2)
    cross(dx2, dx3, dx2_cross_dx3)
    cross(dx1_cross_dx2, dx2_cross_dx3, tmpar1)
    numer = ddot(&THREE, &tmpar1[0], &sd_work, &dx2[0], &sd_dx) / r2
    denom = ddot(&THREE, &dx1_cross_dx2[0], &sd_work,
                 &dx2_cross_dx3[0], &sd_work)
    q[0] = atan2(numer, denom)

    if not (gradient or curvature):
        return 0

    cdef double[:, :] dnumer = work[1, 0, :, :]
    cdef double[:, :] ddenom = work[1, 1, :, :]
    cdef double[:, :] dq_int = work[1, 2, :, :]

    cdef int sd_dq = dq.strides[1] >> 3
    cdef int sd_dint = dq_int.strides[1] >> 3

    cross(dx1, dx3, dx1_cross_dx3)

    cross(dx2_cross_dx3, dx2, ddenom[0])
    cross(dx1, dx2_cross_dx3, ddenom[1])
    cross(dx1_cross_dx2, dx3, tmpar1)
    err = my_daxpy(1., tmpar1, ddenom[1])
    if err != 0: return err
    cross(dx2, dx1_cross_dx2, ddenom[2])

    tmp1 = -r2
    tmp2 = -numer / r22

    err = my_daxpy(tmp1, dx2_cross_dx3, dnumer[0])
    if err != 0: return err
    err = my_daxpy(tmp2, dx2, dnumer[1])
    if err != 0: return err
    err = my_daxpy(r2, dx1_cross_dx3, dnumer[1])
    if err != 0: return err
    err = my_daxpy(tmp1, dx1_cross_dx2, dnumer[2])
    if err != 0: return err

    tmp1 = numer * numer + denom * denom
    tmp2 = denom / tmp1
    tmp3 = -numer / tmp1

    for i in range(3):
        err = my_daxpy(tmp2, dnumer[i], dq_int[i])
        if err != 0: return err
        err = my_daxpy(tmp3, ddenom[i], dq_int[i])
        if err != 0: return err

    err = my_daxpy(1., dq_int[0], dq[a])
    if err != 0: return err

    err = vec_sum(dq_int[1], dq_int[0], dq[b], -1.)
    if err != 0: return err
    err = vec_sum(dq_int[2], dq_int[1], dq[c], -1.)
    if err != 0: return err

    err = my_daxpy(-1., dq_int[2], dq[d])
    if err != 0: return err

    if not curvature:
        return 0
    cdef double[:, :, :, :] d2numer = work[2:5, :, :, :]
    cdef double[:, :, :, :] d2denom = work[5:8, :, :, :]
    cdef double[:, :, :, :] d2q_int = work[8:11, :, :, :]

    cdef int sd_d2q = d2q.strides[1] >> 3
    cdef int sd_d2int = d2numer.strides[1] >> 3

    tmp1 = ddot(&THREE, &dx2[0], &sd_dx, &dx3[0], &sd_dx)
    dlaset('G', &THREE, &THREE, &DZERO, &tmp1, &d2denom[0, 0, 1, 0], &sd_d2int)
    dger(&THREE, &THREE, &DNTWO, &dx2[0], &sd_dx, &dx3[0], &sd_dx,
         &d2denom[0, 0, 1, 0], &sd_d2int)
    dger(&THREE, &THREE, &DUNITY, &dx3[0], &sd_dx, &dx2[0], &sd_dx,
         &d2denom[0, 0, 1, 0], &sd_d2int)

    tmp1 = -r22
    dlaset('L', &THREE, &THREE, &DZERO, &tmp1, &d2denom[0, 0, 2, 0], &sd_d2int)
    dsyr('L', &THREE, &DUNITY, &dx2[0], &sd_dx,
         &d2denom[0, 0, 2, 0], &sd_d2int)
    symmetrize(&d2denom[0, 0, 2, 0], 3, sd_d2int)

    tmp1 = -2 * ddot(&THREE, &dx1[0], &sd_dx, &dx3[0], &sd_dx)
    dlaset('L', &THREE, &THREE, &DZERO, &tmp1, &d2denom[1, 0, 1, 0], &sd_d2int)
    dsyr2('L', &THREE, &DUNITY, &dx1[0], &sd_dx, &dx3[0], &sd_dx,
          &d2denom[1, 0, 1, 0], &sd_d2int)

    tmp1 = ddot(&THREE, &dx1[0], &sd_dx, &dx2[0], &sd_dx)
    dlaset('G', &THREE, &THREE, &DZERO, &tmp1, &d2denom[1, 0, 2, 0], &sd_d2int)
    dger(&THREE, &THREE, &DNTWO, &dx1[0], &sd_dx, &dx2[0], &sd_dx,
         &d2denom[1, 0, 2, 0], &sd_d2int)
    dger(&THREE, &THREE, &DUNITY, &dx2[0], &sd_dx, &dx1[0], &sd_dx,
         &d2denom[1, 0, 2, 0], &sd_d2int)

    skew(dx3, d2numer[0, :, 1, :], -r2)
    tmp1 = 1. / r2
    dger(&THREE, &THREE, &tmp1, &dx2[0], &sd_dx, &dx2_cross_dx3[0], &sd_work,
         &d2numer[0, 0, 1, 0], &sd_d2int)

    skew(dx2, d2numer[0, :, 2, :], r2)

    tmp1 = numer / r22
    dlaset('L', &THREE, &THREE, &DZERO, &tmp1, &d2numer[1, 0, 1, 0], &sd_d2int)
    tmp2 = -tmp1 / r22
    tmp3 = -1 / r2
    dsyr('L', &THREE, &tmp2, &dx2[0], &sd_dx,
         &d2numer[1, 0, 1, 0], &sd_d2int)
    dsyr2('L', &THREE, &tmp3, &dx1_cross_dx3[0], &sd_work, &dx2[0], &sd_dx,
          &d2numer[1, 0, 1, 0], &sd_d2int)

    skew(dx1, d2numer[1, :, 2, :], -r2)
    tmp1 = 1. / r2
    dger(&THREE, &THREE, &tmp1, &dx1_cross_dx2[0], &sd_work, &dx2[0], &sd_dx,
         &d2numer[1, 0, 2, 0], &sd_d2int)

    dsyr2('L', &NINE, &DNUNITY, &dq_int[0, 0], &sd_work,
          &ddenom[0, 0], &sd_work, &d2numer[0, 0, 0, 0], &sd_nine)
    dsyr2('L', &NINE, &DUNITY, &dq_int[0, 0], &sd_work,
          &dnumer[0, 0], &sd_work, &d2denom[0, 0, 0, 0], &sd_nine)

    tmp1 = numer * numer + denom * denom
    tmp2 = denom / tmp1
    tmp3 = -numer / tmp1
    # can't use my_daxpy here, because we are reinterpreting a 4d array
    # as a vector, and there doesn't seem to be a pure cython way to
    # reshape a typed memoryview
    daxpy(&EIGHTYONE, &tmp2, &d2numer[0, 0, 0, 0], &sd_work,
          &d2q_int[0, 0, 0, 0], &sd_work)
    daxpy(&EIGHTYONE, &tmp3, &d2denom[0, 0, 0, 0], &sd_work,
          &d2q_int[0, 0, 0, 0], &sd_work)
    symmetrize(&d2q_int[0, 0, 0, 0], 9, sd_d2int)

    for i in range(3):
        for j in range(i, 3):
            d2q[0, i, 0, j] = d2q[0, j, 0, i] = d2q_int[0, i, 0, j]
            d2q[1, i, 1, j] = d2q[1, j, 1, i] = (d2q_int[1, i, 1, j]
                                                 + d2q_int[0, i, 0, j]
                                                 - d2q_int[0, j, 1, i]
                                                 - d2q_int[0, i, 1, j])
            d2q[2, i, 2, j] = d2q[2, j, 2, i] = (d2q_int[2, i, 2, j]
                                                 + d2q_int[1, i, 1, j]
                                                 - d2q_int[1, i, 2, j]
                                                 - d2q_int[1, j, 2, i])
            d2q[3, i, 3, j] = d2q[3, j, 3, i] = d2q_int[2, i, 2, j]
        for j in range(3):
            d2q[0, i, 1, j] = d2q[1, j, 0, i] = (d2q_int[0, i, 1, j]
                                                 - d2q_int[0, i, 0, j])
            d2q[0, i, 2, j] = d2q[2, j, 0, i] = (d2q_int[0, i, 2, j]
                                                 - d2q_int[0, i, 1, j])
            d2q[0, i, 3, j] = d2q[3, j, 0, i] = -d2q_int[0, i, 2, j]

            d2q[1, i, 2, j] = d2q[2, j, 1, i] = (d2q_int[1, i, 2, j]
                                                 + d2q_int[0, i, 1, j]
                                                 - d2q_int[1, i, 1, j]
                                                 - d2q_int[0, i, 2, j])
            d2q[1, i, 3, j] = d2q[3, j, 1, i] = (d2q_int[0, i, 2, j]
                                                 - d2q_int[1, i, 2, j])

            d2q[2, i, 3, j] = d2q[3, j, 2, i] = (d2q_int[1, i, 2, j]
                                                 - d2q_int[2, i, 2, j])
    return 0


def test_dihedral_mod(pos):
    cdef double q
    dx1 = pos[1] - pos[0]
    dx2 = pos[2] - pos[1]
    dx3 = pos[3] - pos[2]
    dq = np.zeros((4, 3), dtype=float)
    d2q = np.zeros((4, 3, 4, 3), dtype=float)
    work = np.zeros((11, 3, 3, 3), dtype=float)
    cdef int err = cart_to_dihedral_mod(0, 1, 2, 3, dx1, dx2, dx3, &q, dq, d2q, work, True, True)
    if err != 0:
        raise RuntimeError
    return q, dq, d2q


cdef int cart_to_dihedral_mod(int a,
                              int b,
                              int c,
                              int d,
                              double[:] dx1,
                              double[:] dx2,
                              double[:] dx3,
                              double* q,
                              double[:, :] dq,
                              double[:, :, :, :] d2q,
                              double[:, :, :, :] work,
                              bint gradient=False,
                              bint curvature=False) nogil:
    cdef int err, n, m, i, j
    cdef double tau, alpha, beta, sina, sinb, cosa, cosb
    cdef double[:, :] dalpha, dbeta
    cdef double[:, :, :, :] d2alpha, d2beta
    err = cart_to_dihedral(a, b, c, d, dx1, dx2, dx3, &tau, dq,
                           d2q, work, gradient, curvature)
    if err != 0:  return err

    # clear out work
    memset(&work[0, 0, 0, 0], 0, 297 * sizeof(double))

    dalpha = work[0, 0]
    dbeta = work[0, 1]
    d2alpha = work[1:4]
    d2beta = work[4:7]
    err = cart_to_angle(0, 1, 2, dx1, dx2, &alpha, dalpha, d2alpha,
                        work[7:], gradient, curvature)
    if err != 0:  return err

    err = cart_to_angle(0, 1, 2, dx2, dx3, &beta, dbeta, d2beta,
                        work[7:], gradient, curvature)
    if err != 0:  return err

    sina = sin(alpha)
    sinb = sin(beta)

    q[0] = tau * sina * sinb

    if not (gradient or curvature):
        return 0

    cosa = cos(alpha)
    cosb = cos(beta)

    # do curvature before gradient so that we still have dtau/dx
    if curvature:
        # So, I realize how awful this looks...
        # It is awful...
        # I will come up with a better way of doing this.
        # tau^2
        for m in range(4):
            for i in range(3):
                for n in range(4):
                    for j in range(3):
                        d2q[m, i, n, j] *= sina * sinb

        # tau alpha, alpha tau, tau beta, beta tau
        for i in range(3):
            for m in range(3):
                for j in range(3):
                    d2q[0, i, m, j] += cosa * sinb * dq[a, i] * dalpha[m, j]
                    d2q[1, i, m, j] += cosa * sinb * dq[b, i] * dalpha[m, j]
                    d2q[2, i, m, j] += cosa * sinb * dq[c, i] * dalpha[m, j]
                    d2q[3, i, m, j] += cosa * sinb * dq[d, i] * dalpha[m, j]

                    d2q[m, j, 0, i] += cosa * sinb * dq[a, i] * dalpha[m, j]
                    d2q[m, j, 1, i] += cosa * sinb * dq[b, i] * dalpha[m, j]
                    d2q[m, j, 2, i] += cosa * sinb * dq[c, i] * dalpha[m, j]
                    d2q[m, j, 3, i] += cosa * sinb * dq[d, i] * dalpha[m, j]

                    d2q[0, i, m + 1, j] += sina * cosb * dq[a, i] * dbeta[m, j]
                    d2q[1, i, m + 1, j] += sina * cosb * dq[b, i] * dbeta[m, j]
                    d2q[2, i, m + 1, j] += sina * cosb * dq[c, i] * dbeta[m, j]
                    d2q[3, i, m + 1, j] += sina * cosb * dq[d, i] * dbeta[m, j]

                    d2q[m + 1, j, 0, i] += sina * cosb * dq[a, i] * dbeta[m, j]
                    d2q[m + 1, j, 1, i] += sina * cosb * dq[b, i] * dbeta[m, j]
                    d2q[m + 1, j, 2, i] += sina * cosb * dq[c, i] * dbeta[m, j]
                    d2q[m + 1, j, 3, i] += sina * cosb * dq[d, i] * dbeta[m, j]


        # alpha alpha, alpha^2, alpha beta, beta alpha, beta beta, beta^2
        for n in range(3):
            for i in range(3):
                for m in range(3):
                    for j in range(3):
                        d2q[n, i, m, j] -= tau * sina * sinb * dalpha[n, i] * dalpha[m, j]
                        d2q[n, i, m, j] += tau * cosa * sinb * d2alpha[n, i, m, j]
                        d2q[n, i, m + 1, j] += tau * cosa * cosb * dalpha[n, i] * dbeta[m, j]
                        d2q[m + 1, j, n, i] += tau * cosa * cosb * dalpha[n, i] * dbeta[m, j]
                        d2q[n + 1, i, m + 1, j] -= tau * sina * sinb * dbeta[n, i] * dbeta[m, j]
                        d2q[n + 1, i, m + 1, j] += tau * sina * cosb * d2beta[n, i, m, j]

    for i in range(3):
        dq[a, i] *= sina * sinb
        dq[a, i] += tau * cosa * sinb * dalpha[0, i]

        dq[b, i] *= sina * sinb
        dq[b, i] += tau * (cosa * sinb * dalpha[1, i] + sina * cosb * dbeta[0, i])

        dq[c, i] *= sina * sinb
        dq[c, i] += tau * (cosa * sinb * dalpha[2, i] + sina * cosb * dbeta[1, i])

        dq[d, i] *= sina * sinb
        dq[d, i] += tau * sina * cosb * dbeta[2, i]

    return 0
