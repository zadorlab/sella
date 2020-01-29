# cimports

from libc.math cimport sqrt, atan2
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
    # as a vector, and there doesn't seem to be a pure python way to
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
