# cython: language_level=3

# cimports

cimport cython

from libc.math cimport fabs, exp, pi, sqrt
from libc.string cimport memset
from libc.stdint cimport uint8_t

from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dgemv, dcopy

from sella.utilities.math cimport (my_daxpy, vec_sum, mgs, cross, svdr,
                                   normalize, mppi)
from sella.internal.int_eval cimport (cart_to_bond, cart_to_angle,
                                      cart_to_dihedral)

# imports
import warnings
import numpy as np
from ase import Atoms, units
from ase.data import covalent_radii

# Constants for BLAS/LAPACK calls
cdef double DNUNITY = -1.
cdef double DZERO = 0.
cdef double DUNITY = 1.

cdef int UNITY = 1
cdef int THREE = 3

cdef class CartToInternal:
    def __init__(self, atoms, *args, dummies=None, **kwargs):
        if self.natoms <= 0:
            raise ValueError("Must have at least 1 atom!")
        self.nmin = min(self.nq, self.nx)
        self.nmax = max(self.nq, self.nx)

        # Allocate arrays
        #
        # We use memoryview and numpy to initialize these arrays, because
        # some of the arrays may have a dimension of size 0, which
        # cython.view.array does not permit.
        self.pos = memoryview(atoms.positions.copy())
#        self.pos = memoryview(np.zeros((self.natoms, 3), dtype=np.float64))
        self.dx1 = memoryview(np.zeros(3, dtype=np.float64))
        self.dx2 = memoryview(np.zeros(3, dtype=np.float64))
        self.dx3 = memoryview(np.zeros(3, dtype=np.float64))
        self.q1 = memoryview(np.zeros(self.nq, dtype=np.float64))
        self.dq = memoryview(np.zeros((self.nq, self.natoms, 3),
                                      dtype=np.float64))
        self.d2q_bonds = memoryview(np.zeros((self.nbonds, 2, 3, 2, 3),
                                             dtype=np.float64))
        self.d2q_angles = memoryview(np.zeros((self.nangles, 3, 3, 3, 3),
                                              dtype=np.float64))
        self.d2q_dihedrals = memoryview(np.zeros((self.ndihedrals, 4, 3, 4, 3),
                                                 dtype=np.float64))
        self.d2q_angle_sums = memoryview(np.zeros((self.nangle_sums, 4, 3,
                                                   4, 3), dtype=np.float64))
        self.d2q_angle_diffs = memoryview(np.zeros((self.nangle_diffs, 4, 3,
                                                    4, 3), dtype=np.float64))
        self.work1 = memoryview(np.zeros((11, 3, 3, 3), dtype=np.float64))
        self.work2 = memoryview(np.zeros((self.natoms, 3), dtype=np.float64))

        # Things for SVD
        self.lwork = 2 * max(3 * self.nmin + self.nmax, 5 * self.nmin, 1)
        self.work3 = memoryview(np.zeros(self.lwork, dtype=np.float64))

        self.sing = memoryview(np.zeros(self.nmin, dtype=np.float64))
        self.Uint = memoryview(np.zeros((self.nmax, self.nmax),
                                        dtype=np.float64))
        self.Uext = memoryview(np.zeros((self.nmax, self.nmax),
                                        dtype=np.float64))
        self.Binv = memoryview(np.zeros((self.nx, self.nq), dtype=np.float64))
        self.Usvd = memoryview(np.zeros((self.nmax, self.nmax),
                                        dtype=np.float64))
        self.grad = False
        self.curv = False
        if dummies is None:
            self.dummies = Atoms()
        else:
            self.dummies = dummies
        self.nint = -1
        self.next = -1
        self.calc_required = True

    def __cinit__(CartToInternal self,
                  atoms,
                  *args,
                  int[:, :] cart=None,
                  int[:, :] bonds=None,
                  int[:, :] angles=None,
                  int[:, :] dihedrals=None,
                  int[:, :] angle_sums=None,
                  int[:, :] angle_diffs=None,
                  dummies=None,
                  int[:] dinds=None,
                  **kwargs):

        cdef size_t sd = sizeof(double)

        self.natoms = len(atoms)
        if self.natoms <= 0:
            return
        self.nx = 3 * self.natoms

        if cart is None:
            self.cart = memoryview(np.empty((0, 2), dtype=np.int32))
        else:
            self.cart = cart
        self.ncart = len(self.cart)

        if bonds is None:
            self.bonds = memoryview(np.empty((0, 2), dtype=np.int32))
        else:
            self.bonds = bonds
        self.nbonds = len(self.bonds)

        if angles is None:
            self.angles = memoryview(np.empty((0, 3), dtype=np.int32))
        else:
            self.angles = angles
        self.nangles = len(self.angles)

        if dihedrals is None:
            self.dihedrals = memoryview(np.empty((0, 4), dtype=np.int32))
        else:
            self.dihedrals = dihedrals
        self.ndihedrals = len(self.dihedrals)

        if angle_sums is None:
            self.angle_sums = memoryview(np.empty((0, 4), dtype=np.int32))
        else:
            self.angle_sums = angle_sums
        self.nangle_sums = len(self.angle_sums)

        if angle_diffs is None:
            self.angle_diffs = memoryview(np.empty((0, 4), dtype=np.int32))
        else:
            self.angle_diffs = angle_diffs
        self.nangle_diffs = len(self.angle_diffs)

        if dinds is None:
            self.dinds = memoryview(-np.ones(self.natoms, dtype=np.int32))
        else:
            self.dinds = dinds

        self.nq = (self.ncart + self.nbonds + self.nangles
                   + self.ndihedrals + self.nangle_sums + self.nangle_diffs)


    def get_q(self, double[:, :] pos):
        with nogil:
            info = self._update(pos, False, False)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        # We use np.array() instead of np.asarray() because we want to
        # return a copy.
        return np.array(self.q1)

    def get_B(self, double[:, :] pos):
        with nogil:
            info = self._update(pos, True, False)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return np.array(self.dq).reshape((self.nq, self.nx))

    def get_D(self, double[:, :] pos):
        with nogil:
            info = self._update(pos, True, True)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return D2q(self.natoms, self.ncart, self.bonds, self.angles,
                   self.dihedrals, self.angle_sums, self.angle_diffs,
                   self.d2q_bonds, self.d2q_angles, self.d2q_dihedrals,
                   self.d2q_angle_sums, self.d2q_angle_diffs)

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef bint geom_changed(CartToInternal self, double[:, :] pos) nogil:
        cdef int n, i, j
        for n in range(self.nx):
            i = n // 3
            j = n % 3
            if self.pos[i, j] != pos[i, j]:
                return True
        return False

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _update(CartToInternal self,
                     double[:, :] pos,
                     bint grad=False,
                     bint curv=False,
                     bint force=False) nogil:
        # The purpose of this check is to determine whether the positions
        # array has been changed at all since the last internal coordinate
        # evaluation, which is why we are doing exact floating point
        # comparison with ==.
        if not self.calc_required and not force:
            if (self.grad or not grad) and (self.curv or not curv):
                if not self.geom_changed(pos):
                    return 0
        self.calc_required = True
        self.grad = grad or curv
        self.curv = curv
        cdef size_t n, m, i, j, k, l
        cdef int info, err
        cdef size_t sd = sizeof(double)

        # Zero out our arrays
        memset(&self.q1[0], 0, self.nq * sd)
        memset(&self.dq[0, 0, 0], 0, self.nq * self.nx * sd)

        # I'm not sure why these "> 0" checks are necessary; according to the
        # C standard, memset accepts a length of 0 (though it results in a
        # no-op), but Cython keeps giving out-of-bounds errors. Maybe it's
        # because indexing memoryviews of size 0 doesn't work?
        if self.nbonds > 0:
            memset(&self.d2q_bonds[0, 0, 0, 0, 0], 0,
                   self.nbonds * 36 * sd)
        if self.nangles > 0:
            memset(&self.d2q_angles[0, 0, 0, 0, 0], 0,
                   self.nangles * 81 * sd)
        if self.ndihedrals > 0:
            memset(&self.d2q_dihedrals[0, 0, 0, 0, 0], 0,
                   self.ndihedrals * 144 * sd)
        if self.nangle_sums > 0:
            memset(&self.d2q_angle_sums[0, 0, 0, 0, 0], 0,
                   self.nangle_sums * 144 * sd)
        if self.nangle_diffs > 0:
            memset(&self.d2q_angle_diffs[0, 0, 0, 0, 0], 0,
                   self.nangle_diffs * 144 * sd)
        memset(&self.work1[0, 0, 0, 0], 0, 297 * sd)
        memset(&self.work2[0, 0], 0, self.nx * sd)

        self.pos[:, :] = pos

        for n in range(self.ncart):
            i = self.cart[n, 0]
            j = self.cart[n, 1]
            self.q1[n] = pos[i, j]
            self.dq[n, i, j] = 1.
            # d2q is the 0 matrix for cartesian coords

        m = self.ncart
        for n in range(self.nbonds):
            i = self.bonds[n, 0]
            j = self.bonds[n, 1]
            err = vec_sum(pos[j], pos[i], self.dx1, -1.)
            if err != 0: return err
            info = cart_to_bond(i, j, self.dx1, &self.q1[m + n],
                                self.dq[m + n], self.d2q_bonds[n], grad, curv)
            if info < 0: return info

        m += self.nbonds
        for n in range(self.nangles):
            i = self.angles[n, 0]
            j = self.angles[n, 1]
            k = self.angles[n, 2]
            err = vec_sum(pos[k], pos[j], self.dx2, -1.)
            if err != 0: return err
            err = vec_sum(pos[j], pos[i], self.dx1, -1.)
            if err != 0: return err
            info = cart_to_angle(i, j, k, self.dx1, self.dx2, &self.q1[m + n],
                                 self.dq[m + n], self.d2q_angles[n],
                                 self.work1, grad, curv)
            if info < 0: return info

        m += self.nangles
        for n in range(self.ndihedrals):
            i = self.dihedrals[n, 0]
            j = self.dihedrals[n, 1]
            k = self.dihedrals[n, 2]
            l = self.dihedrals[n, 3]
            err = vec_sum(pos[l], pos[k], self.dx3, -1.)
            if err != 0: return err
            err = vec_sum(pos[k], pos[j], self.dx2, -1.)
            if err != 0: return err
            err = vec_sum(pos[j], pos[i], self.dx1, -1.)
            if err != 0: return err
            info = cart_to_dihedral(i, j, k, l, self.dx1, self.dx2, self.dx3,
                                    &self.q1[m + n], self.dq[m + n],
                                    self.d2q_dihedrals[n], self.work1,
                                    grad, curv)
            if info < 0:
                return info

        m += self.ndihedrals
        for n in range(self.nangle_sums):
            info = self._angle_sum_diff(self.angle_sums[n], 1.,
                                        &self.q1[m + n], self.dq[m + n],
                                        self.d2q_angle_sums[n], grad, curv)
            if info < 0:
                return info

        m += self.nangle_sums
        for n in range(self.nangle_diffs):
            info = self._angle_sum_diff(self.angle_diffs[n], -1.,
                                        &self.q1[m + n], self.dq[m + n],
                                        self.d2q_angle_diffs[n], grad, curv)
            if info < 0:
                return info

        self.calc_required = False
        self.nint = -1
        return 0

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _U_update(CartToInternal self,
                       double[:, :] pos,
                       bint force=False) nogil:
        cdef int err = self._update(pos, True, False, force)
        if err != 0:
            return err

        if self.nint > 0:
            return 0

        cdef int sddq = self.dq.strides[2] >> 3
        cdef int sdu = self.Uint.strides[1] >> 3
        memset(&self.Uint[0, 0], 0, self.nmax * self.nmax * sizeof(double))
        memset(&self.Uext[0, 0], 0, self.nmax * self.nmax * sizeof(double))

        for n in range(self.nq):
            dcopy(&self.nx, &self.dq[n, 0, 0], &sddq, &self.Uint[n, 0], &sdu)

        self.nint = mppi(self.nq, self.nx, self.Uint, self.Usvd, self.Uext,
                         self.sing, self.Binv, self.work3)

        #self.nint = svdr(self.nq, self.nx, self.Uint, self.Uext, self.sing,
        #                 self.work3)
        if self.nint < 0:
            return self.nint
        self.next = self.nx - self.nint

        self.calc_required = False
        return 0


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _angle_sum_diff(CartToInternal self,
                             int[:] indices,
                             double sign,
                             double* q,
                             double[:, :] dq,
                             double[:, :, :, :] d2q,
                             bint grad,
                             bint curv) nogil:
        cdef int err
        cdef size_t i, j, k, l, m
        cdef size_t sd = sizeof(double)
        memset(&self.work1[0, 0, 0, 0], 0, 297 * sd)
        memset(&self.work2[0, 0], 0, self.nx * sd)
        i = indices[0]
        j = indices[1]
        k = indices[2]
        l = indices[3]

        # First part of the angle sum/diff
        err = vec_sum(self.pos[l], self.pos[j], self.dx2, -1.)
        if err != 0: return err
        err = vec_sum(self.pos[j], self.pos[i], self.dx1, -1.)
        if err != 0: return err
        info = cart_to_angle(i, j, l, self.dx1, self.dx2,
                             q, dq, self.work1[4:7], self.work1,
                             grad, curv)
        # Second part
        self.dx1[:] = self.pos[j, :]
        daxpy(&THREE, &DNUNITY, &self.pos[k, 0], &UNITY, &self.dx1[0], &UNITY)
        err = cart_to_angle(k, j, l, self.dx1, self.dx2,
                            &self.work1[3, 0, 0, 0], self.work2,
                            self.work1[7:10], self.work1,
                            grad, curv)
        if err != 0: return err

        # Combine results
        q[0] += sign * self.work1[3, 0, 0, 0]
        if grad or curv:
            daxpy(&self.nx, &sign, &self.work2[0, 0], &UNITY,
                  &dq[0, 0], &UNITY)

        # Combine second derivatives
        for i in range(3):
            d2q[0, i, 0, :] = self.work1[4, i, 0, :]
            d2q[0, i, 1, :] = self.work1[4, i, 1, :]
            d2q[0, i, 3, :] = self.work1[4, i, 2, :]

            d2q[1, i, 0, :] = self.work1[5, i, 0, :]
            d2q[1, i, 1, :] = self.work1[5, i, 1, :]
            d2q[1, i, 3, :] = self.work1[5, i, 2, :]

            d2q[3, i, 0, :] = self.work1[6, i, 0, :]
            d2q[3, i, 1, :] = self.work1[6, i, 1, :]
            d2q[3, i, 3, :] = self.work1[6, i, 2, :]

        for i in range(3):
            my_daxpy(sign, self.work1[7, i, 0], d2q[2, i, 2])
            my_daxpy(sign, self.work1[7, i, 1], d2q[2, i, 1])
            my_daxpy(sign, self.work1[7, i, 2], d2q[2, i, 3])

            my_daxpy(sign, self.work1[8, i, 0], d2q[1, i, 2])
            my_daxpy(sign, self.work1[8, i, 1], d2q[1, i, 1])
            my_daxpy(sign, self.work1[8, i, 2], d2q[1, i, 3])

            my_daxpy(sign, self.work1[9, i, 0], d2q[3, i, 2])
            my_daxpy(sign, self.work1[9, i, 1], d2q[3, i, 1])
            my_daxpy(sign, self.work1[9, i, 2], d2q[3, i, 3])
        return 0

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def guess_hessian(self, atoms, double h0cart=70.):
        atoms = atoms + self.dummies
        rcov = covalent_radii[atoms.numbers] / units.Bohr
        rij_np = atoms.get_all_distances() / units.Bohr
        cdef double[:, :] rij = memoryview(rij_np)

        nbonds_np = np.zeros(self.natoms, dtype=np.int32)
        cdef int[:] nbonds = memoryview(nbonds_np)

        h0_np = np.zeros(self.nq, np.float64)
        cdef double[:] h0 = memoryview(h0_np)

        cdef int i
        cdef int n
        cdef double Hartree = units.Hartree
        cdef double Bohr = units.Bohr
        cdef double rcovab, rcovbc
        cdef double conv

        # FIXME: for some reason, this fails at runtime when the gil
        # has been released
        #with nogil:
        for i in range(self.nbonds):
            nbonds[self.bonds[i, 0]] += 1
            nbonds[self.bonds[i, 1]] += 1

        n = 0
        for i in range(self.ncart):
            h0[n] = h0cart
            n += 1

        conv = Hartree / Bohr**2
        for i in range(self.nbonds):
            h0[n] = self._h0_bond(self.bonds[i, 0], self.bonds[i, 1],
                                  rij, rcov, conv)
            n += 1

        conv = Hartree
        for i in range(self.nangles):
            h0[n] = self._h0_angle(self.angles[i, 0], self.angles[i, 1],
                                   self.angles[i, 2], rij, rcov, conv)
            n += 1

        for i in range(self.ndihedrals):
            h0[n] = self._h0_dihedral(self.dihedrals[i, 0],
                                      self.dihedrals[i, 1],
                                      self.dihedrals[i, 2],
                                      self.dihedrals[i, 3],
                                      nbonds, rij, rcov, conv)
            n += 1

        for i in range(self.nangle_sums):
            h0[n] = self._h0_angle(self.angle_sums[i, 0],
                                   self.angle_sums[i, 1],
                                   self.angle_sums[i, 3],
                                   rij, rcov, conv)
            h0[n] += self._h0_angle(self.angle_sums[i, 2],
                                    self.angle_sums[i, 1],
                                    self.angle_sums[i, 3],
                                    rij, rcov, conv)
            n += 1

        for i in range(self.nangle_diffs):
            h0[n] = self._h0_angle(self.angle_diffs[i, 0],
                                   self.angle_diffs[i, 1],
                                   self.angle_diffs[i, 3],
                                   rij, rcov, conv)
            h0[n] -= self._h0_angle(self.angle_diffs[i, 2],
                                    self.angle_diffs[i, 1],
                                    self.angle_diffs[i, 3],
                                    rij, rcov, conv)
            n += 1

        return np.diag(h0_np)


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _h0_bond(CartToInternal self, int a, int b, double[:, :] rij,
                         double[:] rcov, double conv, double Ab=0.3601,
                         double Bb=1.944) nogil:
        return Ab * exp(-Bb * (rij[a, b] - rcov[a] - rcov[b])) * conv


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _h0_angle(CartToInternal self, int a, int b, int c,
                          double[:, :] rij, double[:] rcov, double conv,
                          double Aa=0.089, double Ba=0.11, double Ca=0.44,
                          double Da=-0.42) nogil:
        cdef double rcovab = rcov[a] + rcov[b]
        cdef double rcovbc = rcov[b] + rcov[c]
        return ((Aa + Ba * exp(-Ca * (rij[a, b] + rij[b, c] - rcovab - rcovbc))
                 / (rcovab * rcovbc)**Da) * conv)


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _h0_dihedral(CartToInternal self, int a, int b, int c, int d,
                             int[:] nbonds, double[:, :] rij, double[:] rcov,
                             double conv, double At=0.0015, double Bt=14.0,
                             double Ct=2.85, double Dt=0.57,
                             double Et=4.00) nogil:
        cdef double rcovbc = rcov[b] + rcov[c]
        cdef int L = nbonds[b] + nbonds[c] - 2
        return ((At + Bt * L**Dt * exp(-Ct * (rij[b, c] - rcovbc))
                 / (rij[b, c] * rcovbc)**Et) * conv)


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_Uext(self, double[:, :] pos):
        with nogil:
            info = self._U_update(pos)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return np.array(self.Uext[:self.nx, :self.next])


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_Uint(self, double[:, :] pos):
        with nogil:
            info = self._U_update(pos)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return np.array(self.Uint[:self.nx, :self.nint])


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_Binv(self, double[:, :] pos):
        with nogil:
            info = self._U_update(pos)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return np.array(self.Binv)

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def dq_wrap(CartToInternal self, double[:] dq):
        dq_out_np = np.zeros_like(dq)
        cdef double[:] dq_out = memoryview(dq_out_np)
        dq_out[:] = dq[:]
        cdef int ncba = self.ncart + self.nbonds + self.nangles
        cdef int i
        with nogil:
            for i in range(self.ndihedrals):
                dq_out[ncba + i] = (dq_out[ncba + i] + pi) % (2 * pi) - pi
        return dq_out_np

    def check_for_bad_internal(CartToInternal self, double[:] q):
        # FIXME: implement actual angle check
        return False


cdef class Constraints(CartToInternal):
    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __cinit__(Constraints self,
                  atoms,
                  double[:] target,
                  *args,
                  bint proj_trans=True,
                  bint proj_rot=True,
                  **kwargs):
        self.target = memoryview(np.zeros(self.nq, dtype=np.float64))
        self.target[:] = target[:]
        self.proj_trans = proj_trans
        self.proj_rot = proj_rot
        self.rot_axes = memoryview(np.eye(3, dtype=np.float64))
        self.rot_center = memoryview(atoms.positions.mean(0))

        cdef int npbc = atoms.pbc.sum()
        cdef int i
        if self.proj_rot:
            if npbc == 0:
                self.nrot = 3
            elif npbc == 1:
                self.nrot = 1
                for i, dim in enumerate(atoms.pbc):
                    if dim:
                        self.rot_axes[0, :] = atoms.cell[dim, :]
                normalize(self.rot_axes[0])
            else:
                self.proj_rot = False
                self.nrot = 0
        else:
            self.nrot = 0

        if self.nrot > 0 and self.ncart > 0:
            warnings.warn("Projection of rotational degrees of freedom is not "
                          "currently implemented for systems with fixed atom "
                          "constraints. Disabling projection of rotational "
                          "degrees of freedom.")
            self.proj_rot = False
            self.nrot = 0

        # Ured is fixed, so we initialize it here
        fixed_np = np.zeros((self.natoms, 3), dtype=np.uint8)
        cdef uint8_t[:, :] fixed = memoryview(fixed_np)

        self.Ured = memoryview(np.zeros((self.nx, self.nx - self.ncart),
                                        dtype=np.float64))
        self.trans_dirs = memoryview(np.zeros(3, dtype=np.uint8))
        self.tvecs = memoryview(np.zeros((3, self.nx), dtype=np.float64))

        cdef int n, j
        cdef double invsqrtnat = sqrt(1. / self.natoms)
        with nogil:
            if self.proj_trans:
                for j in range(3):
                    self.trans_dirs[j] = True
            for n in range(self.ncart):
                i = self.cart[n, 0]
                j = self.cart[n, 1]
                fixed[i, j] = True
                self.trans_dirs[j] = False
            n = 0
            for i in range(self.natoms):
                for j in range(3):
                    if fixed[i, j]:
                        continue
                    self.Ured[3 * i + j, n] = 1.
                    n += 1

            self.ntrans = 0
            for j in range(3):
                if self.trans_dirs[j]:
                    for i in range(self.natoms):
                        self.tvecs[self.ntrans, 3 * i + j] = invsqrtnat
                    self.ntrans += 1

        self.ninternal = self.nq
        self.nq += self.ntrans + self.nrot

    def __init__(Constraints self,
                 atoms,
                 double[:] target,
                 *args,
                 **kwargs):
        CartToInternal.__init__(self, atoms, *args, **kwargs)
        self.rot_vecs = memoryview(np.zeros((3, 3), dtype=np.float64))

        self.rvecs = memoryview(np.zeros((3, self.nx), dtype=np.float64))
        self.calc_res = False

        self.q1 = memoryview(np.zeros(self.nq, dtype=np.float64))
        self.dq = memoryview(np.zeros((self.nq, self.natoms, 3),
                                      dtype=np.float64))
        self.res = memoryview(np.zeros(self.nq, dtype=np.float64))

        if self.proj_rot:
            self.rot_axes = memoryview(np.eye(3, dtype=np.float64))
            self.center = memoryview(atoms.positions.mean(0))

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _update(Constraints self,
                     double[:, :] pos,
                     bint grad=False,
                     bint curv=False,
                     bint force=False) nogil:
        if not self.calc_required and not force:
            if (self.grad or not grad) and (self.curv or not curv):
                if not self.geom_changed(pos):
                    return 0

        self.calc_res = False
        memset(&self.res[0], 0, self.nq * sizeof(double))
        cdef int err = CartToInternal._update(self, pos, grad, curv, True)
        if err != 0:
            self.calc_required = True
            return err

        if not self.grad:
            return 0

        cdef int i
        cdef int sddq = self.dq.strides[2] >> 3
        cdef int sdt = self.tvecs.strides[1] >> 3
        cdef int ntot = self.ntrans * self.nx
        dcopy(&ntot, &self.tvecs[0, 0], &sdt,
              &self.dq[self.ninternal, 0, 0], &sddq)

        err = self.project_rotation()
        if err != 0: return err

        cdef int sdr = self.rvecs.strides[1] >> 3
        ntot = self.nrot * self.nx
        dcopy(&ntot, &self.rvecs[0, 0], &sdr,
              &self.dq[self.ninternal + self.ntrans, 0, 0], &sddq)

        return 0


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int project_rotation(Constraints self) nogil:
        cdef int i, j
        for i in range(self.nrot):
            for j in range(self.natoms):
                cross(self.rot_axes[i], self.pos[j],
                      self.rvecs[i, 3*j : 3*(j+1)])
        cdef int err = mgs(self.rvecs[:self.nrot, :].T, self.tvecs.T)
        if err < 0: return err
        if err < self.nrot: return -1
        return 0


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_res(self, double[:, :] pos):
        cdef int err
        err = self._update(pos, False, False)
        if self.calc_res:
            return np.asarray(self.res)
        cdef int i, n
        with nogil:
            for i in range(self.ninternal):
                self.res[i] = self.q1[i] - self.target[i]

            n = self.ncart + self.nbonds + self.nangles
            # Dihedrals are periodic on the range -pi to pi
            for i in range(self.ndihedrals):
                self.res[n + i] = (pi + self.res[n + i]) % (2 * pi) - pi
        self.calc_res = True
        return np.asarray(self.res)

    def get_drdx(self, double[:, :] pos):
        return self.get_B(pos)

    def get_Ured(self):
        return np.asarray(self.Ured)

    def get_Ucons(self, double[:, :] pos):
        return self.get_Uint(pos)

    def get_Ufree(self, double[:, :] pos):
        return self.get_Uext(pos)

    def get_D(self, atoms):
        raise NotImplementedError

    def guess_hessian(self, atoms, double h0cart=70.):
        raise NotImplementedError


cdef class D2q:
    def __cinit__(D2q self,
                  int natoms,
                  int ncart,
                  int[:, :] bonds,
                  int[:, :] angles,
                  int[:, :] dihedrals,
                  int[:, :] angle_sums,
                  int[:, :] angle_diffs,
                  double[:, :, :, :, :] Dbonds,
                  double[:, :, :, :, :] Dangles,
                  double[:, :, :, :, :] Ddihedrals,
                  double[:, :, :, :, :] Dangle_sums,
                  double[:, :, :, :, :] Dangle_diffs):
        self.natoms = natoms
        self.nx = 3 * self.natoms

        self.ncart = ncart

        self.bonds = bonds
        self.nbonds = len(self.bonds)

        self.angles = angles
        self.nangles = len(self.angles)

        self.dihedrals = dihedrals
        self.ndihedrals = len(self.dihedrals)

        self.angle_sums = angle_sums
        self.nangle_sums = len(self.angle_sums)

        self.angle_diffs = angle_diffs
        self.nangle_diffs = len(self.angle_diffs)

        self.nq = (self.nbonds + self.nangles + self.ndihedrals
                   + self.nangle_sums + self.nangle_diffs)

        self.Dbonds = memoryview(np.zeros((self.nbonds, 2, 3, 2, 3),
                                          dtype=np.float64))
        self.Dbonds[...] = Dbonds

        self.Dangles = memoryview(np.zeros((self.nangles, 3, 3, 3, 3),
                                           dtype=np.float64))
        self.Dangles[...] = Dangles

        self.Ddihedrals = memoryview(np.zeros((self.ndihedrals, 4, 3, 4, 3),
                                              dtype=np.float64))
        self.Ddihedrals[...] = Ddihedrals

        self.Dangle_sums = memoryview(np.zeros((self.nangle_sums, 4, 3, 4, 3),
                                               dtype=np.float64))
        self.Dangle_sums[...] = Dangle_sums

        self.Dangle_diffs = memoryview(np.zeros((self.nangle_diffs, 4, 3, 4, 3),
                                                dtype=np.float64))
        self.Dangle_diffs[...] = Dangle_diffs

        self.work1 = memoryview(np.zeros((4, 3), dtype=np.float64))
        self.work2 = memoryview(np.zeros((4, 3), dtype=np.float64))
        self.work3 = memoryview(np.zeros((4, 3), dtype=np.float64))

        self.sw1 = self.work1.strides[1] >> 3
        self.sw2 = self.work2.strides[1] >> 3
        self.sw3 = self.work3.strides[1] >> 3


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def ldot(self, double[:] v1):
        cdef size_t m = self.ncart

        result_np = np.zeros((self.natoms, 3, self.natoms, 3),
                             dtype=np.float64)
        cdef double[:, :, :, :] res = memoryview(result_np)
        with nogil:
            self._ld(m, self.nbonds, 2, self.bonds, self.Dbonds, v1, res)
            m += self.nbonds
            self._ld(m, self.nangles, 3, self.angles, self.Dangles, v1, res)
            m += self.nangles
            self._ld(m, self.ndihedrals, 4, self.dihedrals, self.Ddihedrals,
                     v1, res)
            m += self.ndihedrals
            self._ld(m, self.nangle_sums, 4, self.angle_sums, self.Dangle_sums,
                     v1, res)
            m += self.nangle_sums
            self._ld(m, self.nangle_diffs, 4, self.angle_diffs,
                     self.Dangle_diffs, v1, res)
        return result_np.reshape((self.nx, self.nx))

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _ld(D2q self,
                 size_t start,
                 size_t nq,
                 size_t nind,
                 int[:, :] q,
                 double[:, :, :, :, :] D2,
                 double[:] v,
                 double[:, :, :, :] res) nogil:
        cdef int err
        cdef size_t n, i, a, ai, b, bi

        for n in range(nq):
            for a in range(nind):
                ai = q[n, a]
                for b in range(nind):
                    bi = q[n, b]
                    for i in range(3):
                        err = my_daxpy(v[start + n], D2[n, a, i, b],
                                       res[ai, i, bi])
                        if err != 0:
                            return err
        return 0


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def rdot(self, double[:] v1):
        cdef size_t m = self.ncart

        result_np = np.zeros((self.nq, self.natoms, 3), dtype=np.float64)
        cdef double[:, :, :] res = memoryview(result_np)
        with nogil:
            self._rd(m, self.nbonds, 2, self.bonds, self.Dbonds, v1, res)
            m += self.nbonds
            self._rd(m, self.nangles, 3, self.angles, self.Dangles, v1, res)
            m += self.nangles
            self._rd(m, self.ndihedrals, 4, self.dihedrals, self.Ddihedrals,
                     v1, res)
            m += self.ndihedrals
            self._rd(m, self.nangle_sums, 4, self.angle_sums, self.Dangle_sums,
                     v1, res)
            m += self.nangle_sums
            self._rd(m, self.nangle_diffs, 4, self.angle_diffs,
                     self.Dangle_diffs, v1, res)
        return result_np.reshape((self.nq, self.nx))


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _rd(D2q self,
                 size_t start,
                 size_t nq,
                 size_t nind,
                 int[:, :] q,
                 double[:, :, :, :, :] D2,
                 double[:] v,
                 double[:, :, :] res) nogil:
        cdef size_t n, a, ai
        cdef int sv = v.strides[0] >> 3
        cdef int sres = res.strides[2] >> 3
        cdef int dim = 3 * nind
        cdef int ldD2 = dim * (D2.strides[4] >> 3)
        for n in range(nq):
            for a in range(nind):
                ai = q[n, a]
                self.work1[a, :] = v[3*ai : 3*(ai+1)]
            dgemv('N', &dim, &dim, &DUNITY, &D2[n, 0, 0, 0, 0], &ldD2,
                  &self.work1[0, 0], &self.sw1, &DZERO,
                  &self.work2[0, 0], &self.sw2)
            for a in range(nind):
                ai = q[n, a]
                res[start + n, ai, :] = self.work2[a, :]
        return 0


    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def ddot(self, double[:] v1, double[:] v2):
        cdef size_t m = self.ncart

        result_np = np.zeros(self.nq, dtype=np.float64)
        cdef double[:] res = memoryview(result_np)
        with nogil:
            self._dd(m, self.nbonds, 2, self.bonds, self.Dbonds, v1, v2, res)
            m += self.nbonds
            self._dd(m, self.nangles, 3, self.angles, self.Dangles, v1, v2,
                     res)
            m += self.nangles
            self._dd(m, self.ndihedrals, 4, self.dihedrals, self.Ddihedrals,
                     v1, v2, res)
            m += self.ndihedrals
            self._dd(m, self.nangle_sums, 4, self.angle_sums, self.Dangle_sums,
                     v1, v2, res)
            m += self.nangle_sums
            self._dd(m, self.nangle_diffs, 4, self.angle_diffs,
                     self.Dangle_diffs, v1, v2, res)
        return result_np

    #def ddot(self, double[:] v1, v2):
    #    return self.rdot(v1) @ v2

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _dd(D2q self,
                 size_t start,
                 size_t nq,
                 size_t nind,
                 int[:, :] q,
                 double[:, :, :, :, :] D2,
                 double[:] v1,
                 double[:] v2,
                 double[:] res) nogil:
        cdef size_t n, a, ai
        cdef int sv1 = v1.strides[0] >> 3
        cdef int sv2 = v2.strides[0] >> 3
        cdef int dim = 3 * nind
        cdef int sD2 = dim * (D2.strides[4] >> 3)
        for n in range(nq):
            for a in range(nind):
                ai = q[n, a]
                self.work1[a, :] = v1[3*ai : 3*(ai+1)]
                self.work2[a, :] = v2[3*ai : 3*(ai+1)]
            dgemv('N', &dim, &dim, &DUNITY, &D2[n, 0, 0, 0, 0], &sD2,
                  &self.work1[0, 0], &self.sw1, &DZERO,
                  &self.work3[0, 0], &self.sw3)
            res[start + n] = ddot(&dim, &self.work2[0, 0], &self.sw2,
                                  &self.work3[0, 0], &self.sw3)
        return 0
