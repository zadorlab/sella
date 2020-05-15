# cimports

from libc.math cimport fabs, exp, pi, sqrt, copysign, acos
from libc.string cimport memset
from libc.stdint cimport uint8_t

from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dgemv, dcopy

from sella.utilities.blas cimport my_ddot, my_daxpy, my_dgemv, my_dnrm2
from sella.utilities.math cimport (vec_sum, mgs, cross,
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

cdef inline double _h0_bond(double rab, double rcovab, double conv,
                            double Ab=0.3601, double Bb=1.944) nogil:
    return Ab * exp(-Bb * (rab - rcovab)) * conv


cdef inline double _h0_angle(double rab, double rbc, double rcovab,
                             double rcovbc, double conv, double Aa=0.089,
                             double Ba=0.11, double Ca=0.44,
                             double Da=-0.42) nogil:
    return ((Aa + Ba * exp(-Ca * (rab + rbc - rcovab - rcovbc))
             / (rcovab * rcovbc)**Da) * conv)


cdef inline double _h0_dihedral(double rbc, double rcovbc, int L, double conv,
                                double At=0.0015, double Bt=14.0,
                                double Ct=2.85, double Dt=0.57,
                                double Et=4.00) nogil:
    return ((At + Bt * L**Dt * exp(-Ct * (rbc - rcovbc))
             / (rbc * rcovbc)**Et) * conv)


cdef class CartToInternal:
    def __init__(self, atoms, *args, dummies=None, **kwargs):
        if self.nreal <= 0:
            raise ValueError("Must have at least 1 atom!")
        self.nmin = min(self.nq, self.nx)
        self.nmax = max(self.nq, self.nx)

        # Allocate arrays
        #
        # We use memoryview and numpy to initialize these arrays, because
        # some of the arrays may have a dimension of size 0, which
        # cython.view.array does not permit.
        self.pos = memoryview(np.zeros((self.natoms, 3), dtype=np.float64))
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
                  uint8_t[:] bulklike=None,
                  int[:] tneg=None,
                  double[:, :] cellvecs=None,
                  dummies=None,
                  int[:] dinds=None,
                  double atol=15,
                  **kwargs):

        cdef int i, a, b, n
        cdef size_t sd = sizeof(double)

        self.nreal = len(atoms)
        if self.nreal <= 0:
            return

        if dummies is None:
            self.ndummies = 0
        else:
            self.ndummies = len(dummies)

        self.natoms = self.nreal + self.ndummies

        self.rcov = memoryview(np.zeros(self.natoms, dtype=np.float64))
        for i in range(self.nreal):
            self.rcov[i] = covalent_radii[atoms.numbers[i]].copy()
        for i in range(self.ndummies):
            self.rcov[self.nreal + i] = covalent_radii[0].copy()

        self.cellvecs = cellvecs
        self.tneg = tneg

        bmat_np = np.zeros((self.natoms, self.natoms, len(self.cellvecs)),
                           dtype=np.uint8)
        self.bmat = memoryview(bmat_np)

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

        for i in range(self.nbonds):
            a = self.bonds[i, 0]
            b = self.bonds[i, 1]
            n = self.bonds[i, 2]
            self.bmat[a, b, n] = self.bmat[b, a, self.tneg[n]] = True

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

        if bulklike is None:
            self.bulklike = memoryview(np.zeros(self.nreal, dtype=np.uint8))
        else:
            self.bulklike = bulklike

        if dinds is None:
            self.dinds = memoryview(-np.ones(self.nreal, dtype=np.int32))
        else:
            self.dinds = dinds

        self.nq = self.ncart + self.nbonds + self.nangles + self.ndihedrals

        self.atol = pi * atol / 180.
        self.tneg = tneg


    def get_q(self, double[:, :] pos, double[:, :] dummypos=None):
        cdef int info
        with nogil:
            info = self._update(pos, dummypos, False, False)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        # We use np.array() instead of np.asarray() because we want to
        # return a copy.
        return np.array(self.q1)

    def get_B(self, double[:, :] pos, double[:, :] dummypos=None):
        cdef int info
        with nogil:
            info = self._update(pos, dummypos, True, False)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return np.array(self.dq).reshape((self.nq, self.nx))

    def get_D(self, double[:, :] pos, double[:, :] dummypos=None):
        cdef int info
        with nogil:
            info = self._update(pos, dummypos, True, True)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return D2q(
            self.natoms, self.ncart, self.bonds, self.angles, self.dihedrals,
            self.d2q_bonds, self.d2q_angles, self.d2q_dihedrals
        )

    cdef bint _validate_pos(CartToInternal self, double[:, :] pos,
                            double[:, :] dummypos=None) nogil:
        cdef int n_in = pos.shape[0]
        if dummypos is not None:
            if dummypos.shape[1] != 3:
                return False
            n_in += dummypos.shape[0]
        else:
            if self.ndummies > 0:
                return False

        if n_in != self.pos.shape[0] or pos.shape[1] != 3:
            return False
        return True

    cdef bint geom_changed(CartToInternal self, double[:, :] pos,
                           double[:, :] dummypos=None) nogil:
        cdef int i, j
        for i in range(self.nreal):
            for j in range(3):
                if self.pos[i, j] != pos[i, j]:
                    return True

        for i in range(self.ndummies):
            for j in range(3):
                if self.pos[self.nreal + i, j] != dummypos[i, j]:
                    return True

        return False

    cdef int _update(CartToInternal self,
                     double[:, :] pos,
                     double[:, :] dummypos=None,
                     bint grad=False,
                     bint curv=False,
                     bint force=False) nogil except -1:
        if not self._validate_pos(pos, dummypos):
            return -1
        # The purpose of this check is to determine whether the positions
        # array has been changed at all since the last internal coordinate
        # evaluation, which is why we are doing exact floating point
        # comparison with ==.
        if not self.calc_required and not force:
            if (self.grad or not grad) and (self.curv or not curv):
                if not self.geom_changed(pos, dummypos):
                    return 0
        self.calc_required = True
        self.grad = grad or curv
        self.curv = curv
        cdef size_t n, m, i, j, k, l, nj, nk, nl
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
        memset(&self.work1[0, 0, 0, 0], 0, 297 * sd)
        memset(&self.work2[0, 0], 0, self.nx * sd)

        self.pos[:self.nreal, :] = pos
        if dummypos is not None:
            self.pos[self.nreal:, :] = dummypos

        for n in range(self.ncart):
            i = self.cart[n, 0]
            j = self.cart[n, 1]
            self.q1[n] = self.pos[i, j]
            self.dq[n, i, j] = 1.
            # d2q is the 0 matrix for cartesian coords

        m = self.ncart
        for n in range(self.nbonds):
            i = self.bonds[n, 0]
            j = self.bonds[n, 1]
            nj = self.bonds[n, 2]
            err = self.get_dx(i, j, nj, self.dx1)

            if err != 0: return err

            info = cart_to_bond(i, j, self.dx1, &self.q1[m + n],
                                self.dq[m + n], self.d2q_bonds[n], grad, curv)
            if info < 0: return info

        m += self.nbonds
        for n in range(self.nangles):
            i = self.angles[n, 0]
            j = self.angles[n, 1]
            k = self.angles[n, 2]
            nj = self.angles[n, 3]
            nk = self.angles[n, 4]

            err = self.get_dx(i, j, nj, self.dx1)
            if err != 0: return err
            err = self.get_dx(j, k, nk, self.dx2)
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
            nj = self.dihedrals[n, 4]
            nk = self.dihedrals[n, 5]
            nl = self.dihedrals[n, 6]

            err = self.get_dx(i, j, nj, self.dx1)
            if err != 0: return err
            err = self.get_dx(j, k, nk, self.dx2)
            if err != 0: return err
            err = self.get_dx(k, l, nl, self.dx3)
            if err != 0: return err

            info = cart_to_dihedral(i, j, k, l, self.dx1, self.dx2, self.dx3,
                                    &self.q1[m + n], self.dq[m + n],
                                    self.d2q_dihedrals[n], self.work1,
                                    grad, curv)
            if info < 0:
                return info

        self.calc_required = False
        self.nint = -1
        return 0

    cdef int _U_update(CartToInternal self,
                       double[:, :] pos,
                       double[:, :] dummypos=None,
                       bint force=False) nogil except -1:
        cdef int err = self._update(pos, dummypos, True, False, force)
        if err != 0:
            return err

        if self.nint > 0:
            return 0

        cdef int sddq = self.dq.strides[2] >> 3
        cdef int sdu = self.Uint.strides[1] >> 3
        memset(&self.Uint[0, 0], 0, self.nmax * self.nmax * sizeof(double))
        memset(&self.Uext[0, 0], 0, self.nmax * self.nmax * sizeof(double))

        cdef int i
        if self.nq == 0:
            for i in range(self.nx):
                self.Uext[i, i] = 1.
            self.nint = 0
            self.next = self.nx
            return 0

        for n in range(self.nq):
            dcopy(&self.nx, &self.dq[n, 0, 0], &sddq, &self.Uint[n, 0], &sdu)

        self.nint = mppi(self.nq, self.nx, self.Uint, self.Usvd, self.Uext,
                         self.sing, self.Binv, self.work3)

        if self.nint < 0:
            return self.nint
        self.next = self.nx - self.nint

        self.calc_required = False
        return 0

    def guess_hessian(self, atoms, dummies=None, double h0cart=70.):
        if (dummies is None or len(dummies) == 0) and self.ndummies > 0:
            raise ValueError("Must provide dummy atoms!")
        if len(atoms) != self.nreal:
            raise ValueError("Provided atoms has the wrong number of atoms! "
                             "Expected {}, got {}.".format(self.nreal,
                                                           len(atoms)))
        if dummies is not None:
            if len(dummies) != self.ndummies:
                raise ValueError("Provided dummies has the wrong number of "
                                 " atoms! Expected {}, got {}."
                                 "".format(self.ndummies, len(dummies)))
            atoms = atoms + dummies
        rcov_np = covalent_radii[atoms.numbers].copy() / units.Bohr
        cdef double[:] rcov = memoryview(rcov_np)
        rij_np = atoms.get_all_distances() / units.Bohr
        cdef double[:, :] rij = memoryview(rij_np)

        nbonds_np = np.zeros(self.natoms, dtype=np.int32)
        cdef int[:] nbonds = memoryview(nbonds_np)

        h0_np = np.zeros(self.nq, np.float64)
        cdef double[:] h0 = memoryview(h0_np)

        cdef int i
        cdef int n
        cdef int a, b, c, d, nb, nc, err
        cdef double Hartree = units.Hartree
        cdef double Bohr = units.Bohr
        cdef double rcovab, rcovbc
        cdef double conv

        # FIXME: for some reason, this fails at runtime when the gil
        # has been released
        with nogil:
            err = 0
            for i in range(self.nbonds):
                nbonds[self.bonds[i, 0]] += 1
                nbonds[self.bonds[i, 1]] += 1

            for i in range(self.nreal):
                if self.dinds[i] != -1:
                    nbonds[i] += 1
                    nbonds[self.dinds[i]] += 1

            n = 0
            for i in range(self.ncart):
                h0[n] = h0cart
                n += 1

            conv = Hartree / Bohr**2
            for i in range(self.nbonds):
                if err != 0: break
                a = self.bonds[i, 0]
                b = self.bonds[i, 1]
                nb = self.bonds[i, 2]
                err = self.get_dx(a, b, nb, self.dx1)
                if err != 0: break
                h0[n] = _h0_bond(my_dnrm2(self.dx1), rcov[a] + rcov[b], conv)
                n += 1

            conv = Hartree
            for i in range(self.nangles):
                if err != 0: break
                a = self.angles[i, 0]
                b = self.angles[i, 1]
                c = self.angles[i, 2]
                nb = self.angles[i, 3]
                nc = self.angles[i, 4]
                err = self.get_dx(a, b, nb, self.dx1)
                if err != 0: break
                err = self.get_dx(b, c, nc, self.dx2)
                if err != 0: break
                h0[n] = _h0_angle(
                    my_dnrm2(self.dx1), my_dnrm2(self.dx2), rcov[a] + rcov[b],
                    rcov[b] + rcov[c], conv
                )
                n += 1

            for i in range(self.ndihedrals):
                if err != 0: break
                b = self.dihedrals[i, 1]
                c = self.dihedrals[i, 2]
                nc = self.dihedrals[i, 5]
                err = self.get_dx(b, c, nc, self.dx1)
                if err != 0: break
                h0[n] = _h0_dihedral(
                    my_dnrm2(self.dx1), rcov[b] + rcov[c],
                    nbonds[b] + nbonds[c] - 2, conv
                )
                n += 1
        if err != 0:
            raise RuntimeError("Model Hessian evaluation failed!")

        return np.diag(np.abs(h0_np))


    cdef double _h0_bond(CartToInternal self, int a, int b, double[:, :] rij,
                         double[:] rcov, double conv, double Ab=0.3601,
                         double Bb=1.944) nogil:
        return Ab * exp(-Bb * (rij[a, b] - rcov[a] - rcov[b])) * conv


    cdef double _h0_angle(CartToInternal self, int a, int b, int c,
                          double[:, :] rij, double[:] rcov, double conv,
                          double Aa=0.089, double Ba=0.11, double Ca=0.44,
                          double Da=-0.42) nogil:
        cdef double rcovab = rcov[a] + rcov[b]
        cdef double rcovbc = rcov[b] + rcov[c]
        return ((Aa + Ba * exp(-Ca * (rij[a, b] + rij[b, c] - rcovab - rcovbc))
                 / (rcovab * rcovbc)**Da) * conv)


    cdef double _h0_dihedral(CartToInternal self, int a, int b, int c, int d,
                             int[:] nbonds, double[:, :] rij, double[:] rcov,
                             double conv, double At=0.0015, double Bt=14.0,
                             double Ct=2.85, double Dt=0.57,
                             double Et=4.00) nogil:
        cdef double rcovbc = rcov[b] + rcov[c]
        cdef int L = nbonds[b] + nbonds[c] - 2
        return ((At + Bt * L**Dt * exp(-Ct * (rij[b, c] - rcovbc))
                 / (rij[b, c] * rcovbc)**Et) * conv)


    def get_Uext(self, double[:, :] pos, double[:, :] dummypos=None):
        with nogil:
            info = self._U_update(pos, dummypos)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return np.array(self.Uext[:self.nx, :self.next])


    def get_Uint(self, double[:, :] pos, double[:, :] dummypos=None):
        with nogil:
            info = self._U_update(pos, dummypos)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return np.array(self.Uint[:self.nx, :self.nint])


    def get_Binv(self, double[:, :] pos, double[:, :] dummypos=None):
        with nogil:
            info = self._U_update(pos, dummypos)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))

        return np.array(self.Binv)

    def dq_wrap(CartToInternal self, double[:] dq):
        dq_out_np = np.zeros_like(dq)
        cdef double[:] dq_out = memoryview(dq_out_np)
        dq_out[:] = dq[:]
        cdef int ncba = self.ncart + self.nbonds + self.nangles
        cdef int i
        cdef int err = 0
        with nogil:
            for i in range(self.ndihedrals):
                dq_out[ncba + i] = (dq_out[ncba + i] + pi) % (2 * pi) - pi
                if dq_out[ncba + i] < -pi:
                    dq_out[ncba + i] += 2 * pi
                if not (-pi < dq_out[ncba + i] < pi):
                    err = 1
                    break
        if err == 1:
            raise RuntimeError('dq_wrap failed unexpectedly!')
        return dq_out_np

    def check_for_bad_internal(CartToInternal self, double[:, :] pos,
                               double[:, :] dummypos):
        cdef int info
        with nogil:
            info = self._update(pos, dummypos, False, False)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}"
                               "".format(info))
        cdef int start
        cdef int i, j, n
        cdef bint bad = False
        dx_np = np.zeros(3, dtype=np.float64)
        cdef double[:] dx = memoryview(dx_np)
        cdef int sddx = dx.strides[0] >> 3
        cdef double dist, rcovij
        cdef double dxmin
        bond_check_np = np.zeros((self.nbonds, 3), dtype=np.int32)
        cdef int[:, :] bond_check = memoryview(bond_check_np)
        cdef int nbond_check = 0

        angle_check_np = np.zeros((self.nangles, 5), dtype=np.int32)
        cdef int[:, :] angle_check = memoryview(angle_check_np)
        cdef int nangle_check = 0

        dihedral_check_np = np.zeros((self.ndihedrals, 7), dtype=np.int32)
        cdef int[:, :] dihedral_check = memoryview(dihedral_check_np)
        cdef int ndihedral_check = 0

        work_np = np.zeros((2, 3), dtype=np.float64)
        cdef double[:, :] work = memoryview(work_np)

        with nogil:
            # Check whether there are any linear angles
            start = self.ncart + self.nbonds
            for i in range(self.nangles):
                if not self.check_angle(self.q1[start + i]):
                    angle_check[nangle_check, :] = self.angles[i]
                    nangle_check += 1
                    bad = True

            # Check whether any atoms are too close, using a different
            # threshold for bonded and nonbonded atoms.
            for i in range(self.nreal):
                for j in range(i, self.nreal):
                    if self.bulklike[i] and self.bulklike[j]:
                        continue
                    for n in range(len(self.cellvecs)):
                        if i == j and n == 0:
                            continue
                        if self.bmat[i, j, n]:
                            dxmin = 0.5
                        else:
                            dxmin = 1.25
                        info = self.get_dx(i, j, n, dx)
                        if info != 0:  break
                        dist = dnrm2(&THREE, &dx[0], &sddx)
                        rcovij = self.rcov[i] + self.rcov[j]
                        if dist < dxmin * rcovij:
                            if self.bmat[i, j, n]:
                                bond_check[nbond_check, 0] = i
                                bond_check[nbond_check, 1] = j
                                bond_check[nbond_check, 2] = n
                                nbond_check += 1
                            bad = True
                if info != 0:  break

            ndihedral_check = self.check_dihedrals(dihedral_check, work)
            if ndihedral_check < 0:
                info = ndihedral_check
            elif ndihedral_check > 0:
                bad = True

        if info != 0:
            raise RuntimeError("Failed while checking for bad internals!")
        if bad:
            check = {'bonds': bond_check_np[:nbond_check],
                     'angles': angle_check_np[:nangle_check],
                     'dihedrals': dihedral_check_np[:ndihedral_check]}
            return check
        return None

    cdef bint check_angle(CartToInternal self, double angle) nogil:
        return (self.atol < angle < pi - self.atol)

    cdef int get_dx(CartToInternal self, int i, int j, int nj,
                    double[:] dx) nogil:
        err = vec_sum(self.pos[j], self.pos[i], dx, -1.)
        if err != 0: return err
        return my_daxpy(1., self.cellvecs[nj], dx)

    cdef int check_dihedrals(CartToInternal self,
                             int[:, :] dihedral_check,
                             double[:, :] work) nogil:
        cdef int ndihedral_check = 0
        cdef int n, m, i, j, k, l, nj, nk
        cdef double adotb, anorm, bnorm, angle
        for n in range(self.ndihedrals):
            for m in range(2):
                i = self.dihedrals[n, m + 0]
                j = self.dihedrals[n, m + 1]
                k = self.dihedrals[n, m + 2]
                nj = self.dihedrals[n, m + 4]
                nk = self.dihedrals[n, m + 5]

                info = self.get_dx(i, j, nj, work[0])
                if info != 0:  break
                info = self.get_dx(k, j, self.tneg[nk], work[1])
                if info != 0:  break
                adotb = my_ddot(work[0], work[1])
                anorm = my_dnrm2(work[0])
                bnorm = my_dnrm2(work[1])
                angle = acos(adotb / (anorm * bnorm))
                if not self.check_angle(angle):
                    dihedral_check[ndihedral_check, :] = self.dihedrals[n]
                    ndihedral_check += 1
            if info != 0:  break
        if info != 0:
            return info

        return ndihedral_check


cdef class Constraints(CartToInternal):
    def __cinit__(Constraints self,
                  atoms,
                  double[:] target,
                  *args,
                  bint proj_trans=True,
                  bint proj_rot=True,
                  **kwargs):
        self.target = memoryview(np.zeros(self.nq, dtype=np.float64))
        self.target[:len(target)] = target[:]
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

        if self.ntrans == 0:
            self.proj_trans = False

        self.ninternal = self.nq
        self.nq += self.nrot + self.ntrans

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

    cdef int _update(Constraints self,
                     double[:, :] pos,
                     double[:, :] dummypos=None,
                     bint grad=False,
                     bint curv=False,
                     bint force=False) nogil except -1:
        if not self._validate_pos(pos, dummypos):
            return -1
        if not self.calc_required and not force:
            if (self.grad or not grad) and (self.curv or not curv):
                if not self.geom_changed(pos, dummypos):
                    return 0

        cdef int i, err, sddq, sdt, ntot, sdr
        self.calc_res = False
        if self.nq == 0:
            return 0
        memset(&self.res[0], 0, self.nq * sizeof(double))
        err = CartToInternal._update(self, pos, dummypos, grad,
                                     curv, True)
        if err != 0:
            self.calc_required = True
            return err

        if not self.grad:
            return 0

        if self.proj_trans:
            sddq = self.dq.strides[2] >> 3
            sdt = self.tvecs.strides[1] >> 3
            ntot = self.ntrans * self.nx
            dcopy(&ntot, &self.tvecs[0, 0], &sdt,
                  &self.dq[self.ninternal, 0, 0], &sddq)

        if self.proj_rot:
            err = self.project_rotation()
            if err != 0: return err

            sdr = self.rvecs.strides[1] >> 3
            ntot = self.nrot * self.nx
            dcopy(&ntot, &self.rvecs[0, 0], &sdr,
                  &self.dq[self.ninternal + self.ntrans, 0, 0], &sddq)

        return 0


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


    def get_res(self, double[:, :] pos, double[:, :] dummypos=None):
        cdef int info
        with nogil:
            info = self._update(pos, dummypos, False, False)
        if info < 0:
            raise RuntimeError("Internal update failed with error code {}",
                               "".format(info))
        if self.calc_res:
            return np.asarray(self.res)
        cdef int i, n
        with nogil:
            for i in range(self.ninternal):
                self.res[i] = self.q1[i] - self.target[i]

            n = self.ncart + self.nbonds + self.nangles
            # Dihedrals are periodic on the range -pi to pi
            for i in range(self.ndihedrals):
                self.res[n + i] = (pi + self.res[n + i]) % (2 * pi) #- pi
                self.res[n + i] -= copysign(pi, self.res[n + i])
        self.calc_res = True
        return np.asarray(self.res)

    def get_drdx(self, double[:, :] pos, double[:, :] dummypos=None):
        return self.get_B(pos, dummypos)

    def get_Ured(self):
        return np.asarray(self.Ured)

    def get_Ucons(self, double[:, :] pos, double[:, :] dummypos=None):
        return self.get_Uint(pos, dummypos)

    def get_Ufree(self, double[:, :] pos, double[:, :] dummypos=None):
        return self.get_Uext(pos, dummypos)

    def guess_hessian(self, atoms, double h0cart=70.):
        raise NotImplementedError


cdef class D2q:
    def __cinit__(
        D2q self,
        int natoms,
        int ncart,
        int[:, :] bonds,
        int[:, :] angles,
        int[:, :] dihedrals,
        double[:, :, :, :, :] Dbonds,
        double[:, :, :, :, :] Dangles,
        double[:, :, :, :, :] Ddihedrals
    ):
        self.natoms = natoms
        self.nx = 3 * self.natoms

        self.ncart = ncart

        self.bonds = bonds
        self.nbonds = len(self.bonds)

        self.angles = angles
        self.nangles = len(self.angles)

        self.dihedrals = dihedrals
        self.ndihedrals = len(self.dihedrals)

        self.nq = self.ncart + self.nbonds + self.nangles + self.ndihedrals

        self.Dbonds = memoryview(np.zeros((self.nbonds, 2, 3, 2, 3),
                                          dtype=np.float64))
        self.Dbonds[...] = Dbonds

        self.Dangles = memoryview(np.zeros((self.nangles, 3, 3, 3, 3),
                                           dtype=np.float64))
        self.Dangles[...] = Dangles

        self.Ddihedrals = memoryview(np.zeros((self.ndihedrals, 4, 3, 4, 3),
                                              dtype=np.float64))
        self.Ddihedrals[...] = Ddihedrals

        self.work1 = memoryview(np.zeros((4, 3), dtype=np.float64))
        self.work2 = memoryview(np.zeros((4, 3), dtype=np.float64))
        self.work3 = memoryview(np.zeros((4, 3), dtype=np.float64))

        self.sw1 = self.work1.strides[1] >> 3
        self.sw2 = self.work2.strides[1] >> 3
        self.sw3 = self.work3.strides[1] >> 3


    def ldot(self, double[:] v1):
        cdef size_t m = self.ncart
        #assert len(v1) == self.nq, (len(v1), self.nq)

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
        return result_np.reshape((self.nx, self.nx))

    cdef int _ld(D2q self,
                 size_t start,
                 size_t nq,
                 size_t nind,
                 int[:, :] q,
                 double[:, :, :, :, :] D2,
                 double[:] v,
                 double[:, :, :, :] res) nogil except -1:
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

    def rdot(self, double[:] v1):
        cdef size_t m = self.ncart
        assert len(v1) == self.nx

        result_np = np.zeros((self.nq, self.natoms, 3), dtype=np.float64)
        cdef double[:, :, :] res = memoryview(result_np)
        with nogil:
            self._rd(m, self.nbonds, 2, self.bonds, self.Dbonds, v1, res)
            m += self.nbonds
            self._rd(m, self.nangles, 3, self.angles, self.Dangles, v1, res)
            m += self.nangles
            self._rd(m, self.ndihedrals, 4, self.dihedrals, self.Ddihedrals,
                     v1, res)
        return result_np.reshape((self.nq, self.nx))


    cdef int _rd(D2q self,
                 size_t start,
                 size_t nq,
                 size_t nind,
                 int[:, :] q,
                 double[:, :, :, :, :] D2,
                 double[:] v,
                 double[:, :, :] res) nogil except -1:
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

    def ddot(self, double[:] v1, double[:] v2):
        cdef size_t m = self.ncart
        assert len(v1) == self.nx
        assert len(v2) == self.nx

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
        return result_np

    cdef int _dd(D2q self,
                 size_t start,
                 size_t nq,
                 size_t nind,
                 int[:, :] q,
                 double[:, :, :, :, :] D2,
                 double[:] v1,
                 double[:] v2,
                 double[:] res) nogil except -1:
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
