#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from .internal_cython import get_internal, cart_to_internal

from ase.data import covalent_radii, vdw_radii
from ase.units import Hartree, Bohr

class Internal:
    def __init__(self, atoms, angles=True, dihedrals=True, extra_bonds=None):
        if extra_bonds is None:
            extra_bonds = []

        out = get_internal(atoms, angles, dihedrals, extra_bonds)
        self.c10y, self.nbonds, self.bonds, self.angles, self.dihedrals = out

        self.n = dict(bonds=len(self.bonds),
                      angles=len(self.angles),
                      dihedrals=len(self.dihedrals))
        self.n.update(total=sum(self.n.values()))

        self.last = dict(x=None, q=None, B=None, U=None, Binv=None, D=None)

    def _update(self, pos, gradient=False, curvature=False):
        pos = pos.reshape((-1, 3))
        if np.all(pos == self.last['x']):
            if gradient and (self.last['B'] is None):
                pass
            elif curvature and (self.last['D'] is None):
                pass
            else:
                return

        mask = np.ones(self.n['total'], dtype=np.uint8)

        q, B, D = cart_to_internal(pos,
                                   self.bonds,
                                   self.angles,
                                   self.dihedrals,
                                   mask,
                                   gradient,
                                   curvature)
        if gradient:
            Binv = np.linalg.pinv(B)
            G = B @ B.T
            lams, vecs = eigh(G)
            indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-8]
            U = vecs[:, indices]
        else:
            B = None
            Binv = None
            U = None

        if not curvature:
            D = None

        self.last = dict(x=pos.copy(), q=q, B=B, U=U, Binv=Binv, D=D)

    def q(self, pos):
        self._update(pos)
        return self.last['q']

    def B(self, pos):
        self._update(pos, True)
        return self.last['B']

    def U(self, pos):
        self._update(pos, True)
        return self.last['U']

    def Binv(self, pos):
        self._update(pos, True)
        return self.last['Binv']

    def D(self, pos):
        self._update(pos, True, True)
        return self.last['D']

    def q_wrap(self, vec):
        nba = self.n['bonds'] + self.n['angles']
        out = vec.copy()
        out[nba:] = (out[nba:] + np.pi) % (2 * np.pi) - np.pi
        return out

    def guess_hessian(self, atoms):
        rcov = covalent_radii[atoms.numbers]

        # Stretch parameters
        Ab = 0.3601 * Hartree / Bohr**2
        Bb = 1.944 / Bohr

        # Bend parameters
        Aa = 0.089 * Hartree
        Ba = 0.11 * Hartree
        Ca = 0.44 / Bohr
        Da = -0.42

        # Torsion parameters
        At = 0.0015 * Hartree
        Bt = 14.0 * Hartree
        Ct = 2.85 / Bohr
        Dt = 0.57
        Et = 4.00

        h0 = np.zeros(self.n['total'])

        n = 0
        start = 0
        for i, (a, b) in enumerate(self.bonds):
            rab = atoms.get_distance(a, b)
            rcovab = rcov[a] + rcov[b]
            h0[n] = Aa * np.exp(-Bb * (rab - rcovab))
            n += 1

        start += self.n['bonds']
        for i, (a, b, c) in enumerate(self.angles):
            rab = atoms.get_distance(a, b)
            rbc = atoms.get_distance(b, c)
            rcovab = rcov[a] + rcov[b]
            rcovbc = rcov[b] + rcov[c]
            h0[n] = Aa + Ba * np.exp(-Ca * (rab + rbc - rcovab - rcovbc)) / (rcovab * rcovbc / Bohr**2)**Da
            n += 1

        start += self.n['angles']
        for i, (a, b, c, d) in enumerate(self.dihedrals):
            rbc = atoms.get_distance(b, c)
            rcovbc = rcov[b] + rcov[c]
            L = self.nbonds[b] + self.nbonds[c] - 2
            h0[n] = At + Bt * L**Dt * np.exp(-Ct * (rbc - rcovbc)) / (rbc * rcovbc / Bohr**2)**Et
            n += 1

        H0 = np.diag(h0)
        P = self.B(atoms.positions) @ self.Binv(atoms.positions)
        return P.T @ H0 @ P



#class Internal(object):
#    def __init__(self, atoms, angles=True, dihedrals=True, extra_bonds=None):
#        self.extra_bonds = []
#        if extra_bonds is not None:
#            self.extra_bonds = extra_bonds
#        self.use_angles = angles
#        self.use_dihedrals = dihedrals
#        self._mask = None
#
#        self.atoms = atoms
#        self._pos_old = np.zeros_like(self.atoms.get_positions())
#
#        self.path = None
#        self.v0 = None
#        self.v1 = None
#
#
#    def get_internal(self):
#        self.c10y, self.nbonds, self.bonds, self.angles, self.dihedrals = get_internal(
#                self._atoms, self.use_angles, self.use_dihedrals, self.extra_bonds)
#        self.nb = len(self.bonds)
#        if self.use_angles:
#            self.na = len(self.angles)
#        else:
#            self.na = 0
#
#        if self.use_dihedrals:
#            self.nd = len(self.dihedrals)
#        else:
#            self.nd = 0
#
#        self.ninternal = self.nb + self.na + self.nd
#        self._mask = np.ones(self.ninternal, dtype=np.uint8)
#
#        self._p = None
#        self._U = None
#        self._B = None
#        self._Binv = None
#        self._D = None
#
#    @property
#    def atoms(self):
#        return self._atoms
#
#    @atoms.setter
#    def atoms(self, new_atoms):
#        self._atoms = new_atoms
#        self.natoms = len(self._atoms)
#
#        self.get_internal()
#
#        # Index of atom to be placed at the origin
#        for ind0 in range(self.natoms):
#            # Index of atom to be placed along X axis
#            ind1 = min(self.c10y[ind0, :self.nbonds[ind0]])
#            tmp = set(self.c10y[ind1, :self.nbonds[ind1]])
#            tmp.discard(ind0)
#            if tmp:
#                # Index of atom to be placed in X-Y plane
#                ind2 = min(tmp)
#                break
#
#    def _calc_internal(self, gradient=False, curvature=False):
#        mask = np.ones(self.ninternal, dtype=np.uint8)
#        self._p, _, _ = cart_to_internal(self.atoms.positions,
#                                         self.bonds,
#                                         self.angles,
#                                         self.dihedrals,
#                                         mask)
#
#        added = True
#        # Check for near-collinear angles and make sure the terminal atoms are
#        # considered connected. This is done iteratively.
#        while added:
#            added = False
#            for i, thetai in enumerate(self._p[self.nb:self.nb+self.na]):
#                # Check to see if an angle is near collinear
#                if np.pi - thetai < np.pi / 20:
#                #if False:
#                    a, b, c = self.angles[i]
#                    # If it is, check to make sure the two terminal atoms are bound.
#                    # If they aren't, add a bond explicitly.
#                    for j, k in self.bonds:
#                        if j == a and k == c:
#                            break
#                    else:
#                        self.extra_bonds.append(tuple(sorted((a, c))))
#                        added = True
#            if added:
#                self.get_internal()
#                self._p, _, _ = cart_to_internal(self.atoms.positions,
#                                                 self.bonds,
#                                                 self.angles,
#                                                 self.dihedrals,
#                                                 self._mask)
#        # Next, identify all angles near 180 degrees.
#        collinear = []
#        self._mask = np.ones(self.ninternal, dtype=np.uint8)
#        for i, thetai in enumerate(self._p[self.nb:self.nb+self.na]):
#            if np.pi - thetai < np.pi / 20:
#            #if False:
#                collinear.append(i)
#        # Next, mask all dihedral angles for which three of the atoms
#        # form an angle near 180 degrees.
#        for i, di in enumerate(self.dihedrals):
#            for j in collinear:
#                a, b, c = self.angles[j]
#                if (a in di) and (b in di) and (c in di):
#                    self._mask[self.nb + self.na + i] = False
#                    break
#        self._p, self._B, self._D = cart_to_internal(self.atoms.positions,
#                                                     self.bonds,
#                                                     self.angles,
#                                                     self.dihedrals,
#                                                     self._mask,
#                                                     gradient, curvature)
#        self._pos_old = self.atoms.positions.copy()
#
#        if self._B is not None:
#            self._Binv = np.linalg.pinv(self._B)
#            G = self._B @ self._B.T
#            lams, vecs = np.linalg.eigh(G)
#            indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-8]
#            self._U = vecs[:, indices]
#
#    @property
#    def p(self):
#        if self._p is not None and np.all(self.atoms.positions == self._pos_old):
#            return self._p
#        self._calc_internal()
#        self._U = None
#        self._B = None
#        self._Binv = None
#        self._D = None
#        return self._p
#
#    def p_wrap(self, vec):
#        nba = self.nb + self.na
#        out = vec.copy()
#        out[nba:] = (out[nba:] + np.pi) % (2 * np.pi) - np.pi
#        return out
#
#    def _p_setter_naive(self, target):
#        x0 = self.atoms.get_positions().ravel().copy()
#        x = x0.copy()
#        dp0 = None
#        errlast = np.infty
#        for _ in range(20):
#            Binv = np.linalg.pinv(self.B)
#            dp = self.p_wrap(target - self.p)
#            if dp0 is None:
#                dp0 = dp.copy()
#            err = np.linalg.norm(dp)
#            if err < 1e-8:
#                return True
#            elif abs(err - errlast) < 1e-8:
#                return False
#            errlast = err
#            x += Binv @ dp
#            self.atoms.positions = x.reshape((-1, 3))
#        # If the error is bigger than after the first step, revert entirely
#        if err > np.linalg.norm(dp0):
#            self.atoms.positions = x0.reshape((-1, 3))
#        return False
#
#
#
#    @p.setter
#    def p(self, target):
#        # Try naive algorithm first
#        if self._p_setter_naive(target):
#            return
#
#        # Dihedral angles can differ by at most pi and at least -pi.
#        # Modify dihedral angles accordingly
#        dp = self.p_wrap(target - self.p)
#
#        # Calculate linearized cartesian displacement vector.
#        # Rather than calculating pinv(B), we do SVD and explicitly
#        # leave out vectors corresponding to small singular values.
#        lvecs, lams, rvecs = np.linalg.svd(self.B, full_matrices=False)
#        indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
#        dx = rvecs[indices, :].T @ ((lvecs[:, indices].T @ dp) / lams[indices])
#        self.v0 = dx.copy()
#        dxnorm = np.linalg.norm(dx)
#        dx_max = 1e-2  # Maximum linear displacement to use
#
#        # If the total displacement is less than dx_max, just do a linear
#        # displacement and skip the rest of the algorithm
#        if dxnorm <= dx_max:
#            self.path = [self.atoms.get_positions().copy()]
#            self.atoms.positions += dx.reshape((-1, 3))
#            self.path.append(self.atoms.get_positions().copy())
#            self.v1 = dx.copy()
#            return
#        nsteps = int(np.ceil(dxnorm / dx_max))
#        dt = dxnorm / nsteps  # "time step"
#        # Initial "velocity" is initial displacement vector normalized
#        v = dx / dxnorm
#        # Acceleration comes from Coriolis forces in internal coordinates.
#        a = -rvecs[indices, :].T @ ((lvecs[:, indices].T @ self.D.ddot(v, v)) / lams[indices])
#        self.path = [self.atoms.get_positions().copy()]
#
#        # Parameters for DIIS (see below)
#        nscf = 100
#        ndiis = 5
#        viter = np.zeros((len(dx), nscf))
#        errors = np.zeros((len(dx), nscf))
#        p = self.p.copy()
#        # Outer loop controls linear displacements in cartesian coordinates
#        for k in range(nsteps):
#            print('step {} of {}'.format(k, nsteps))
#            # Use explicit velocity-verlet to integrate
#            vhalf = v + a * dt / 2
#            self.atoms.positions += (vhalf * dt).reshape((-1, 3))
#            D = self.D  # This calculates p and B as well
#            self.path.append(self.atoms.get_positions().copy())
#            v = vhalf.copy()
#            lvecs, lams, rvecs = np.linalg.svd(self.B, full_matrices=False)
#            indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
#            # Inner loop controls self consistency. Because of Coriolis
#            # forces, v(t) depends on a(t) which depends on v(t)... This
#            # is solved self-consistently using DIIS
#            for i in range(nscf):
#                ## Simple iterative algorithm. While this usually works,
#                ## it occasionally fails to converge
#                #vlast = v.copy()
#                #a = -rvecs[indices, :].T @ ((lvecs[:, indices].T @ self.D.ddot(v, v)) / lams[indices])
#                #v = vhalf + a * dt / 2
#                #if np.linalg.norm(v - vlast) < 1e-12:
#                #    break
#                vlast = v.copy()
#                viter[:, i] = v
#                a = -rvecs[indices, :].T @ ((lvecs[:, indices].T @ self.D.ddot(v, v)) / lams[indices])
#                errors[:, i] = vhalf + a * dt / 2 - v
#                # Don't do DIIS for first two iterations
#                if i <= 1:
#                    v = vhalf + a * dt / 2
#                    continue
#                # Only use at most ndiis histories
#                nhist = min(i+1, ndiis)
#                # DIIS interpolation
#                A = np.ones((nhist+1, nhist+1))
#                A[:nhist, :nhist] = errors[:, i+1-nhist:i+1].T @ errors[:, i+1-nhist:i+1]
#                A[-1, -1] = 0
#                rhs = np.zeros(nhist+1)
#                rhs[-1] = 1
#                cs = np.linalg.lstsq(A, rhs, rcond=None)[0]
#                v = viter[:, i+1-nhist:i+1] @ cs[:nhist]
#                if np.linalg.norm(v - vlast) < 1e-12:
#                    break
#            else:
#                print(np.linalg.norm(v - vlast))
#                raise RuntimeError('Failed to converge Coriolis forces')
#        self.v1 = dxnorm * v.copy()
#
#    def xpolate(self, alpha):
#        """Interpolate/extrapolate most recent trajectory"""
#        if self.path is None:
#            raise RuntimeError('No path to interpolate/extrapolate!')
#        # Extrapolate backwards, starting from original configuration
#        if alpha < 0:
#            self.atoms.positions = self.path[0]
#            self.p = self.p + alpha * self.B @ self.v0
#        # Linearly interpolate from two adjacent images in path
#        elif 0 < alpha < 1:
#            nsteps = len(self.path) - 1
#            beta = alpha * nsteps
#            idx = int(beta)
#            beta -= idx
#            self.atoms.positions = self.path[idx + 1] * beta + self.path[idx] * (1 - beta)
#        # Extrpolate forwards, starting from final configuration
#        else:
#            self.atoms.positions = self.path[-1]
#            self.p = self.p + (1 - alpha) * self.B @ self.v1
#        return self.p
#
#    @property
#    def B(self):
#        if self._B is not None and np.all(self.atoms.positions == self._pos_old):
#            return self._B
#        self._calc_internal(gradient=True)
#        self._D = None
#        return self._B
#
#    @property
#    def Binv(self):
#        if self._Binv is not None and np.all(self.atoms.positions == self._pos_old):
#            return self._Binv
#        self._calc_internal(gradient=True)
#        self._D = None
#        return self._Binv
#
#    @property
#    def U(self):
#        if self._U is not None and np.all(self.atoms.positions == self._pos_old):
#            return self._U
#        self._calc_internal(gradient=True)
#        self._D = None
#        return self._U
#
#    @property
#    def D(self):
#        if self._D is not None and np.all(self.atoms.positions == self._pos_old):
#            return self._D
#        self._calc_internal(gradient=True, curvature=True)
#        return self._D
#

