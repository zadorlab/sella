#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np
import sympy as sym
import networkx as nx
from scipy.optimize import minimize
from .internal_cython import get_internal, cart_to_internal

from ase.data import covalent_radii, vdw_radii

def _B_ind(indices):
    B_ind = []
    for ind in indices:
        for i in range(3):
            B_ind.append(3 * ind + i)
    return B_ind

class Internal(object):
    def __init__(self, atoms, bonds=True, angles=False, dihedrals=False):

        self.use_angles = angles
        self.use_dihedrals = dihedrals

        self.atoms = atoms
        self._pos_old = np.zeros_like(self.atoms.get_positions())

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, new_atoms):
        self._atoms = new_atoms.copy()
        self.natoms = len(self._atoms)

        c10y, nbonds, self.bonds, self.angles, self.dihedrals = get_internal(
                self._atoms, self.use_angles, self.use_dihedrals)
        self.nb = len(self.bonds)
        if self.use_angles:
            self.na = len(self.angles)
        else:
            self.na = 0

        if self.use_dihedrals:
            self.nd = len(self.dihedrals)
        else:
            self.nd = 0

        print(self.nb, self.na, self.nd)

        self.ninternal = self.nb + self.na + self.nd

        self._p = None
        self._B = None
        self._D = None

        # Index of atom to be placed at the origin
        ind0 = 0
        # Index of atom to be placed along X axis
        ind1 = min(c10y[ind0, :nbonds[ind0]])
        tmp = set(c10y[ind1, :nbonds[ind1]])
        tmp.discard(ind0)
        # Index of atom to be placed in X-Y plane
        ind2 = min(tmp)
        
        # Translate ind0 to origin
        self._atoms.positions -= self._atoms.get_positions()[ind0]
        
        # Rotate ind1 into X axis
        pos = self._atoms.get_positions()
        axis = np.cross([1., 0., 0.], pos[ind1])
        angle = -np.arccos(pos[ind1, 0] / np.linalg.norm(pos[ind1])) * 180. / np.pi
        self._atoms.rotate(angle, axis)
        
        # Rotate ind2 into X-Y plane
        pos = self._atoms.get_positions()
        angle = -np.arccos(pos[ind2, 1] /  np.linalg.norm(pos[ind2, 1:])) * 180. / np.pi
        self._atoms.rotate(angle, 'x')

    @property
    def p(self):
        if self._p is not None and np.all(self.atoms.positions == self._pos_old):
            return self._p
        self._p, _, _ = cart_to_internal(self.atoms.positions,
                                         self.bonds,
                                         self.angles,
                                         self.dihedrals)
        self._B = None
        self._D = None
        self._pos_old = self.atoms.positions.copy()
        return self._p

    #@p.setter
    #def p(self, target):
    #    x0 = self.atoms.get_positions().ravel().copy()
    #    self.res = minimize(self._p_min_obj, x0, jac=True, method='BFGS', args=(target,), options={'gtol': 1e-8})
    #    pos = self.res['x'].reshape((self.natoms, 3))
    #    self.atoms.set_positions(pos)

    #def _p_min_obj(self, x, p_target):
    #    self.atoms.set_positions(x.reshape((self.natoms, 3)))
    #    dp = self.p - p_target
    #    dt = dp[self.na+self.nb:]
    #    dp[self.na+self.nb:] = (dt + np.pi) % (2 * np.pi) - np.pi
    #    mu = np.dot(dp, dp)
    #    jac = 2 * np.dot(dp, self.B)
    #    return mu, jac

    @p.setter
    def p(self, target):
        x = self.atoms.get_positions().ravel().copy()

        x1 = None
        for i in range(100):
            dp = target - self.p
            dp[self.nb+self.na:] = (dp[self.nb+self.na:] + np.pi) % (2 * np.pi) - np.pi
            B = self.B
            G = B @ B.T
            Ginv = np.linalg.pinv(G)
            dx = B.T @ Ginv @ dp
            if np.linalg.norm(dx) < 1e-8:
                break
            x += dx
            if x1 is None:
                x1 = x.copy()
            self.atoms.set_positions(x.reshape((self.natoms, 3)))
        else:
            warnings.warn("Internal coordinate setter did not converge!",
                          RuntimeWarning)
            self.atoms.set_positions(x1.reshape((self.natoms, 3)))

#    @property
#    def q(self):
#        self._q = np.dot(self.C, self.p)
#        return self._q
#
#    @q.setter
#    def q(self, target):
#        target_full = np.zeros(3 * self.natoms)
#        target_full[:self.ncart] = target
#        q_full = np.dot(self.C_full, self.p)
#        dq_full = target_full - q_full
#        q_full[:self.ncart] = target
#        dp_target = np.linalg.lstsq(self.C_full, dq_full, rcond=None)[0]
#        self.p += dp_target

    @property
    def B(self):
        if self._B is not None and np.all(self.atoms.positions == self._pos_old):
            return self._B
        self._p, self._B, _ = cart_to_internal(self.atoms.positions,
                                               self.bonds,
                                               self.angles,
                                               self.dihedrals,
                                               gradient=True)
        self._D = None
        self._pos_old = self.atoms.positions.copy()
        return self._B

    @property
    def D(self):
        if self._D is not None and np.all(self.atoms.positions == self._pos_old):
            return self._D
        self._p, self._B, self._D = cart_to_internal(self.atoms.positions,
                                                     self.bonds,
                                                     self.angles,
                                                     self.dihedrals,
                                                     gradient=True,
                                                     curvature=True)
        self._pos_old = self.atoms.positions.copy()
        return self._D
