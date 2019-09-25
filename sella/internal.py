#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from .internal_cython import get_internal, cart_to_internal, expand_internal

from ase.data import covalent_radii, vdw_radii
from ase.units import Hartree, Bohr

class Internal:
    def __init__(self, atoms, angles=True, dihedrals=True, extra_bonds=None):
        if extra_bonds is None:
            extra_bonds = []

        self.bond_weight = 1.
        self.angle_weight = 1.
        self.dihedral_weight = 1.

        self.atoms = atoms
        self.use_angles = angles
        self.use_dihedrals = dihedrals
        self.extra_bonds = extra_bonds
        self._masked_angles = []
        self._get_internal()

        # A flag used to indicate to GeomInt that the bonding structure has
        # changed since the last time it checked
        self._bonding_changed = False

    def _get_internal(self):
        out = get_internal(self.atoms, self.use_angles, self.use_dihedrals, self.extra_bonds)
        self.c10y, self.nbonds, self.bonds, self.angles, self.dihedrals = out

        self.n = dict(bonds=len(self.bonds),
                      angles=len(self.angles) if self.use_angles else 0,
                      dihedrals=len(self.dihedrals) if self.use_dihedrals else 0)
        self.n.update(total=sum(self.n.values()))
        self._mask = np.ones(self.n['total'], dtype=np.uint8)
        self.last = dict(x=None, q=None, B=None, U=None, Binv=None, D=None, w=None)
        self._update(self.atoms.positions, True, False)


    def _update(self, pos, gradient=False, curvature=False):
        pos = pos.reshape((-1, 3))
        if np.all(pos == self.last['x']):
            if gradient and (self.last['B'] is None):
                pass
            elif curvature and (self.last['D'] is None):
                pass
            else:
                return

        self._check_for_bad_internal()

        q, B, D = cart_to_internal(pos,
                                   self.bonds,
                                   self.angles,
                                   self.dihedrals,
                                   self._mask,
                                   gradient,
                                   curvature)

        w = np.zeros_like(q)
        n = 0
        for i in range(self.n['bonds']):
            if self._mask[i]:
                w[n] = self.bond_weight
                n += 1

        for i in range(self.n['angles']):
            if self._mask[self.n['bonds'] + i]:
                w[n] = self.angle_weight
                n += 1

        for i in range(self.n['dihedrals']):
            if self._mask[self.n['bonds'] + self.n['angles'] + i]:
                w[n] = self.dihedral_weight
                n += 1

        assert n == len(q)

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

        self.last = dict(x=pos.copy(), q=q, B=B, U=U, Binv=Binv, D=D, w=w)

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

    def w(self, pos):
        self._update(pos)
        return self.last['w']

    def q_wrap(self, vec):
        nba = self.n['bonds'] + self.n['angles']
        out = vec.copy()
        out[nba:] = (out[nba:] + np.pi) % (2 * np.pi) - np.pi
        return out

    def _check_for_bad_internal(self):
        mask = np.ones(self.n['total'], dtype=np.uint8)
        q, _, _ = cart_to_internal(self.atoms.positions,
                                   self.bonds,
                                   self.angles,
                                   self.dihedrals,
                                   mask)

        nb = self.n['bonds']
        na = self.n['angles']
        bad = set()
        for i, theta in enumerate(q[nb : nb+na]):
            # ignore this angle if it's already been masked
            if not self._mask[nb + i]:
                continue

            if (theta < np.pi / 36  # angle between 175 and 180 degrees
                    or theta + np.pi / 36 > np.pi):  # Angle less than 5 degrees
                # Add the terminal atoms to the "bad" list
                bad.add(self.angles[i, 0])
                bad.add(self.angles[i, 2])

                # Sanity check: We've already decided that this angle is not
                # yet masked (though we are going to mask it), because its
                # entry in self._mask is set to True. That means the angle
                # shouldn't be in self._masked_angles. Verify this.
                for masked_angle in self._masked_angles:
                    assert not np.all(self.angles[i] == masked_angle)
                
                #print('masking new angle:', self.angles[i], "with value", theta)
                self._masked_angles.append(self.angles[i].copy())

        if not bad:
            return

        self._bonding_changed = True

        bad_np = np.array(list(bad), dtype=np.uintp)
        out = expand_internal(self.atoms, self.use_angles, self.use_dihedrals,
                              self.c10y, self.nbonds, self.bonds, bad_np)
        self.c10y, self.nbonds, self.bonds, self.angles, self.dihedrals = out

        self.n = dict(bonds=len(self.bonds),
                      angles=len(self.angles) if self.use_angles else 0,
                      dihedrals=len(self.dihedrals) if self.use_dihedrals else 0)
        self.n.update(total=sum(self.n.values()))
        self._mask = np.ones(self.n['total'], dtype=np.uint8)
        self.last = dict(x=None, q=None, B=None, U=None, Binv=None, D=None)

        # Now we update the mask
        nb = self.n['bonds']
        na = self.n['angles']
        for masked_angle in self._masked_angles:
            for i, angle in enumerate(self.angles):
                if np.all(angle == masked_angle):
                    self._mask[nb + i] = False
                    break
            else:
                raise RuntimeError("Couldn't find a masked angle in the list of all angles!")
        for i, dihedral in enumerate(self.dihedrals):
            for masked_angle in self._masked_angles:
                if (np.all(dihedral[:3] == masked_angle)
                        or np.all(dihedral[:3] == masked_angle[::-1])
                        or np.all(dihedral[1:] == masked_angle)
                        or np.all(dihedral[1:] == masked_angle[::-1])):
                    self._mask[na + nb + i] = False

    def guess_hessian(self, atoms):
        rcov = covalent_radii[atoms.numbers] / Bohr

        # Stretch parameters
        Ab = 0.3601 #* Hartree / Bohr**2
        Bb = 1.944 #/ Bohr

        # Bend parameters
        Aa = 0.089 #* Hartree
        Ba = 0.11 #* Hartree
        Ca = 0.44 #/ Bohr
        Da = -0.42

        # Torsion parameters
        At = 0.0015 #* Hartree
        Bt = 14.0 #* Hartree
        Ct = 2.85 #/ Bohr
        Dt = 0.57
        Et = 4.00

        h0 = np.zeros(self.n['total'] - int((1 - self._mask).sum()))

        n = 0
        start = 0
        for i, (a, b) in enumerate(self.bonds):
            if not self._mask[i]:
                continue
            rab = atoms.get_distance(a, b) / Bohr
            rcovab = rcov[a] + rcov[b]
            h0[n] = (Aa * np.exp(-Bb * (rab - rcovab))) * Hartree / Bohr**2
            n += 1

        start += self.n['bonds']
        for i, (a, b, c) in enumerate(self.angles):
            if not self._mask[start + i]:
                continue
            rab = atoms.get_distance(a, b) / Bohr
            rbc = atoms.get_distance(b, c) / Bohr
            rcovab = rcov[a] + rcov[b]
            rcovbc = rcov[b] + rcov[c]
            h0[n] = (Aa + Ba * np.exp(-Ca * (rab + rbc - rcovab - rcovbc)) / (rcovab * rcovbc)**Da) * Hartree
            n += 1

        start += self.n['angles']
        for i, (a, b, c, d) in enumerate(self.dihedrals):
            if not self._mask[start + i]:
                continue
            rbc = atoms.get_distance(b, c) / Bohr
            rcovbc = rcov[b] + rcov[c]
            L = self.nbonds[b] + self.nbonds[c] - 2
            h0[n] = (At + Bt * L**Dt * np.exp(-Ct * (rbc - rcovbc)) / (rbc * rcovbc)**Et) * Hartree
            n += 1

        return np.diag(h0)
