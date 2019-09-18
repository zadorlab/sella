import numpy as np

from scipy.linalg import eigh

from sella.geom import CartGeom
from sella.eigensolvers import rayleigh_ritz
from sella.linalg import NumericalHessian, ProjectedMatrix
from sella.hessian_update import update_H, symmetrize_Y


class DummyTrajectory:
    def write(self):
        pass


_valid_diag_keys = set(('gamma', 'threepoint', 'maxiter'))


class PESWrapper:
    def __init__(self, atoms, eigensolver='jd0', constraints=None,
                 trajectory=None, eta=1e-4, v0=None):
        self.geom = CartGeom(atoms, constraints, trajectory=trajectory)
        self.eigensolver = eigensolver
        self.H = None
        self.eta = eta
        self.v0 = v0

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, target):
        if target is None:
            self._H = None
            self.Hred = None
            self.lams = None
            self.vecs = None
            return
        self._H = target
        self.Hred = self.geom.Ufree.T @ self._H @ self.geom.Ufree
        self.lams, self.vecs = eigh(self.Hred)

    def _calc_eg(self, x):
        self.geom.x = x
        return self.geom.f, self.geom.g

    def converged(self, fmax, maxres=1e-5):
        return ((self.geom.forces**2).sum(1).max() < fmax**2
                and (np.linalg.norm(self.geom.res) < maxres))

    def update(self, dx, diag=False, **diag_kwargs):
        """Update the molecular geometry according
        to the displacement vector dx"""

        # Before we do anything, make sure the diag_kwargs are valid
        invalid_keys = set(diag_kwargs.keys()) - _valid_diag_keys
        if invalid_keys:
            raise ValueError("Unknown keywords passed to update:",
                             invalid_keys)

        x0 = self.geom.x.copy()
        f0 = self.geom.f
        g0 = self.geom.g.copy()
        h0 = self.geom.h.copy()

        self.geom.xfree = self.geom.xfree + dx

        df_actual = self.geom.f - f0

        dx_full = self.geom.x - x0
        dx_free, dx_cons = self.geom.split_dx(dx_full)
        if self.H is not None:
            df_pred = (g0.T @ dx_full - dx_free @ self.H @ dx_cons
                       + (dx_free @ self.H @ dx_free) / 2.
                       + (dx_cons @ self.H @ dx_cons) / 2.)

            ratio = df_actual / df_pred
        else:
            ratio = None

        dh = self.geom.h - h0
        self.H = update_H(self.H, dx_full, dh)

        if diag:
            self.diag(**diag_kwargs)

        return self.geom.f, self.geom.gfree, ratio

    def diag(self, gamma=0.5, threepoint=False, maxiter=None):
        x0 = self.geom.x.copy()

        P = self.Hred
        v0 = None
        if P is None:
            P = np.eye(len(self.geom.xfree))
            if self.v0 is not None:
                v0 = self.geom.Ufree.T @ self.v0
            else:
                v0 = self.geom.gfree

        Htrue = NumericalHessian(self._calc_eg, self.geom.x.copy(),
                                 self.geom.g.copy(), self.eta, threepoint)

        Hproj = ProjectedMatrix(Htrue, self.geom.Ufree)
        lams, Vs, AVs = rayleigh_ritz(Hproj, gamma, P, v0=v0,
                                      method=self.eigensolver,
                                      maxiter=maxiter)

        Vs = Hproj.Vs
        AVs = Hproj.AVs

        self.geom.x = x0
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2)
        theta, X = eigh(Atilde)

        Vs = Vs @ X
        AVs = AVs @ X
        AVstilde = AVs - self.geom.drdx @ self.geom.Ucons.T @ AVs

        self.H = update_H(self.H, Vs, AVstilde)
