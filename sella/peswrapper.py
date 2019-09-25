import numpy as np

from scipy.linalg import eigh

from sella.geom import CartGeom, IntGeom
from sella.eigensolvers import rayleigh_ritz
from sella.linalg import NumericalHessian, ProjectedMatrix
from sella.hessian_update import update_H, symmetrize_Y


class DummyTrajectory:
    def write(self):
        pass


_valid_diag_keys = set(('gamma', 'threepoint', 'maxiter'))


class PESWrapper:
    def __init__(self, atoms, eigensolver='jd0', constraints=None,
                 trajectory=None, eta=1e-4, v0=None, internal=False):
        if internal:
            self.geom = IntGeom(atoms, constraints, trajectory=trajectory)
        else:
            self.geom = CartGeom(atoms, constraints, trajectory=trajectory)
        self.eigensolver = eigensolver
        self.H = None
        self.eta = eta
        self.v0 = v0

    @property
    def calls(self):
        return self.geom.neval

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
        pos0 = self.geom.atoms.positions.copy()
        self.geom.x = x
        g = self.geom.g
        f = self.geom.f
        self.geom.atoms.positions = pos0
        return f, g

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
        #h0 = self.geom.h.copy()

        Binv0 = self.geom.Binv.copy()

        if self.H is not None:
            df_pred = self.geom.gfree.T @ dx + (dx.T @ self.Hred @ dx) / 2.

        dx_full = self.geom.kick(dx)

        df_actual = self.geom.f - f0

        ##dx_free, dx_cons = self.geom.dx(x0, split=True)
        ##dx_full = self.geom.dx(x0, split=False)
        #if self.H is not None:
        #    P = B0 @ self.geom.Binv
        #    H = P.T @ self.H @ P
        #    g0 = g0 @ P
        #    df_pred = g0.T @ dx_full + (dx_full.T @ H @ dx_full) / 2.
        #    #df_pred = (g0.T @ dx_full - dx_free @ self.H @ dx_cons
        #    #           + (dx_free @ self.H @ dx_free) / 2.
        #    #           + (dx_cons @ self.H @ dx_cons) / 2.)
        #    #df_pred = g0.T @ dx_full + (dx_full.T @ self.H @ dx_full) / 2.


        #    ratio = df_actual / df_pred
        #else:
        #    ratio = None

        if self.H is not None:
            ratio = df_actual / df_pred
        else:
            ratio = None


        ##dh = self.geom.h - h0
        ##self.H = update_H(self.H, dx_full, dh)
        #dg = self.geom.dg
        #if self.H is not None:
        #    Bnew = self.geom.int.B(self.geom.lastlast['x'])
        #    P = Bnew @ Binv0
        #    self.H = P @ self.H @ P.T
        #self.H = update_H(self.H, dx_full, dg)
        self.H = self.geom.update_H(self.H)

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
