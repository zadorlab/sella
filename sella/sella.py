#!/usr/bin/env python

import numpy as np
from scipy.linalg import eigh, null_space
from .eigensolvers import NumericalHessian, ProjectedMatrix, atoms_tr_projection, davidson
from .hessian_update import update_H

class MinMode(object):
    def __init__(self, f, d, minmode=davidson, molecule=False, H0=None, v0=None,
                 trshift=1000, trshift_factor=4.):
        self.f = f
        self.d = d
        self.v = v0
        self.lam = 1
        self.minmode = minmode
        self.H = None
        self.xlast = None
        self.glast = None
        self.Hproj = None
        self.molecule = molecule
        self.H0 = H0
        self.shift = trshift
        self.shift_factor = trshift_factor
        self.calls = 0
        self.Hlast = None
        self.Hinv = None

    def f_update(self, x):
        self.calls += 1
        f, g = self.f(x)
        if self.xlast is not None and self.glast is not None:
            self.H = update_H(self.H, x - self.xlast, g - self.glast)
            self.lams, self.vecs = eigh(self.H)
        self.xlast = x.copy()
        self.glast = g.copy()
        return f, g

    def f_minmode(self, x0, dxL=1e-5, maxres=5e-3, threepoint=False, **kwargs):
        d = len(x0)
        
        f0, g0 = self.f_update(x0)

        Htrue = NumericalHessian(self.f, x0, g0, dxL, threepoint)

        I = np.eye(self.d)
        if self.molecule:
            T = atoms_tr_projection(x0)
            _, ntr = T.shape
        else:
            T = np.empty((self.d, 0))
            ntr = 0

        P = self.H
        if P is None:
            if self.H0 is not None:
                self.H = self.H0
                P = self.H
            elif self.v is not None:
                P = I - 2 * np.outer(self.v, self.v)
            else:
                u = g0 / np.linalg.norm(g0)
                P = I - 2 * np.outer(u, u)
        H = Htrue

        if self.v is None or np.abs(self.lams[0]) < 1e-8:
            if self.H is not None:
                self.v = self.vecs[:, 0]
            else:
                self.v = np.random.normal(size=d)
                self.v /= np.linalg.norm(self.v)

        v0 = self.v

        lams, Vs, AVs = self.minmode(H, maxres, P, T, shift=self.shift, **kwargs)
        self.calls += H.calls

        #Proj = I - T @ T.T
        if ntr > 0:
            Tnull = null_space(T.T)
            Proj = Tnull @ Tnull.T
        else:
            Proj = np.eye(self.d)

        if self.H is None:
            #lam0 = lams[-(ntr + 1)]
            #lam0 = np.sqrt(np.average(lams[:-ntr]**2))
            if ntr == 0:
                lam0 = np.average(np.abs(lams))
            else:
                lam0 = np.average(np.abs(lams[:-ntr]))
            #self.H = lam0 * I + (self.shift - lam0) * T @ T.T
            self.H = lam0 * Proj + self.shift * T @ T.T
            #self.H = lam0 * (Proj @ I @ Proj) + 1000 * T @ T.T

        self.H = update_H(self.H, Vs, AVs)

        lams_all, vecs_all = eigh(self.H)
        self.shift = self.shift_factor * lams_all[-(ntr + 1)]

        self.H = Proj @ (self.H) @ Proj + self.shift * (T @ T.T)

        self.lams, self.vecs = eigh(self.H)

        return f0, g0

    def f_dimer(self, x0, *args, **kwargs):
        f, g = self.f_minmode(self, x0, *args, **kwargs)
        gpar = self.vecs[:, 0] * (self.vecs[:, 0] @ g)
        if self.lams[0] > 0:
            return f, -gpar
        return f, g - 2 * gpar

