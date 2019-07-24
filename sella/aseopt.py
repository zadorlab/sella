#!/usr/bin/env python3

import warnings

import numpy as np

from ase.optimize.optimize import Optimizer

from .sella import MinModeAtoms
from .optimize import rs_newton

class Sella(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=False, r_trust=5e-4,
                 inc_factr=1.1, dec_factr=0.9, dec_ratio=5.0, inc_ratio=1.01,
                 order=1, eig=True, dxL=1e-4, mmkwargs=None, **kwargs):
        Optimizer.__init__(self, atoms, restart, logfile, None, master, force_consistent)
        self.mm = MinModeAtoms(atoms, atoms.calc, trajectory=trajectory, **kwargs)
        self.atoms = self.mm.atoms  # override assignment from Optimizer.__init__
        self.r_trust = r_trust * len(self.mm.x_m)
        self.sinc = inc_factr
        self.sdec = dec_factr
        self.Qinc = inc_ratio
        self.Qdec = dec_ratio
        self.ord = order
        self.eig = eig
        self.dxL = dxL
        self.r_trust_min = self.dxL
        if mmkwargs is None:
            self.mmkwargs = dict()
        else:
            self.mmkwargs = mmkwargs

        if self.ord != 0 and not self.eig:
            warnings.warn("Saddle point optimizations with eig=False will "
                          "most likely fail!\n Proceeding anyway, but you "
                          "shouldn't be optimistic.")

        self.initialized = False
        self.xi = 1.


    def step(self):
        if not self.initialized:
            f, self.glast, _ = self.mm.kick(np.zeros_like(self.mm.x_m))
            if self.eig:
                self.mm.f_minmode(dxL=self.dxL, **self.mmkwargs)
            self.initialized = True

        # Find new search step
        dx, dx_mag, self.xi, bound_clip = rs_newton(self.mm, self.glast, self.r_trust, self.ord, self.xi)

        # Determine if we need to call the eigensolver, then step
        ev = (self.eig and self.mm.lams is not None
                  and np.any(self.mm.lams[:self.ord] > 0))
        f, self.glast, dx = self.mm.kick(dx, ev, **self.mmkwargs)

        # Update trust radius
        ratio = self.mm.ratio
        if ratio is None:
            ratio = 1.
        if ratio < 1./self.Qdec or ratio > self.Qdec:
            self.r_trust = max(dx_mag * self.sdec, self.r_trust_min)
        elif 1./self.Qinc < ratio < self.Qinc:
            self.r_trust = max(self.sinc * dx_mag, self.r_trust)
