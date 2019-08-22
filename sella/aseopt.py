#!/usr/bin/env python3

import warnings

import numpy as np

from ase.optimize.optimize import Optimizer

from .peswrapper import PESWrapper
from .optimize import rs_newton, rs_rfo, rs_prfo


class Sella(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=False, delta0=1.3e-3,
                 sigma_inc=1.15, sigma_dec=0.65, rho_dec=5.0, rho_inc=1.035,
                 order=1, eig=True, eta=1e-4, peskwargs=None, method='rsprfo',
                 gamma=0.4, constraints_tol=1e-5, **kwargs):
        Optimizer.__init__(self, atoms, restart, logfile, None, master,
                           force_consistent)
        self.pes = PESWrapper(atoms, atoms.calc, trajectory=trajectory,
                              **kwargs)
        self.delta = delta0 * len(self.pes.x_m)
        self.sigma_inc = sigma_inc
        self.sigma_dec = sigma_dec
        self.rho_inc = rho_inc
        self.rho_dec = rho_dec
        self.ord = order
        self.eig = eig
        self.eta = eta
        self.delta_min = self.eta
        self.constraints_tol = constraints_tol
        self.niter = 0
        if peskwargs is None:
            self.peskwargs = dict(eta=self.eta, gamma=gamma)
        else:
            self.peskwargs = peskwargs
            if 'eta' not in self.peskwargs:
                self.peskwargs['eta'] = self.eta
            if 'gamma' not in self.peskwargs:
                self.peskwargs['gamma'] = gamma

        if self.ord != 0 and not self.eig:
            warnings.warn("Saddle point optimizations with eig=False will "
                          "most likely fail!\n Proceeding anyway, but you "
                          "shouldn't be optimistic.")

        self.initialized = False
        self.xi = 1.
        self.method = method
        if self.method not in ['gmtrm', 'rsrfo', 'rsprfo']:
            raise ValueError('Unknown method:', method)

    def step(self):
        if not self.initialized:
            f, self.glast, _ = self.pes.kick(np.zeros_like(self.pes.x_m))
            if self.eig:
                self.pes.diag(**self.peskwargs)
            self.initialized = True

        # Find new search step
        if self.method == 'gmtrm':
            s, smag, self.xi, bound_clip = rs_newton(self.pes, self.glast,
                                                     self.delta, self.ord,
                                                     self.xi)
        elif self.method == 'rsrfo':
            s, smag, self.xi, bound_clip = rs_rfo(self.pes, self.glast,
                                                  self.delta, self.ord,
                                                  self.xi)
        elif self.method == 'rsprfo':
            s, smag, self.xi, bound_clip = rs_prfo(self.pes, self.glast,
                                                   self.delta, self.ord,
                                                   self.xi)
        else:
            raise RuntimeError("Don't know what to do for method", self.method)

        # Determine if we need to call the eigensolver, then step
        ev = (self.eig and self.pes.lams is not None
              and np.any(self.pes.lams[:self.ord] > 0))
        f, self.glast, dx = self.pes.kick(s, ev, **self.peskwargs)
        self.niter += 1

        # Update trust radius
        rho = self.pes.ratio
        if rho is None:
            rho = 1.
        if rho < 1./self.rho_dec or rho > self.rho_dec:
            self.delta = max(smag * self.sigma_dec, self.delta_min)
        elif 1./self.rho_inc < rho < self.rho_inc:
            self.delta = max(self.sigma_inc * smag, self.delta)

    def _project_forces(self, forces):
        # Fool the optimizer into thinking the gradient is orthogonal to the
        # constraint subspace
        return (self.pes.Tm @ self.pes.Tm.T @ forces.ravel()).reshape((-1, 3))

    def converged(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        forces = self._project_forces(forces)
        return (np.all(np.abs(self.pes.res) < self.constraints_tol)
                and (forces**2).sum(1).max() < self.fmax**2)

    def log(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        return Optimizer.log(self, self._project_forces(forces))
