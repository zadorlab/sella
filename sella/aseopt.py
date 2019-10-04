#!/usr/bin/env python3

import warnings

import numpy as np

from ase.optimize.optimize import Optimizer
from ase.utils import basestring

from sella.optimize import rs_newton, rs_rfo, rs_prfo
from sella.peswrapper import BasePES, CartPES, IntPES

_default_kwargs = dict(minimum=dict(delta0=1e-1,
                                    sigma_inc=1.15,
                                    sigma_dec=0.90,
                                    rho_inc=1.035,
                                    rho_dec=100,
                                    method='rsrfo',
                                    eig=False),
                       saddle=dict(delta0=1.3e-3,
                                   sigma_inc=1.15,
                                   sigma_dec=0.65,
                                   rho_inc=1.035,
                                   rho_dec=5.0,
                                   method='rsprfo',
                                   eig=True))


class Sella(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=False, delta0=None,
                 sigma_inc=None, sigma_dec=None, rho_dec=None, rho_inc=None,
                 order=1, eig=None, eta=1e-4, method=None, gamma=0.4,
                 threepoint=False, constraints=None, constraints_tol=1e-5,
                 v0=None, internal=False, append_trajectory=False):
        if order == 0:
            default = _default_kwargs['minimum']
        else:
            default = _default_kwargs['saddle']

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(trajectory, mode=mode,
                                        atoms=atoms, master=master)

        if isinstance(atoms, BasePES):
            asetraj = trajectory
            self.pes = atoms
            atoms = self.pes.atoms
        else:
            asetraj = None
            if internal:
                self.pes = IntPES(atoms, constraints=constraints,
                                  trajectory=trajectory, eta=eta, v0=v0)
            else:
                self.pes = CartPES(atoms, constraints=constraints,
                                   trajectory=trajectory, eta=eta, v0=v0)
        Optimizer.__init__(self, atoms, restart, logfile, asetraj, master,
                           force_consistent)

        if delta0 is None:
            self.delta = default['delta0'] * len(self.pes.xfree)
        else:
            self.delta = delta0 * len(self.pes.xfree)

        if sigma_inc is None:
            self.sigma_inc = default['sigma_inc']
        else:
            self.sigma_inc = sigma_inc

        if sigma_dec is None:
            self.sigma_dec = default['sigma_dec']
        else:
            self.sigma_dec = sigma_dec

        if rho_inc is None:
            self.rho_inc = default['rho_inc']
        else:
            self.rho_inc = rho_inc

        if rho_dec is None:
            self.rho_dec = default['rho_dec']
        else:
            self.rho_dec = rho_dec

        if method is None:
            self.method = default['method']
        else:
            self.method = method

        if eig is None:
            self.eig = default['eig']
        else:
            self.eig = eig

        self.ord = order
        self.eta = eta
        self.delta_min = self.eta
        self.constraints_tol = constraints_tol
        self.niter = 0
        self.peskwargs = dict(gamma=gamma, threepoint=threepoint)

        if self.ord != 0 and not self.eig:
            warnings.warn("Saddle point optimizations with eig=False will "
                          "most likely fail!\n Proceeding anyway, but you "
                          "shouldn't be optimistic.")

        self.initialized = False
        self.xi = 1.
        if self.method not in ['gmtrm', 'rsrfo', 'rsprfo']:
            raise ValueError('Unknown method:', self.method)

    def _predict_step(self):
        if not self.initialized:
            self.glast = self.pes.gfree
            if self.eig:
                self.pes.diag(**self.peskwargs)
            self.initialized = True

        # Find new search step
        if self.method == 'gmtrm':
            s, smag, self.xi, bound_clip = rs_newton(self.pes, self.glast,
                                                     self.delta,
                                                     self.pes.Winv,
                                                     self.ord, self.xi)
        elif self.method == 'rsrfo':
            s, smag, self.xi, bound_clip = rs_rfo(self.pes, self.glast,
                                                  self.delta,
                                                  self.pes.Winv,
                                                  self.ord, self.xi)
        elif self.method == 'rsprfo':
            s, smag, self.xi, bound_clip = rs_prfo(self.pes, self.glast,
                                                   self.delta, self.pes.Winv,
                                                   self.ord, self.xi)
        else:
            raise RuntimeError("Don't know what to do for method", self.method)

        return s, smag

    def step(self):
        s, smag = self._predict_step()

        # Determine if we need to call the eigensolver, then step
        ev = (self.eig and self.pes.lams is not None
              and np.any(self.pes.lams[:self.ord] > 0))
        f, self.glast, rho = self.pes.kick(s, ev, **self.peskwargs)
        self.niter += 1

        # Update trust radius
        if rho is None:
            pass
        elif rho < 1./self.rho_dec or rho > self.rho_dec:
            self.delta = max(smag * self.sigma_dec, self.delta_min)
        elif 1./self.rho_inc < rho < self.rho_inc:
            self.delta = max(self.sigma_inc * smag, self.delta)

    def converged(self, forces=None):
        return self.pes.converged(self.fmax)

    def log(self, forces=None):
        return Optimizer.log(self, self.pes.forces)
