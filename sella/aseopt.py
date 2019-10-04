#!/usr/bin/env python3

import warnings

import numpy as np

from scipy.optimize.linesearch import scalar_search_wolfe1

from ase.io.trajectory import Trajectory
from ase.optimize.optimize import Optimizer
from ase.utils import basestring

from .peswrapper import PESWrapper
from .optimize import rs_newton, rs_rfo, rs_prfo

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
                 order=1, eig=None, eta=1e-4, peskwargs=None, method=None,
                 gamma=0.4, constraints_tol=1e-5, H0=None,
                 append_trajectory=None, **kwargs):
        if order == 0:
            default = _default_kwargs['minimum']
        else:
            default = _default_kwargs['saddle']

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(trajectory, mode=mode,
                                        atoms=atoms, master=master)

        if isinstance(atoms, PESWrapper):
            asetraj = trajectory
            self.pes = atoms
            atoms = self.pes.atoms
        else:
            asetraj = None
            self.pes = PESWrapper(atoms, atoms.calc, trajectory=trajectory,
                                  **kwargs)
        Optimizer.__init__(self, atoms, restart, logfile, asetraj, master,
                           force_consistent)

        if delta0 is None:
            self.delta = default['delta0'] * len(self.pes.x_m)
        else:
            self.delta = delta0 * len(self.pes.x_m)

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
        if self.method not in ['gmtrm', 'rsrfo', 'rsprfo']:
            raise ValueError('Unknown method:', self.method)

        if H0 is not None:
            self.pes.H = H0.copy()

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
