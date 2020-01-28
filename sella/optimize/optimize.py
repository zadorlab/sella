#!/usr/bin/env python3

import warnings
from time import localtime, strftime

from ase.optimize.optimize import Optimizer
from ase.utils import basestring
from ase.io.trajectory import Trajectory

from .restricted_step import get_restricted_step
from sella.peswrapper import PES, InternalPES

_default_kwargs = dict(minimum=dict(delta0=1e-1,
                                    sigma_inc=1.15,
                                    sigma_dec=0.90,
                                    rho_inc=1.035,
                                    rho_dec=100,
                                    method='rfo',
                                    eig=False),
                       saddle=dict(delta0=1.3e-3,
                                   sigma_inc=1.15,
                                   sigma_dec=0.65,
                                   rho_inc=1.035,
                                   rho_dec=5.0,
                                   method='prfo',
                                   eig=True))


class Sella(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=False, delta0=None,
                 sigma_inc=None, sigma_dec=None, rho_dec=None, rho_inc=None,
                 order=1, eig=None, eta=1e-4, method=None, gamma=0.4,
                 threepoint=False, constraints=None, constraints_tol=1e-5,
                 v0=None, internal=False, append_trajectory=False,
                 rs=None, **kwargs):
        if order == 0:
            default = _default_kwargs['minimum']
        else:
            default = _default_kwargs['saddle']

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(trajectory, mode=mode,
                                        atoms=atoms, master=master)

        asetraj = None
        if internal:
            MyPES = InternalPES
        else:
            MyPES = PES
        self.pes = MyPES(atoms, constraints=constraints,
                         trajectory=trajectory, eta=eta, v0=v0, **kwargs)

        if rs is None:
            rs = 'mis' if internal else 'tr'
        self.rs = get_restricted_step(rs)
        Optimizer.__init__(self, atoms, restart, logfile, asetraj, master,
                           force_consistent)

        if delta0 is None:
            delta0 = default['delta0']
        self.delta = delta0 * self.pes.get_Ufree().shape[1]

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
        self.peskwargs = dict(gamma=gamma, threepoint=threepoint)
        self.rho = 1.

        if self.ord != 0 and not self.eig:
            warnings.warn("Saddle point optimizations with eig=False will "
                          "most likely fail!\n Proceeding anyway, but you "
                          "shouldn't be optimistic.")

        self.initialized = False
        self.xi = 1.

    def _predict_step(self):
        if not self.initialized:
            self.pes.get_g()
            if self.eig:
                self.pes.diag(**self.peskwargs)
            self.initialized = True

        s, smag = self.rs(self.pes, self.ord, self.delta,
                          method=self.method).get_s()
        return s, smag

    def step(self):
        s, smag = self._predict_step()

        # Determine if we need to call the eigensolver, then step
        if self.eig:
            if self.pes.H.evals is None:
                ev = True
            else:
                Unred = self.pes.get_Unred()
                ev = (self.pes.get_HL().project(Unred)
                                       .evals[:self.ord] > 0).any()
        else:
            ev = False
        rho = self.pes.kick(s, ev, **self.peskwargs)

        # Update trust radius
        if rho is None:
            pass
        elif rho < 1./self.rho_dec or rho > self.rho_dec:
            self.delta = max(smag * self.sigma_dec, self.delta_min)
        elif 1./self.rho_inc < rho < self.rho_inc:
            self.delta = max(self.sigma_inc * smag, self.delta)
        self.rho = rho
        if self.rho is None:
            self.rho = 1.

    def converged(self, forces=None):
        return self.pes.converged(self.fmax)[0]

    def log(self, forces=None):
        if self.logfile is None:
            return
        _, fmax, cmax = self.pes.converged(self.fmax)
        e = self.pes.get_f()
        T = strftime("%H:%M:%S", localtime())
        name = self.__class__.__name__
        buf = " " * len(name)
        if self.nsteps == 0:
            self.logfile.write(buf + "{:>4s} {:>8s} {:>15s} {:>12s} {:>12s} "
                               "{:>12s} {:>12s}\n"
                               .format("Step", "Time", "Energy", "fmax",
                                       "cmax", "rtrust", "rho"))
        self.logfile.write("{} {:>3d} {:>8s} {:>15.6f} {:>12.4f} {:>12.4f} "
                           "{:>12.4f} {:>12.4f}\n"
                           .format(name, self.nsteps, T, e, fmax, cmax,
                                   self.delta, self.rho))
        self.logfile.flush()
