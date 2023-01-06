import warnings

import numpy as np
from scipy.linalg import eigh

from sella.peswrapper import PES

from ase.optimize.optimize import Optimizer
from .restricted_step import IRCTrustRegion
from .stepper import QuasiNewtonIRC

class IRCInnerLoopConvergenceFailure(RuntimeError):
    pass


class IRC(Optimizer):
    def __init__(
        self,
        atoms,
        restart=None,
        logfile='-',
        trajectory=None,
        master=None,
        force_consistent=False,
        ninner_iter=10,
        irctol=1e-2,
        dx=0.1,
        eta=1e-4,
        gamma=0.1,
        peskwargs=None,
        **kwargs
    ):
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master,
                           force_consistent)
        self.ninner_iter = ninner_iter
        self.irctol = irctol
        self.dx = dx
        if peskwargs is None:
            self.peskwargs = dict(gamma=gamma)

        if 'masses' not in self.atoms.arrays:
            try:
                self.atoms.set_masses('most_common')
            except ValueError:
                warnings.warn("The version of ASE that is installed does not "
                              "contain the most common isotope masses, so "
                              "Earth-abundance-averaged masses will be used "
                              "instead!")
                self.atoms.set_masses('defaults')

        self.sqrtm = np.repeat(np.sqrt(self.atoms.get_masses()), 3)

        def get_W(self):
            return np.diag(1. / np.sqrt(np.repeat(self.atoms.get_masses(), 3)))

        PES.get_W = get_W
        self.pes = PES(atoms, eta=eta, proj_trans=False, proj_rot=False,
                       **kwargs)

        self.lastrun = None
        self.x0 = self.pes.get_x().copy()
        self.v0ts = None
        self.H0 = None
        self.peslast = None
        self.xi = 1.
        self.first = True

    def irun(self, fmax=0.05, fmax_inner=0.01, steps=None, direction='forward'):
        if direction not in ['forward', 'reverse']:
            raise ValueError('direction must be one of "forward" or '
                             '"reverse"!')

        if self.v0ts is None:
            # Initial diagonalization
            self.pes.kick(0, True, **self.peskwargs)
            self.H0 = self.pes.get_H().asarray().copy()
            Hw = self.H0 / np.outer(self.sqrtm, self.sqrtm)
            _, vecs = eigh(Hw)
            self.v0ts = self.dx * vecs[:, 0] / self.sqrtm
            if self.v0ts[np.nonzero(self.v0ts)[0][0]] < 0: 
                self.v0ts *= -1 #force v0ts to be the direction where the first non-zero component is positive
            self.pescurr = self.pes.curr.copy()
            self.peslast = self.pes.last.copy()
        else:
            # Or, restore from last diagonalization for new direction
            self.pes.set_x(self.x0)
            self.pes.curr = self.pescurr.copy()
            self.pes.last = self.peslast.copy()
            self.pes.set_H(self.H0.copy(), initialized=True)

        if direction == 'forward':
            self.d1 = self.v0ts.copy()
        elif direction == 'reverse':
            self.d1 = -self.v0ts.copy()

        self.first = True
        self.fmax_inner = min(fmax, fmax_inner)
        return Optimizer.irun(self, fmax, steps)

    def run(self, fmax=0.05, fmax_inner=0.01, steps=None, direction='forward'):
        for converged in self.irun(fmax, fmax_inner, steps, direction):
            pass
        return converged

    def step(self):
        x0 = self.pes.get_x()
        if self.first:
            self.pes.kick(self.d1)
            self.first = False
        for n in range(self.ninner_iter):
            s, smag = IRCTrustRegion(
                self.pes, 0, self.dx, method=QuasiNewtonIRC, sqrtm=self.sqrtm,
                d1=self.d1
            ).get_s()

            bound_clip = abs(smag - self.dx) < 1e-8
            self.d1 += s

            self.pes.kick(s)
            g1 = self.pes.get_g()

            d1m = self.d1 * self.sqrtm
            d1m /= np.linalg.norm(d1m)
            g1m = g1 / self.sqrtm

            g1m_proj = g1m - d1m * (d1m @ g1m)
            fmax = np.linalg.norm((g1m_proj * self.sqrtm).reshape((-1, 3)), axis=1).max()

            g1m /= np.linalg.norm(g1m)
            dot = np.abs(d1m @ g1m)
            snorm = np.linalg.norm(s)
            #print(bound_clip, snorm, dot, fmax)
            if bound_clip and fmax < self.fmax_inner:
                break
            elif self.converged():
                break
        else:
            raise IRCInnerLoopConvergenceFailure

        self.d1 *= 0.

    def converged(self, forces=None):
        return self.pes.converged(self.fmax)[0] and self.pes.H.evals[0] > 0
