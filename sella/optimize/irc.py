import warnings
from typing import Optional, Union, Dict, Any

import numpy as np
from scipy.linalg import eigh

from ase import Atoms
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.optimize.optimize import Optimizer

from sella.peswrapper import PES
from .restricted_step import IRCTrustRegion
from .stepper import QuasiNewtonIRC


class IRCInnerLoopConvergenceFailure(RuntimeError):
    pass


class IRC(Optimizer):
    def __init__(
        self,
        atoms: Atoms,
        logfile: str = '-',
        trajectory: Optional[Union[str, TrajectoryWriter]] = None,
        master: Optional[bool] = None,
        force_consistent: bool = False,
        ninner_iter: int = 10,
        irctol: float = 1e-2,
        dx: float = 0.1,
        eta: float = 1e-4,
        gamma: float = 0.1,
        peskwargs: Optional[Dict[str, Any]] = None,
        keep_going: bool = False,
        **kwargs
    ):
        Optimizer.__init__(
            self,
            atoms,
            None,
            logfile,
            trajectory,
            master,
            force_consistent,
        )
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

        self.pes = PES(atoms, eta=eta, proj_trans=False, proj_rot=False,
                       **kwargs)

        self.lastrun = None
        self.x0 = self.pes.get_x().copy()
        self.v0ts: Optional[np.ndarray] = None
        self.H0: Optional[np.ndarray] = None
        self.peslast = None
        self.xi = 1.
        self.first = True
        self.keep_going = keep_going

    def irun(
        self,
        fmax: float = 0.05,
        fmax_inner: float = 0.01,
        steps: Optional[int] = None,
        direction: str = 'forward',
    ):
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

            # force v0ts to be the direction where the first non-zero
            # component is positive
            if self.v0ts[np.nonzero(self.v0ts)[0][0]] < 0:
                self.v0ts *= -1

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

    def run(self, *args, **kwargs):
        for converged in self.irun(*args, **kwargs):
            pass
        return converged

    def step(self):
        if self.first:
            self.pes.kick(self.d1)
            self.first = False
        for n in range(self.ninner_iter):
            s, smag = IRCTrustRegion(
                self.pes,
                0,
                self.dx,
                method=QuasiNewtonIRC,
                sqrtm=self.sqrtm,
                d1=self.d1,
                W=self.get_W(),
            ).get_s()

            bound_clip = abs(smag - self.dx) < 1e-8
            self.d1 += s

            self.pes.kick(s)
            g1 = self.pes.get_g()

            d1m = self.d1 * self.sqrtm
            d1m /= np.linalg.norm(d1m)
            g1m = g1 / self.sqrtm

            g1m_proj = g1m - d1m * (d1m @ g1m)
            fmax = np.linalg.norm(
                (g1m_proj * self.sqrtm).reshape((-1, 3)), axis=1
            ).max()

            g1m /= np.linalg.norm(g1m)
            if bound_clip and fmax < self.fmax_inner:
                break
            elif self.converged():
                break
        else:
            if self.keep_going:
                warnings.warn(
                    'IRC inner loop failed to converge! The trajectory is no '
                    'longer a trustworthy IRC.'
                )
            else:
                raise IRCInnerLoopConvergenceFailure

        self.d1 *= 0.

    def converged(self, forces=None):
        return self.pes.converged(self.fmax)[0] and self.pes.H.evals[0] > 0

    def get_W(self):
        return np.diag(1. / np.sqrt(np.repeat(self.atoms.get_masses(), 3)))
