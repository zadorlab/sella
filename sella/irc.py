import warnings

import numpy as np
from scipy.linalg import eigh

from sella.peswrapper import PESWrapper
from sella.constraints import initialize_constraints, calc_constr_basis

from ase.optimize.optimize import Optimizer


def rs_newton_irc(pes, sqrtm, g, d1, dx, xi=1.):
    Lcart = pes.lams
    vecscart = pes.Tm @ pes.vecs

    Hmw = (vecscart @ np.diag(Lcart) @ vecscart.T) / np.outer(sqrtm, sqrtm)
    L, vecs = eigh(Hmw)

    # exclude indices corresponding to neglected
    # translational/rotational modes
    ind = []
    for i, Li in enumerate(L):
        if abs(Li) > 1e-8:
            ind.append(i)

    L = np.abs(L[ind])
    vecs = vecs[:, ind]

    gmw = g / sqrtm
    d1mw = d1 * sqrtm
    Vg = vecs.T @ gmw
    Vd1 = vecs.T @ d1mw

    epsmw = -vecs @ (Vg / L)
    d2mw = d1mw + epsmw
    d2mw_mag = np.linalg.norm(d2mw)
    if d2mw_mag < dx:
        eps = epsmw / sqrtm
        return eps, xi, False

    xilower = 0.
    xiupper = None

    for _ in range(100):
        epsmw = -vecs @ ((Vg + xi * Vd1) / (L + xi))
        d2mw = d1mw + epsmw
        d2mw_mag = np.linalg.norm(d2mw)

        if abs(d2mw_mag - dx) < 1e-14 * dx:
            break
        depsmw = -vecs @ (Vd1 / (L + xi) - (Vg + xi * Vd1) / (L + xi)**2)
        dd2mw_mag = (d2mw @ depsmw) / d2mw_mag
        dxi = (dx - d2mw_mag) / dd2mw_mag

        if dxi < 0:
            xiupper = xi
        elif dxi > 0:
            xilower = xi

        if xiupper is not None and np.nextafter(xilower, xiupper) >= xiupper:
            break

        xi1 = xi + dxi

        if xi1 <= xilower or (xiupper is not None and xi1 >= xiupper):
            if xiupper is None:
                xi += (xi - xilower)
            else:
                xi = (xiupper + xilower) / 2.
        else:
            xi = xi1
    else:
        raise RuntimeError("IRC Newton step failed!")
    eps = epsmw / sqrtm

    return eps, xi, True


class IRC(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=False, irctol=1e-2, dx=0.1,
                 eta=1e-4, gamma=0.4, peskwargs=None, H0=None, **kwargs):
        if isinstance(atoms, PESWrapper):
            self.pes = atoms
            atoms = self.pes.atoms
        else:
            self.pes = PESWrapper(atoms, atoms.calc,
                                  project_rotations=False,
                                  project_translations=False,
                                  **kwargs)
        self.pes_cons = PESWrapper(atoms, atoms.calc,
                                   project_rotations=True,
                                   project_translations=True,
                                   **kwargs)
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master,
                           force_consistent)
        self.irctol = irctol
        self.dx = dx
        if peskwargs is None:
            self.peskwargs = dict(eta=eta, gamma=gamma)

        if 'masses' not in self.atoms.arrays:
            try:
                self.atoms.set_masses('most_common')
            except ValueError:
                warnings.warn("The version of ASE that is installed does not "
                              "contain the most common isotope masses, so "
                              "Earth-abundance-averaged masses will be used "
                              "instead!")
                self.atoms.set_masses('defaults')

        self.sqrtm = np.sqrt(self.atoms.get_masses()[:, np.newaxis]
                             * np.ones_like(self.atoms.positions)).ravel()

        self.x0 = self.pes.x.copy()
        self.lastrun = None

        self.peslast = None
        self.xi = 1.
        self.first = True

        self.H0 = H0
        self._calc_v0ts()

    def _calc_v0ts(self):
        if self.H0 is None:
            self.v0ts = None
            return
        Hw = self.H0 / np.outer(self.sqrtm, self.sqrtm)
        _, vecs = eigh(Hw)
        self.v0ts = self.dx * vecs[:, 0] / self.sqrtm

    def irun(self, fmax=0.05, steps=None, direction='forward'):
        if direction not in ['forward', 'reverse']:
            raise ValueError('direction must be one of "forward" or '
                             '"reverse"!')

        if self.v0ts is None:
            # Initial diagonalization
            self.pes.diag(**self.peskwargs)
            self.H0 = self.pes.H.copy()
            self._calc_v0ts()
            self.peslast = self.pes.last.copy()
        else:
            # Or, restore from last diagonalization for new direction
            self.pes.x = self.x0.copy()
            self.pes.H = self.H0.copy()
            if self.peslast is not None:
                self.pes.last = self.peslast.copy()

        if direction == 'forward':
            self.d1 = self.v0ts.copy()
        elif direction == 'reverse':
            self.d1 = -self.v0ts.copy()

        self.first = True
        return Optimizer.irun(self, fmax, steps)

    def run(self, fmax=0.05, steps=None, direction='forward'):
        for converged in self.irun(fmax, steps, direction):
            pass
        return converged

    def step(self):
        x0 = self.pes.x.copy()
        Tm = self.pes.Tm.copy()
        g1 = self.pes.last['g']
        for n in range(100):
            if self.first:
                epsnorm = np.linalg.norm(self.d1)
                bound_clip = True
                self.first = False
            else:
                eps, self.xi, bound_clip = rs_newton_irc(self.pes, self.sqrtm,
                                                         g1, self.d1, self.dx,
                                                         self.xi)
                epsnorm = np.linalg.norm(eps)
                self.d1 += eps
            f1, g1 = self.pes.evaluate(x0 + self.d1)

            d1m = self.d1 * self.sqrtm
            d1m /= np.linalg.norm(d1m)
            g1m = ((Tm @ Tm.T) @ g1) / self.sqrtm
            g1m /= np.linalg.norm(g1m)
            dot = np.abs(d1m @ g1m)
            if bound_clip:
                if abs(1 - dot) < self.irctol:
                    break
            else:
                if self.converged():
                    break
        else:
            raise RuntimeError("Inner IRC loop failed to converge")

        g1w = g1 / self.sqrtm
        d1w = -self.dx * g1w / np.linalg.norm(g1w)
        self.d1 = d1w / self.sqrtm

    def converged(self, forces=None):
        Hproj = self.pes_cons.Tm.T @ self.pes.H @ self.pes_cons.Tm
        lams, _ = np.linalg.eigh(Hproj)
        return Optimizer.converged(self, forces) and lams[0] > 0
