#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np
from scipy.linalg import eigh

from sella.peswrapper import PESWrapper

from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Bohr
from ase.io import Trajectory
from ase.optimize.optimize import Optimizer


def rs_newton_irc(pes, sqrtm, g, d1, dx, xi=1.):
    lams = pes.lams
    vecs = pes.Tm @ pes.vecs

    L = np.abs(lams)
    Vg = vecs.T @ g

    W = np.diag(sqrtm**2)

    xilower = 0
    xiupper = None

    eps = -vecs @ (Vg / L)
    dxw = (d1 + eps) * sqrtm
    if np.linalg.norm(dxw) < dx:
        return eps, xi, False

    Hproj = vecs @ np.diag(L) @ vecs.T

    for _ in range(100):
        Hshift = Hproj + xi * W
        T0g = np.linalg.solve(Hshift, g)
        T0W = np.linalg.solve(Hshift, W)
        eps = -T0g - xi * T0W @ d1
        deps = -T0W @ (d1 + eps)

        dxw = (d1 + eps) * sqrtm
        dxwmag = np.linalg.norm(dxw)
        if abs(dxwmag - dx) < 1e-14 * dx:
            break
        dmagdxi = 2 * (d1 + eps) @ W @ deps
        dxi = (dx - dxwmag) / dmagdxi

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

    return eps, xi, True


class IRC(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=False, irctol=1e-4, dx=0.1,
                 eta=1e-4, gamma=0.4, peskwargs=None, **kwargs):
        if isinstance(atoms, PESWrapper):
            self.pes = atoms
            atoms = self.pes.atoms
        else:
            self.pes = PESWrapper(atoms, atoms.calc, **kwargs)
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

        self.lastrun = None
        self.x0 = self.pes.x.copy()
        self.v0ts = None
        self.H0 = None
        self.peslast = None
        self.xi = 1.
        self.first = True

    def irun(self, fmax=0.05, steps=None, direction='forward'):
        if direction not in ['forward', 'reverse']:
            raise ValueError('direction must be one of "forward" or '
                             '"reverse"!')

        if self.v0ts is None:
            # Initial diagonalization
            self.pes.diag(**self.peskwargs)
            Hw = self.pes.H / np.outer(self.sqrtm, self.sqrtm)
            _, vecs = eigh(Hw)
            self.v0ts = self.dx * vecs[:, 0] / self.sqrtm
            self.H0 = self.pes.H.copy()
            self.peslast = self.pes.last.copy()
        else:
            # Or, restore from last diagonalization for new direction
            self.pes.x = self.x0.copy()
            self.pes.H = self.H0.copy()
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
        for _ in range(100):
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
            print('Epsnorm is {}, dot product is {}, bound clip?: {}'
                  ''.format(epsnorm, dot, bound_clip))
            if bound_clip and abs(1 - dot) < self.irctol:
                print('Found new point on IRC')
                break
        else:
            raise RuntimeError("Inner IRC loop failed to converge")
        g1w = g1 / self.sqrtm
        d1w = -self.dx * g1w / np.linalg.norm(g1w)
        self.d1 = d1w / self.sqrtm

    def converged(self, forces=None):
        return Optimizer.converged(self, forces) and self.pes.lams[0] > 0
