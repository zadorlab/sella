#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np
from scipy.linalg import eigh

from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Bohr
from ase.io import Trajectory

def rs_newton_irc(pes, sqrtm, g, d1, dx, xi=1.):
    lams = pes.lams
    vecs = pes.Tm @ pes.vecs

    L = np.abs(lams)
    Vg = vecs.T @ g

    W = np.diag(sqrtm**2)
    Wd1 = W @ d1

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
        #deps = -(T0 @ W @ (d1 + eps))

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


def irc(pes, maxiter, ftol, irctol=1e-4, dx=0.2 * Bohr, direction='both', **kwargs):
    try:
        from ase.data import atomic_masses_common
    except ImportError:
        warnings.warn("The version of ASE that is installed does not contain "
                      "the most common isotope masses, so Earth-abundance-"
                      "averaged masses will be used instead!")
        mass_type = "defaults"
    else:
        mass_type = "most_common"

    pes.atoms.set_masses(mass_type)

    if direction not in ['forward', 'reverse', 'both']:
        raise ValueError("Don't understand direction='{}'".format(direction))

    conf = pes.atoms.copy()
    calc = SinglePointCalculator(conf, **pes.atoms.calc.results)
    conf.set_calculator(conf)
    path = [conf]
    f1, g1 = pes.evaluate(pes.x.copy())

    pes.diag(**kwargs)

    if np.linalg.norm(g1) > ftol:
        warnings.warn('Initial forces are greater than convergence tolerance! '
                      'Are you sure this is a transition state?')

    # Square root of the mass array, not to be confused with
    # scipy.linalg.sqrtm, the matrix square root function.
    sqrtm = np.sqrt(pes.atoms.get_masses()[:, np.newaxis]
                    * np.ones_like(pes.atoms.positions)).ravel()

    Hw = pes.H / np.outer(sqrtm, sqrtm)
    _, vecs = eigh(Hw)
    d1w = dx * vecs[:, 0]
    d1 = d1w / sqrtm

    if direction == 'forward':
        return _irc_half(pes, maxiter, ftol, d1, irctol, dx, sqrtm, 'forward')
    elif direction == 'reverse':
        return _irc_half(pes, maxiter, ftol, -d1, irctol, dx, sqrtm, 'reverse')
    elif direction == 'both':
        x0 = pes.x.copy()
        last = pes.last.copy()
        H = pes.H.copy()
        path = list(reversed(_irc_half(pes, maxiter, ftol, d1, irctol, dx, sqrtm, 'forward')))
        pes.x = x0
        pes.last = last
        pes.H = H
        path += _irc_half(pes, maxiter, ftol, -d1, irctol, dx, sqrtm, 'reverse')[1:]
        return path

def _irc_half(pes, maxiter, ftol, d1, irctol, dx, sqrtm, label='forward'):
    traj = Trajectory('sella_irc_{}.traj'.format(label), 'w', pes.atoms)
    f1 = pes.last['f']
    g1 = pes.last['g']
    traj.write(pes.atoms, energy=f1, forces=-g1.reshape((-1, 3)))
    path = []
    # Outer loop finds all points along the MEP
    xi = 1.
    converged = False
    first = True
    x0 = pes.x.copy()
    while True:
        # Back up current geometry and projection matrix, Tm
        Tm = pes.Tm.copy()
        # Inner loop optimizes each individual point along the MEP
        for _ in range(100):
            #f1last, g1last, d1last = f1, g1.copy(), d1.copy()
            if first:
                epsnorm = np.linalg.norm(d1)
                bound_clip = True
                first = False
            else:
                eps, xi, bound_clip = rs_newton_irc(pes, sqrtm, g1, d1, dx, xi)
                epsnorm = np.linalg.norm(eps)
                d1 += eps
                #d1 = (Tm @ Tm.T) @ (d1 + eps)
            f1, g1 = pes.evaluate(x0 + d1)
            #if f1 > f1last:
            #    f1, g1, d1 = f1last, g1last.copy(), d1last.copy()
            #    continue

            if np.linalg.norm(g1) < ftol and pes.lams[0] > 0:
                print('IRC done')
                converged = True
                break

            d1m = d1 * sqrtm
            g1m = ((Tm @ Tm.T) @ g1) / sqrtm
            dot = np.abs(d1m @ g1m) / (np.linalg.norm(d1m) * np.linalg.norm(g1m))
            print('Epsnorm is {}, dot product is {}, bound clip?: {}'.format(epsnorm, dot, bound_clip))
            if bound_clip and abs(1 - dot) < irctol:
                print('Found new point on IRC')
                break
        else:
            raise RuntimeError("Inner IRC loop failed to converge")
        traj.write(pes.atoms, energy=f1, forces=-g1.reshape((-1, 3)))

        conf = pes.atoms.copy()
        calc = SinglePointCalculator(conf, **pes.atoms.calc.results)
        conf.set_calculator(calc)
        path.append(conf)
        if converged:
            return path

        x0 += d1
        g1w = g1 / sqrtm
        d1w = -dx * g1w / np.linalg.norm(g1w)
        d1 = d1w / sqrtm
