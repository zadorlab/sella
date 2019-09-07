#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np
from scipy.linalg import eigh

from ase.calculators.singlepoint import SinglePointCalculator

def rs_newton_irc(pes, sqrtm, g, d1, dx, xi=1.):
    lams = pes.lams
    vecs = pes.Tm @ pes.vecs
    mass = sqrtm**2

    L = np.abs(lams)
    Vg = vecs.T @ g
    Vd1 = vecs.T @ (sqrtm**2 * d1)
    Vw = vecs.T @ (sqrtm**2)

    W = np.diag(sqrtm**2)
    Wg = W @ g
    Wd1 = W @ d1

    xilower = 0
    xiupper = None

    eps = -vecs @ (Vg / L)
    dxw = (d1 + eps) * sqrtm
    if np.linalg.norm(dxw) < dx:
        return eps, xi, False

    Hproj = (vecs @ vecs.T) @ pes.H @ (vecs @ vecs.T)

    for _ in range(100):
        #if xiupper is None:
        #    xi *= 2
        #else:
        #    xi = (xilower + xiupper) / 2.
        T0 = np.linalg.inv(Hproj + xi * W)
        eps = -T0 @ (g + xi * Wd1)
        deps = -(T0 @ W @ (d1 + eps))

        #eps = -vecs @ ((Vg + xi * Vd1) / (L + xi * Vw))
        dxw = (d1 + eps) * sqrtm
        dxwmag = np.linalg.norm(dxw)
        #print(xi, dxwmag, dx, xiupper, xilower)
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

        #if dxwmag > dx:
        #    xilower = xi
        #else:
        #    xiupper = xi
    else:
        raise RuntimeError("IRC Newton step failed!")

    return eps, xi, True


def irc(pes, maxiter, ftol, dx=0.1, direction='both', diag=True, **kwargs):
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

    x = pes.x_m.copy()
    conf = pes.atoms.copy()
    calc = SinglePointCalculator(conf, **pes.atoms.calc.results)
    conf.set_calculator(conf)
    path = [conf]
    f1, g1, _ = pes.kick(np.zeros_like(x))

    if diag:
        pes.diag(**kwargs)

    if np.linalg.norm(g1) > ftol:
        warnings.warn('Initial forces are greater than convergence tolerance! '
                      'Are you sure this is a transition state?')

    if direction == 'both':
        x0 = pes.x.copy()
        H = pes.H.copy()
        last = pes.last.copy()
        fpath = irc(pes, maxiter, ftol, dx, 'forward', diag=False, **kwargs)
        pes.x = x0
        pes.H = H
        pes.last = last
        rpath = irc(pes, maxiter, ftol, dx, 'reverse', diag=False, **kwargs)
        return list(reversed(fpath)) + rpath[1:]

    # Square root of the mass array, not to be confused with
    # scipy.linalg.sqrtm, the matrix square root function.
    sqrtm = np.sqrt(pes.atoms.get_masses()[:, np.newaxis]
                    * np.ones_like(pes.atoms.positions)).ravel()

    Hw = pes.H / np.outer(sqrtm, sqrtm)
    _, vecs = eigh(Hw)
    d1w = dx * vecs[:, 0]
    d1 = d1w / sqrtm

    if direction == 'reverse':
        d1 *= -1

    xi = 1.

    # Outer loop finds all points along the MEP
    while True:
        # Back up current geometry and projection matrix, Tm
        Tm = pes.Tm.copy()
        x0 = pes.x.copy()
        f1, g1 = pes.evaluate(x0 + d1)
        # Inner loop optimizes each individual point along the MEP
        for _ in range(100):
            eps, xi, bound_clip = rs_newton_irc(pes, sqrtm, g1, d1, dx, xi)
            epsnorm = np.linalg.norm(eps)
            #if epsnorm < 1e-4:
            #    break
            d1 += eps
            f1, g1 = pes.evaluate(x0 + d1)
            d1m = d1 * sqrtm
            g1m = ((Tm @ Tm.T) @ g1) / sqrtm
            dot = np.abs(d1m @ g1m) / (np.linalg.norm(d1m) * np.linalg.norm(g1m))
            print('Epsnorm is {}, dot product is {}'.format(epsnorm, dot))
            if bound_clip and abs(1 - dot) < 1e-4:
                break
            elif not bound_clip and np.linalg.norm(g1) < ftol:
                break
        else:
            raise RuntimeError("Inner IRC loop failed to converge")
        g1w = g1 / sqrtm
        d1w = -dx * g1w / np.linalg.norm(g1w)
        d1 = d1w / sqrtm

        conf = pes.atoms.copy()
        calc = SinglePointCalculator(conf, **pes.atoms.calc.results)
        conf.set_calculator(calc)
        path.append(conf)
        if np.all(pes.lams > 0) and pes.converged(ftol):
            return path
