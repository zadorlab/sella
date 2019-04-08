#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np

from ase.calculators.singlepoint import SinglePointCalculator


def rs_newton_irc(minmode, g, d1, dx, xi=1.):
    lams = minmode.lams
    vecs = minmode.vecs
    L = np.abs(lams)
    Vg = vecs.T @ g
    Vd1 = vecs.T @ d1

    xilower = 0
    xiupper = None

    eps = -vecs @ (Vg / L)
    d2 = d1 + eps
    d2mag = np.linalg.norm(d2)
    if d2mag < dx:
        return eps, xi

    while True:
        if xiupper is None:
            xi *= 2
        else:
            xi = (xilower + xiupper) / 2.
        eps = -vecs @ ((Vg + xi * Vd1) / (L + xi))
        d2 = d1 + eps
        d2mag = np.linalg.norm(d2)
        if abs(d2mag - dx) < 1e-14 * dx:
            break

        if d2mag > dx:
            xilower = xi
        else:
            xiupper = xi

    return eps, xi


def irc(minmode, maxiter, ftol, dx=0.01, direction='both', **kwargs):
    d = minmode.d

    if direction not in ['forward', 'reverse', 'both']:
        raise ValueError("Don't understand direction='{}'".format(direction))

    x = minmode.x_m.copy()
    conf = minmode.atoms.copy()
    calc = SinglePointCalculator(conf, **minmode.atoms.calc.results)
    conf.set_calculator(conf)
    path = [conf]
    f1, g1, _ = minmode.kick(np.zeros_like(x))
    minmode.f_minmode(**kwargs)

    if np.linalg.norm(g1) > ftol:
        warnings.warn('Initial forces are greater than convergence tolerance! '
                      'Are you sure this is a transition state?')

    if direction == 'both':
        x0 = minmode.x.copy()
        H = minmode.H.copy()
        last = minmode.last.copy()
        fpath = irc(minmode, maxiter, ftol, dx, 'forward', **kwargs)
        minmode.x = x0
        minmode.H = H
        minmode.last = last
        rpath = irc(minmode, maxiter, ftol, dx, 'reverse', **kwargs)
        return list(reversed(fpath)) + rpath[1:]

    d1 = minmode.vecs[:, 0]
    d1 *= dx / np.linalg.norm(d1)
    if direction == 'reverse':
        d1 *= -1

    xi = 1.

    # Outer loop finds all points along the MEP
    while True:
        f1, g1, _ = minmode.kick(d1)
        # Inner loop optimizes each individual point along the MEP
        while True:
            eps, xi = rs_newton_irc(minmode, g1, d1, dx, xi)
            epsnorm = np.linalg.norm(eps)
            print('Epsnorm is {}'.format(epsnorm))
            if epsnorm < 1e-4:
                break
            f1, g1, _ = minmode.kick(eps)
            d1 += eps

        conf = minmode.atoms.copy()
        calc = SinglePointCalculator(conf, **minmode.atoms.calc.results)
        conf.set_calculator(calc)
        path.append(conf)
        if np.all(minmode.lams > 0) and minmode.converged(ftol):
            return path
