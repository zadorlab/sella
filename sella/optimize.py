#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np
import scipy
from scipy.linalg import eigh, lstsq


def rs_newton(minmode, g, r_tr, order=1, xi=1.):
    """Perform a trust-radius Newton step towards an
    arbitrary-order saddle point (use order=0 to seek a minimum)"""
    lams = minmode.lams
    vecs = minmode.vecs

    # If we don't have any curvature information yet, just do steepest
    # descent.
    if lams is None:
        dx = -g
        dx_mag = np.linalg.norm(dx)
        bound_clip = False
        if dx_mag > r_tr:
            dx *= r_tr / dx_mag
            dx_mag = r_tr
            bound_clip = True
        return dx, dx_mag, xi, bound_clip

    L = np.abs(lams)
    L[:order] *= -1
    Vg = vecs.T @ g
    dx = -vecs @ (Vg / L)
    dx_mag = np.linalg.norm(dx)
    bound_clip = False
    if dx_mag > r_tr:
        bound_clip = True
        rfo_counter = 0
        xilower = 0
        xiupper = None
        LL = L * L
        while True:
            if xiupper is None:
                xi *= 2
            else:
                xi = (xilower + xiupper) / 2.
            if order == 0:
                dx = -vecs @ (Vg / (L + xi))
            else:
                dx = -vecs @ (Vg * (L / (LL + xi)))
            dx_mag = np.linalg.norm(dx)
            if abs(dx_mag - r_tr) < 1e-14 * r_tr:
                break

            if dx_mag > r_tr:
                xilower = xi
            else:
                xiupper = xi

    return dx, dx_mag, xi, bound_clip


# These interpolators are not currently being used, but we'll keep them
# in case we find a use for them later.
def interpolate_quartic_constrained(f0, f1, g0, g1, dx, rmax=np.infty):
    """Constructs a 1-D quartic interpolation between two points given
    the function value and directional gradient at the endpoints, and
    finds and returns the extremum"""

    tmax = rmax / np.linalg.norm(dx)

    # Quartic interpolation is only possible if the descriminant D
    # is non-negative
    df = f1 - f0
    gdx0 = g0.T @ dx
    gdx1 = g1.T @ dx
    gdxtot = gdx0 + gdx1
    D = -6 * df**2 + 6*df*(gdx0 + gdx1) - gdx0**2 - 4*gdx0*gdx1 - gdx1**2
    if D < 0:
        raise ValueError
    dg = g1 - g0

    # There are two possible quartic fits. We will test both, and use
    # the one that results in the largest predicted function value
    # change.
    sqrt2D = np.sqrt(2 * D)
    a_1 = 0.5 * (gdx1 - gdx0 - sqrt2D)
    a_2 = 0.5 * (gdx1 - gdx0 + sqrt2D)
    b_1 = -2 * a_1 - 2 * df + gdx0 + gdx1
    b_2 = -2 * a_2 - 2 * df + gdx0 + gdx1
    c_1 = a_1 + 3 * df - 2 * gdx0 - gdx1
    c_2 = a_2 + 3 * df - 2 * gdx0 - gdx1
    d = gdx0
    e = f0

    roots1 = np.roots([4 * a_1, 3 * b_1, 2 * c_1, d])
    for root in roots1:
        if root.imag == 0.0:
            t1 = min(root.real, tmax)
            break
    else:
        raise RuntimeError

    roots2 = np.roots([4 * a_2, 3 * b_2, 2 * c_2, d])
    for root in roots2:
        if abs(root.imag) < 1e-8:
            t2 = min(root.real, tmax)
            break
    else:
        raise RuntimeError

    ft1 = a_1 * t1**4 + b_1 * t1**3 + c_1 * t1**2 + d * t1 + e
    ft2 = a_2 * t2**4 + b_2 * t2**3 + c_2 * t2**2 + d * t2 + e
    if np.sign(gdx0 * (ft1 - f0)) == -1:
        t1 = -1.
    if np.sign(gdx0 * (ft2 - f0)) == -1:
        t2 = -1.

    if (t1 < 0) and (t2 < 0):
        raise ValueError
    elif (t1 < 0):
        t = t2
        ft = ft2
    elif (t2 < 0):
        t = t1
        ft = ft1
    else:
        # If c is positive, we are looking for a minimum, and
        # if c is negative, we are looking for a maximum.
        if c_1 > 0:
            df1 = min(f0, f1) - ft1
        else:
            df1 = ft1 - max(f0, f1)

        if c_2 > 0:
            df2 = min(f0, f1) - ft2
        else:
            df2 = ft2 - max(f0, f1)

        if df2 > df1:
            t = t2
            ft = ft2
        else:
            t = t1
            ft = ft1

    gt = g0 + t * dg

    return ft, gt, t


def interpolate_cubic(f0, f1, g0, g1, dx, rmax=np.infty):
    tmax = rmax / np.linalg.norm(dx)
    df = f1 - f0
    gdx0 = g0.T @ dx
    gdx1 = g1.T @ dx
    gdxtot = gdx0 + gdx1
    dg = g1 - g0

    a = -2 * df + gdx0 + gdx1
    b = 3 * df - 2 * gdx0 - gdx1
    c = gdx0
    d = f0

    D = b**2 - 3 * a * c
    if D < 0:
        raise ValueError

    rad = np.sqrt(D)
    t1 = min((-b + rad) / (3 * a), tmax)
    t2 = min((-b - rad) / (3 * a), tmax)

    t = None
    if (t1 < 0):
        # if both roots are less than 0, we have a problem
        raise ValueError
    elif (t2 < 0) or ((t1 < 1) and (1 - t1 > t2)):
        t = t1
    else:
        t = t2

    ft = a * t**3 + b * t**2 + c * t + d
    gt = g0 + t * dg

    return ft, gt, t


def interpolate_quadratic(f0, f1, g0, g1, dx, rmax=np.infty):
    tmax = rmax / np.linalg.norm(dx)
    df = f1 - f0
    dg = g1 - g0
    gdx0 = g0.T @ dx
    gdx1 = g1.T @ dx
    if gdx1 / gdx0 > 1:
        t = tmax
    else:
        t = min(gdx0 / (gdx0 - gdx1), tmax)

    ft = f0 + t * df
    gt = g0 + t * dg

    return ft, gt, t


def optimize(minmode, maxiter, ftol, r_trust, inc_factr=1.1, dec_factr=0.9,
             dec_ratio=5.0, inc_ratio=1.01, order=1, eig=True, **kwargs):

    if order != 0 and not eig:
        warnings.warn("Saddle point optimizations with eig=False will "
                      "most likely fail!\n Proceeding anyway, but you "
                      "shouldn't be optimistic.")

    r_trust_min = kwargs.get('dxL', r_trust / 100.)

    f, g, _ = minmode.kick(np.zeros_like(minmode.x_m))

    if eig:
        minmode.f_minmode(**kwargs)

    xi = 1.
    while True:
        # Find new search step
        dx, dx_mag, xi, bound_clip = rs_newton(minmode, g, r_trust, order, xi)

        # Determine if we need to call the eigensolver, then step
        ev = (eig and minmode.lams is not None
                  and np.any(minmode.lams[:order] > 0))
        f, g, dx = minmode.kick(dx, ev, **kwargs)

        # Loop exit criterion: convergence or maxiter reached
        if minmode.converged(ftol) or minmode.calls >= maxiter:
            return minmode.last['x']

        # Update trust radius
        ratio = minmode.ratio
        if ratio is None:
            ratio = 1.
        if ratio < 1/dec_ratio or ratio > dec_ratio:
            r_trust = max(dx_mag * dec_factr, r_trust_min)
        elif bound_clip and 1/inc_ratio < ratio < inc_ratio:
            r_trust *= inc_factr

        # Debug print statement
        print(f, np.linalg.norm(g), ratio, dx_mag / r_trust, r_trust, minmode.lams[0])


class GDIIS(object):
    def __init__(self, d, nhist):
        self.d = d
        self.nhist = nhist

        self.E = np.zeros(nhist, dtype=np.float64)
        self.R = np.zeros((d, nhist), dtype=np.float64)
        self.G = np.zeros((d, nhist), dtype=np.float64)
        self.HG = np.zeros((d, nhist), dtype=np.float64)
        self.GTG = np.zeros((nhist, nhist), dtype=np.float64)

        self._n = 0

    def update(self, e, r, g, minmode):
        vecs = minmode.vecs
        L = abs(minmode.lams)
        L[0] *= -1

        self.E = np.roll(self.E, 1)
        self.R = np.roll(self.R, 1, 1)
        self.G = np.roll(self.G, 1, 1)
        self.HG = np.roll(self.HG, 1, 1)
        self.GTG = np.roll(self.GTG, 1, (0, 1))

        self.E[0] = e
        self.R[:, 0] = r.copy()
        self.G[:, 0] = g.copy()
        self.HG[:, 0] = vecs @ ((vecs.T @ g) / L)
        self.GTG[0, :] = self.GTG[:, 0] = self.HG.T @ self.HG[:, 0]

        self.n = min(self.n + 1, self.nhist)

    def _calc_c(self):
        c = np.zeros(self.nhist)
        c[0] = 1.
        resmin = np.infty
        for mask in mask_gen(self.n):
            c[:self.n] = mask * lstsq(self.GTG[:self.n, :self.n], mask)[0]
            c /= c.sum()
            if np.all(c >= 0.):
                res = np.linalg.norm(self.HG @ c)
                if res < resmin:
                    self._c = c
                    resmin = res
        print(self._c, resmin)

    def reset(self):
        self.n = 1
        self._calc_c()

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        self._n = val
        self._c = None

    @property
    def c(self):
        if self._c is None:
            self._calc_c()
        return self._c

    @property
    def r(self):
        return self.R @ self.c

    @property
    def g(self):
        return self.G @ self.c

    @property
    def e(self):
        return self.E @ self.c


def gediis(minmode, x0, maxcalls, ftol, nqn=0, nhist=10,
           r_trust=0.2, inc_factr=0.8, dec_factr=0.8, dec_lb=0., dec_ub=5.,
           inc_lb=0.8, inc_ub=1.2, gnorm_ev_thr=2., order=1, **kwargs):
    d = len(x0)
    minmode.calls = 0
    f1, g1 = minmode.f_minmode(x0, **kwargs)
    gnorm = np.linalg.norm(g1)
    gnormlast = 0.
    evnext = True

    gdiis = GDIIS(d, nhist)
    gdiis.update(f1, x0, g1, minmode)

    f, g = f1, g1

    xi = 0.5

    neval = 0
    lam_last = 1
    while True:
        dx, dx_mag, xi, bound_clip = rs_newton(minmode, gdiis.g,
                                               r_trust, order, xi)

        lams = minmode.lams
        vecs = minmode.vecs
        allow_increase = True

        L = np.abs(lams)

        x1 = gdiis.r + dx
        if evnext or (nqn > 0 and neval % nqn == 0):
            V0 = dx[:, np.newaxis].copy()
            f1, g1 = minmode.f_minmode(x1, V0=V0, **kwargs)
            neval = 0
            evnext = False
        else:
            f1last, g1last = f1, g1
            f1, g1 = minmode.f_update(x1)

        if minmode.calls >= maxcalls:
            return x1

        neval += 1

        if np.linalg.norm(g1) < ftol:
            return x1

        ratio = minmode.ratio

        gnormlast = gnorm
        gnorm = np.linalg.norm(g1)

        x = x1

        if ratio < dec_lb or ratio > dec_ub:
            gdiis.n = 0
            r_trust = dx_mag * dec_factr
            f1, g1 = minmode.f_update(x)
        elif bound_clip and inc_lb < ratio < inc_ub:
            r_trust /= inc_factr

        gdiis.update(f1, x1, g1, minmode)

        print(f1, np.linalg.norm(g1), ratio, dx_mag/r_trust, r_trust, lams[0])


def mask_gen(n):
    mask = np.ones(n, dtype=int)
    yield mask.copy()
    while True:
        mask[-1] -= 1
        for i in range(n - 2, -1, -1):
            mask[i] += mask[i + 1] // 2
            mask[i + 1] %= 2
        if mask[0] == 0:
            return
        yield mask.copy()
