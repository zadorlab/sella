#!/usr/bin/env python

from __future__ import division

import warnings

import numpy as np
import scipy
from scipy.linalg import eigh, lstsq
from sella.cython_routines import aheig

def _trmag(xi, num, denom):
    """Helper used to find Lagrange multiplier for trust radius methods"""
    s = -num / (denom + xi)
    smag = np.linalg.norm(s)
    dsmag = -(s**2 / (denom + xi)).sum() / smag
    return smag, dsmag

def _root(fun,
          x0,
          target=0.,
          args=tuple(),
          bounds=None,
          tol=1e-15,
          maxiter=100):
    """Finds x > 0 such that fun(x, *args) == target"""
    f, df = fun(x0, *args)
    err = f - target
    x = x0
    if bounds is None:
        bounds = [-np.infty, np.infty]
    else:
        bounds = list(bounds)

    if bounds[0] is None:
        bounds[0] = -np.infty
    if bounds[1] is None:
        bounds[1] = np.infty

    niter = 0
    while abs(err) > tol and niter < maxiter:
        x1 = x - err / df
        if x1 <= bounds[0]:
            x = (x + bounds[0]) / 2.
        elif x1 >= bounds[1]:
            x = (bounds[1] + x) / 2.
        else:
            x = x1
        f, df = fun(x, *args)
        err = f - target
        niter += 1
    if abs(err) > tol:
        raise RuntimeError("Rootfinder failed!")
    return x

def rs_newton(minmode, g, r_tr, order=1, xi=1.):
    """Perform a trust-radius Newton step towards an
    arbitrary-order saddle point (use order=0 to seek a minimum)"""
    # If we don't have any curvature information yet, just do steepest
    # descent.
    if minmode.lams is None:
        dx = -g
        dx_mag = np.linalg.norm(dx)
        bound_clip = False
        if dx_mag > r_tr:
            dx *= r_tr / dx_mag
            dx_mag = r_tr
            bound_clip = True
        return dx, dx_mag, xi, bound_clip

    L = np.abs(minmode.lams)
    L[:order] *= -1
    V = minmode.vecs

    # Standard trust radius
    num = V.T @ g
    denom = L

    # First try with xi=0
    dx = -V @ (num / denom)
    dx_mag = np.linalg.norm(dx)
    # If dx lies within the trust radius, we're done
    if dx_mag <= r_tr:
        return dx, dx_mag, xi, False

    # Gradient minimization trust radius
    # Note: we do this *after* checking the xi=0 case, since then the extra
    # L terms in the numerator and denominator would cancel anyway.
    if order > 0:
        num *= L
        denom *= L

    # scipy.optimize.root_scalar doesn't let us provide one-sided bounds,
    # so we rolled our own!
    xi = _root(_trmag, xi, r_tr, args=(num, denom), bounds=[0., None])
    dx = -V @ (num / (denom + xi))
    dx_mag = np.linalg.norm(dx)
    assert abs(dx_mag - r_tr) < 1e-12
    return dx, dx_mag, xi, True

def lams2vecs(d, z, lams):
    n = len(lams)
    vecs = np.zeros((n, n))
    for i, lam in enumerate(lams[:-1]):
        for j, dj in enumerate(d):
            if np.abs(dj - lam)**2 < 1e-15:
                #raise RuntimeError(i, lam, j, dj)
                vecs[j:i, i] = z[j:i] * z[i] / (z[j:i] @ z[j:i])
                vecs[i, i] = -1
                break
        else:
            vecs[:-1, i] = z / (d - lam)
            vecs[-1, i] = -1
        vecs[:, i] /= np.linalg.norm(vecs[:, i])
    return vecs

def rs_rfo(minmode, g, r_tr, order=1, alpha=0.5):
    lams = minmode.lams
    vecs = minmode.vecs

    gmax = vecs[:, :order].T @ g
    gmin = vecs[:, order:].T @ g

    Hmax0 = np.block([[np.diag(lams[:order]), gmax[:, np.newaxis]],
                      [gmax, 0]])
    Hmin0 = np.block([[np.diag(lams[order:]), gmin[:, np.newaxis]],
                      [gmin, 0]])

    lmax, vmax = eigh(Hmax0)

    lmin, vmin = aheig(lams[order:], gmin, 0)

    smax = vmax[:-1, -1] / vmax[-1, -1]
    smin = vmin[:-1, 0] / vmin[-1, 0]

    s = vecs[:, :order] @ smax + vecs[:, order:] @ smin
    smag = np.linalg.norm(s)

    if smag <= r_tr:
        return s, smag, 1., False

    lower = 0.
    upper = 1.

    dHmax = np.zeros_like(Hmax0)
    dHmax[:, :-1] = 1.
    dHmax[:-1, :] += 1.

    dHmin = np.zeros_like(Hmin0)
    dHmin[:, :-1] = 1.
    dHmin[:-1, :] += 1.

    alpha = r_tr

    n = 1
    while abs(r_tr - smag) > 1e-15:
        if n >= 1000:
            raise RuntimeError('RFO failed!')
        n += 1

        Hmax = Hmax0.copy() * alpha
        Hmin = Hmin0.copy() * alpha

        Hmax[:-1, :-1] *= alpha
        Hmin[:-1, :-1] *= alpha

        lmax, vmax = eigh(Hmax)
        lmin, vmin = aheig(lams[order:] * alpha**2, gmin * alpha, 0)

        smax = vmax[:-1, -1] * alpha / vmax[-1, -1]
        smin = vmin[:-1, 0] * alpha / vmin[-1, 0]
        s = vecs[:, :order] @ smax + vecs[:, order:] @ smin
        smag = np.linalg.norm(s)

        if smag > r_tr:
            upper = alpha
        else:
            lower = alpha

        dHmaxda = Hmax0.copy()
        dHmaxda[:-1, :-1] *= 2 * alpha

        dHminda = Hmin0.copy()
        dHminda[:-1, :-1] *= 2 * alpha

        dvmaxda = vmax[:, :-1] @ ((vmax[:, :-1].T @ dHmaxda @ vmax[:, -1]) / (lmax[-1] - lmax[:-1]))

        dvminda = vmin[:, 1:] @ ((vmin[:, 1:].T @ dHminda @ vmin[:, 0]) / (lmin[0] - lmin[1:]))

        dsmaxda = vmax[:-1, -1] / vmax[-1, -1] + (alpha / vmax[-1, -1]) * dvmaxda[:-1] - (vmax[:-1, -1] * alpha / vmax[-1, -1]**2) * dvmaxda[-1]
        dsminda = vmin[:-1, 0] / vmin[-1, 0] + (alpha / vmin[-1, 0]) * dvminda[:-1] - (vmin[:-1, 0] * alpha / vmin[-1, 0]**2) * dvminda[-1]

        dsmagda = (smin @ dsminda + smax @ dsmaxda) / smag
        err = smag - r_tr

        alpha -=  err / dsmagda
        if np.isnan(alpha) or alpha < lower or alpha > upper:
            alpha = (lower + upper) / 2.

    s = vecs[:, :order] @ smax + vecs[:, order:] @ smin
    return s, smag, alpha, True

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


def optimize(minmode, maxiter, ftol, r_trust=5e-4, inc_factr=1.1,
             dec_factr=0.9, dec_ratio=5.0, inc_ratio=1.01, order=1, eig=True,
             dump_xs_hessians=False, **kwargs):

    if order != 0 and not eig:
        warnings.warn("Saddle point optimizations with eig=False will "
                      "most likely fail!\n Proceeding anyway, but you "
                      "shouldn't be optimistic.")
    r_trust *= len(minmode.x_m)

    r_trust_min = kwargs.get('dxL', r_trust / 100.)

    f, g, _ = minmode.kick(np.zeros_like(minmode.x_m))

    if eig:
        minmode.f_minmode(**kwargs)

    if dump_xs_hessians:
        all_hessians = []
        all_xs = []
        all_xs.append(minmode.last['x'])
        all_hessians.append(minmode.H.copy())

    xi = 1.
    while True:
        # Find new search step
        dx, dx_mag, xi, bound_clip = rs_newton(minmode, g, r_trust, order, xi)
        #dx, dx_mag, xi, bound_clip = rs_rfo(minmode, g, r_trust, order, xi)

        # Determine if we need to call the eigensolver, then step
        ev = (eig and minmode.lams is not None
                  and np.any(minmode.lams[:order] > 0))
        f, g, dx = minmode.kick(dx, ev, **kwargs)
        if dump_xs_hessians:
            all_xs.append(minmode.last['x'])
            all_hessians.append(minmode.H.copy())

        # Loop exit criterion: convergence or maxiter reached
        if minmode.converged(ftol) or minmode.calls >= maxiter:
            if dump_xs_hessians:
                return all_xs, all_hessians
            return minmode.last['x']

        # Update trust radius
        ratio = minmode.ratio
        if ratio is None:
            ratio = 1.
        if ratio < 1./dec_ratio or ratio > dec_ratio:
            r_trust = max(dx_mag * dec_factr, r_trust_min)
        elif 1./inc_ratio < ratio < inc_ratio:
            r_trust = max(inc_factr * dx_mag, r_trust)

        # Debug print statement
        print(f, np.linalg.norm(g), ratio, dx_mag / r_trust, r_trust, xi, minmode.lams[0])


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
