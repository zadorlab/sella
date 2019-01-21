#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy
from scipy.linalg import eigh, lstsq
from .eigensolvers import lobpcg, NumericalHessian, ProjectedMatrix, atoms_tr_projection, exact, davidson

def p_rfo(minmode, x0, maxiter, ftol, **kwargs):
    d = len(x0)
    f, g = minmode.f_minmode(x0, **kwargs)
    x = x0.copy()

    flast = f
    xlast = x.copy()
    glast = g.copy()

    eps1 = 1e-4
    eps2 = 1e-5

    Lb = 0.
    Ub = 20.
    Sf = 2.
    Sfdown = np.sqrt(3)

    delta_e = 0.75
    delta_i = 0.8

    #re_l = Lb + delta_e
    re_l = 0
    re_u = Ub - delta_e
    ri_l = Lb + delta_i
    ri_u = Ub - delta_i

    A1 = np.zeros((2, 2))
    A2 = np.zeros((d, d))
    B1 = np.eye(2)
    B2 = np.eye(d)

    dq = np.zeros(d)

    R = 0.1
    Rmax = 0.2

    xlast = x.copy()
    lastfailed = False

    while True:
        alpha = 1.
        
        H = minmode.H
        lams = minmode.lams
        vecs = minmode.vecs

        if minmode.molecule:
            trvecs = atoms_tr_projection(x)
            H = minmode.H + 2 * lams[-1] * trvecs @ trvecs.T
            lams, vecs = np.linalg.eigh(H)

        g_normal = np.dot(g, vecs)



        A1[0, 1] = A1[1, 0] = g_normal[0]
        A1[1, 1] = lams[0]
        
        A2[0, 1:] = A2[1:, 0] = g_normal[1:]
        A2[1:, 1:] = np.diag(lams[1:])
        
        alphalast = 1.
        alpha = 1.
        mu = 0

        def f_dx(alpha, opt=False):
            B1[1, 1] = alpha
            B2[1:, 1:] = np.eye(d - 1) * alpha
            l1, v1 = eigh(A1, b=B1)
            l2, v2 = eigh(A2, b=B2)

            dq[0] = v1[1, 1] / v1[0, 1]
            dq[1:] = v2[1:, 0] / v2[0, 0]
            dx = vecs @ dq
            dx_norm = np.linalg.norm(dx)
            if not opt:
                return dx, dx_norm, l1, v1, l2, v2

            dqmax = (g_normal[0] / (lams[0] - l1[1] * alpha))**2
            dqmin = np.sum((g_normal[1:] / (lams[1:] - l2[0] * alpha))**2)
            ddqda = 2 * l1[1] / (1 + dqmax * alpha) * (g_normal[0]**2 / (lams[0] - l1[1] * alpha)**3)
            ddqda += 2 * l2[0] / (1 + dqmin * alpha) * np.sum(g_normal[1:]**2 / (lams[1:] - l2[0] * alpha)**3)
            dxdalpha = -2 * (R * dx_norm - dx_norm**2) / ddqda
            return dx_norm, dxdalpha

        dx, dx_norm, l1, v1, l2, v2 = f_dx(1)
        if dx_norm > R:            
            dx_norm, dxdalpha = f_dx(alpha, True)
            if dxdalpha < 0.:
                alpha_lower = 1.
                alpha_upper = None
            else:
                alpha_lower = 0.
                alpha_upper = 1.
            mu = 0
            while True:
                if alpha_upper is None:
                    alpha -= dxdalpha
                else:
                    alpha = 0.5 * (alpha_lower + alpha_upper)
                dx_norm, dxdalpha = f_dx(alpha, True)
                mu += 1
                if abs(dx_norm - R) < 0.01 * R:
                    break

                if dxdalpha < 0.:
                    alpha_lower = alpha
                else:
                    alpha_upper = alpha
            dx, dx_norm, l1, v1, l2, v2 = f_dx(alpha)

        df_pred = 0.5 * (l1[1] / v1[0, 1]**2 + l2[0] / v2[0, 0]**2)
#        f, g = minmode.f(x + dx)
        f, g = minmode.f_update(x + dx)

        r = (f - flast) / df_pred
        if (r < re_l) or (r > re_u):
            print("Scaling back R:", r, R, dx_norm)
            R = dx_norm / Sfdown
        elif (ri_l <= r <= ri_u) and abs(dx_norm - R) <= eps1:
            R = min(R * np.sqrt(Sf), Rmax)

        if r < Lb or r > Ub:
#            if lastfailed:
#                print("Search failed, backtracking:", r, R, dx_norm)
##                minmode.f_minmode(x, **kwargs)
            f = flast
            g = glast.copy()
            lastfailed = True
            continue

        x += dx
        flast = f
        xlast = x.copy()
        glast = g.copy()

        minmode.f_minmode(x, **kwargs)

        print(f, np.linalg.norm(g), R, np.sign(lams[0]) * np.sqrt(np.abs(lams[0])) * 3276.4983)
        lastfailed = False

        if np.linalg.norm(minmode.glast) < ftol:
            return x

def rfo(minmode, x0, maxiter, ftol, **kwargs):
    d = len(x0)
    f, g = minmode.f_minmode(x0, **kwargs)
    x = x0.copy()
    lam = minmode.lam
    AH = np.zeros((d + 1, d + 1))
    dx_max = 0.1

    while True:
        AH[1:, 1:] = minmode.H
        AH[1:, 0] = AH[0, 1:] = g
        lam_rfo, vecs_rfo = eigh(AH)
        dx = vecs_rfo[1:, 0] / vecs_rfo[0, 0]
        dx *= min(1.0, dx_max / np.linalg.norm(dx))
        x += dx
        f, g = f, g = minmode.f_minmode(x, **kwargs)
        gnorm = np.linalg.norm(g)
        print(f, gnorm, np.sign(minmode.lam) * np.sqrt(np.abs(minmode.lam)) * 3276.4983)
        if gnorm < ftol:
            return x


def rs_prfo(minmode, g, r_tr, order=1, xi=1.):
    d = len(g)
    lams = minmode.lams
    vecs = minmode.vecs
    g_proj = vecs.T @ g

    A1 = np.zeros((order + 1, order + 1))
    r1 = np.arange(1, order + 1)  # diagonal indices excluding [0,0]

    A1[0, 1:] = A1[1:, 0] = g_proj[:order]
    A1[r1, r1] = lams[:order]

    A2 = np.zeros((d - order + 1, d - order + 1))
    r2 = np.arange(1, d - order + 1)  # diagonal indices excluding [0,0]

    A2[0, 1:] = A2[1:, 0] = g_proj[order:]
    A2[r2, r2] = lams[order:]

    B1 = np.eye(order + 1)
    B2 = np.eye(d - order + 1)

    xi = 1.

    bound_clip = False
    while True:
        l1, v1 = eigh(A1, B1)
        l2, v2 = eigh(A2, B2)

        dx_proj = np.zeros_like(g_proj)
        dx_proj[:order] = v1[1:, -1] / v1[0, -1]
        dx_proj[order:] = v2[1:, 0] / v2[0, 0]

        dx = vecs @ dx_proj
        dx_mag = np.linalg.norm(dx)
        #print(xi, dx_mag / r_tr)

        if not bound_clip:
            if dx_mag < r_tr:
                break
            else:
                xilower = 0
                xiupper = None
                bound_clip = True
                continue
        else:
            if abs(dx_mag - r_tr) < 1e-10 * r_tr:
                break

            if dx_mag > r_tr:
                xilower = xi
            else:
                xiupper = xi

            if xiupper is None:
                xi *= 2
            else:
                xi = (xilower + xiupper) / 2.

            B1[r1, r1] = xi
            B2[r2, r2] = xi

    de_pred = (l1[-1] / v1[0, -1]**2 + l2[0] / v2[0, 0]**2) / 2.
    return dx, dx_mag, xi, bound_clip, de_pred
        

def rs_newton(minmode, g, r_tr, order=1, xi=1.):
    """Perform a trust-radius Newton step towards an
    arbitrary-order saddle point (use order=0 to seek a minimum)"""
    lams = minmode.lams
    vecs = minmode.vecs
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

    de_pred = g.T @ dx + (dx.T @ minmode.H @ dx) / 2.
    return dx, dx_mag, xi, bound_clip, de_pred

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
    #D = 6 * df * (gdxtot - df) - gdxtot**2 - 2 * gdx0 * gdx1
    D = -6 * df**2 + 6 * df * (gdx0 + gdx1) - gdx0**2 - 4 * gdx0 * gdx1 - gdx1**2
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


def berny(minmode, x0, maxiter, ftol, nqn=0, qntol=0.05,
          r_trust=0.2, inc_factr=0.8, dec_factr=0.8, dec_lb=0., dec_ub=5.,
          inc_lb=0.8, inc_ub=1.2, order=1, **kwargs):
    d = len(x0)

    r_trust_min = kwargs.get('dxL', r_trust / 100.)

    x = x0.copy()
    f1, g1 = minmode.f_minmode(x, **kwargs)
    xlast = x.copy()

    ## Detect if x0 lies on a ridge; if it is, push it off
    #print(minmode.lams)
    #ridge = False
    #for i, lam in enumerate(minmode.lams):
    #    if lam > 0:
    #        break
    #    if minmode.vecs[:, i].T @ g1 < 1e-3 * np.linalg.norm(g1):
    #        print("Ridge found!")
    #        ridge = True
    #        x += 1e-2 * minmode.vecs[:, i]

    #if ridge:
    #    f1, g1 = minmode.f_minmode(x, **kwargs)

    f, g = f1, g1

    I = np.eye(d)

    evnext = False

    gnormlast = np.linalg.norm(g)

    n = 1
    xi = 1.
    while True:
        dx, dx_mag, xi, bound_clip, df_pred = rs_newton(minmode, g, r_trust, order, xi)
        #dx, dx_mag, xi, bound_clip, df_pred =  rs_prfo(minmode, g, r_trust, order, xi)

        f0, g0 = f, g
        H0 = minmode.H.copy()
        xlast = x.copy()
        ev = (minmode.lams[0] > 0 and order > 0) or evnext or (nqn > 0 and n % nqn == 0)
        f1, g1, dx1 = minmode.kick(dx, ev, **kwargs)
        n += 1
        if ev:
            n = 1

        if np.linalg.norm(g1) < ftol:
            return minmode.xlast
    
        method = None
        alpha = 1.

        # FIXME: this looks like it's wrong because of poor naming conventions.
        # Rename minmode.xlast to minmode.x.
        dx_actual = minmode.dx(xlast)

        try:
            f, g, alpha = interpolate_quartic_constrained(f0, f1, g0, g1, dx_actual, r_trust)
        except ValueError:
            pass
        else:
            method = 'quartic'

        if method is None:
            try:
                f, g, alpha = interpolate_cubic(f0, f1, g0, g1, dx_actual, r_trust)
            except ValueError:
                pass
            else:
                method = 'cubic'

        if method is None:
            try:
                f, g, alpha = interpolate_quadratic(f0, f1, g0, g1, dx_actual, r_trust)
            except ValueError:
                pass
            else:
                method = 'quadratic'

        if method is None:
            f, g = f1, g1

        if alpha < 0:
            raise RuntimeError("Extrapolation wants to go backwards! This should never happen.")

        evnext = False
        reeval = False
        ratio = minmode.ratio
        if ratio < dec_lb or ratio > dec_ub:
            r_trust = max(dx_mag * dec_factr, r_trust_min)
            reeval = True
        elif bound_clip and inc_lb < ratio < inc_ub:
            r_trust /= inc_factr

        if reeval:
            f, g, v1 = minmode.kick((1 - alpha) * dx1)
            x = minmode.xlast.copy()
        else:
            x = minmode.xpolate(alpha)

        gnorm = np.linalg.norm(g)
        gnormlast = gnorm
        
        print(f1, np.linalg.norm(g1), ratio, minmode.ratio, alpha, dx_mag / r_trust, r_trust, method, minmode.lams[0])

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
        #self.HG = vecs @ np.diag(L) @ vecs.T @ self.G
        #self.GTG = self.HG.T @ self.HG
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
            # only interpolate, and ensure latest point contributes to fit
            #if np.all(self._c >= 0.) and self._c[self.n - 1] > 1. / self.n:
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
    I = np.eye(d)

    xi = 0.5

    neval = 0
    lam_last = 1
    while True:
        dx, dx_mag, xi, bound_clip, df_pred = rs_newton(minmode, gdiis.g, r_trust, order, xi)

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
        #if gnorm > gnormlast:
        #    allow_increase = False

        x = x1

        if ratio < dec_lb or ratio > dec_ub:
            gdiis.n = 0
            r_trust = dx_mag * dec_factr
            f1, g1 = minmode.f_update(x)
        elif bound_clip and inc_lb < ratio < inc_ub:
            r_trust /= inc_factr

        gdiis.update(f1, x1, g1, minmode)
        #if bound_clip:
        #    gdiis.n = 1

        print(f1, np.linalg.norm(g1), ratio, dx_mag/r_trust, r_trust, lams[0])

def lbfgs(minmode, x0, maxiter, ftol, inicurv=1., maxls=10, exact_diag=False, **kwargs):
    d = len(x0)
    invcurv = 1./inicurv
    f, g = minmode.f_dimer(x0, **kwargs)
    c2 = 0.9
    dx_max = 0.1

    x = x0.copy()

    converged = False
    m = 5
    ys = np.zeros((m, d))
    ss = np.zeros((m, d))
    rhos = np.zeros(m)
    alphas = np.zeros(m)
    nhist = -1
    while True:
        q = g.copy()
        for i in range(nhist, -1, -1):
            alphas[i] = rhos[i] * np.dot(ss[i], q)
            q -= alphas[i] * ys[i]
        dx = q * invcurv
        for i in range(nhist + 1):
            beta = rhos[i] * np.dot(ys[i], dx)
            dx += ss[i] * (alphas[i] - beta)
        dx *= -1

        # Scale step size
        dx_mag = np.linalg.norm(dx)
        dx *= min(1.0, dx_max / dx_mag)
        dx_mag = min(dx_mag, dx_max)

        gdx = np.dot(g, dx)
        if gdx > 0:
            nhist = 0
            dx = -g / inicurv
            dx_mag = np.linalg.norm(dx)
            dx *= min(1.0, dx_max / dx_mag)
            dx_mag = min(dx_mag, dx_max)
            gdx = np.dot(g, dx)

        lam0 = minmode.lams[0]

        f1, g1 = minmode.f_dimer(x + dx, **kwargs)
        g1dx = g1 @ dx

        gdxlow = gdx
        gdxhigh = g1dx
        if converged:
            return x + dx
        lam1 = minmode.lams[0]
#        if lam0 < 0 and lam1 > 0:
#            f1, g1 = minmode.f_dimer(x + dx, maxres=0., **kwargs)
#            print("Approx lam: {}, exact lam: {}".format(lam1, minmode.lams[0]))

        taulow, tauhigh = 0., 1.
        tau = 1.

        ls_failed = False
        if (lam0 < 0 and lam1 > 0) or (lam0 > 0 and lam1 < 0):
            ls_failed = True
        
        if g1dx > 0 or np.abs(dx_mag - dx_max) > 1e-5:
            k = 0
            while np.abs(g1dx) > c2 * np.abs(gdx) and not ls_failed:
                tau = (taulow * gdxhigh - tauhigh * gdxlow) / (gdxhigh - gdxlow)
                # if we hit the max step size, but the extrapolation wants us to keep going,
                # abort with the max step size.
                if tau < 1e-8:
                    tau = tauhigh * 1.5
                tau = min(tau, tauhigh * 1.5, dx_max / dx_mag)
                if np.abs(tauhigh * dx_mag - dx_max) < 1e-4 and tau - tauhigh > -1e-4:
                    ls_failed = True
                    tau = dx_max / dx_mag
                    break
                f1, g1 = minmode.f_dimer(x + tau * dx, **kwargs)
                k += 1
                g1dx = np.dot(g1, dx)
                print(f1, gdx, taulow, gdxlow, tauhigh, gdxhigh, tau, g1dx, dx_max / dx_mag, lam0, lam1, minmode.lams[0])
                if g1dx > 0:
                    tauhigh = tau
                    gdxhigh = g1dx
                elif gdxhigh > 0:
                    taulow = tau
                    gdxlow = g1dx
                else:
                    if tau < tauhigh:
                        taulow = tau
                        gdxlow = g1dx
                    else:
                        taulow = tauhigh
                        gdxlow = gdxhigh
                        tauhigh = tau
                        gdxhigh = g1dx
                if (lam0 < 0 and minmode.lams[0] > 0) or (lam0 > 0 and minmode.lams[0] < 0):
                    ls_failed = True
                    break
                if k == maxls:
                    ls_failed = True
                    break
            else:
                ls_failed = False

        dx *= tau

        x += dx

        if ls_failed:
            invcurv = 1 / inicurv
            nhist = 0
        elif gdx < g1dx:
            if nhist >= m - 1:
                ys = np.roll(ys, -1, 0)
                ss = np.roll(ss, -1, 0)
                rhos = np.roll(rhos, -1, 0)
            else:
                nhist += 1

            ys[nhist] = g1 - g
            ss[nhist] = dx
            rhos[nhist] = 1. / np.dot(ys[nhist], ss[nhist])

            invcurv = np.dot(ss[nhist], ys[nhist]) / np.dot(ys[nhist], ys[nhist])
        else:
            print("Failed to update Hessian")
        f, g = f1, g1

        print(f, np.linalg.norm(g), tau, np.sign(minmode.lams[0]) * np.sqrt(np.abs(minmode.lams[0])) * 3276.4983)

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
