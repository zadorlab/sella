#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal, logm, expm, eigh, sqrtm
from .eigensolvers import lobpcg, NumericalHessian, ProjectedMatrix, atoms_tr_projection, exact, davidson
from .cython_routines import symmetrize_Y2

class MinMode(object):
    def __init__(self, f, d, minmode=davidson, molecule=False, H0=None, v0=None,
                 trshift=1000, trshift_factor=4.):
        self.f = f
        self.d = d
        self.v = v0
        self.calls = 0
        self.lam = 1
        self.minmode = minmode
        self.H = None
        self.xlast = None
        self.glast = None
        self.Hproj = None
        self.molecule = molecule
        self.H0 = H0
        self.shift = trshift
        self.shift_factor = trshift_factor
        self.calls = 0
        self.Hlast = None
        self.Hinv = None

    def f_update(self, x):
        self.calls += 1
        f, g = self.f(x)
        if self.xlast is not None and self.glast is not None:
            self.update_H(x - self.xlast, g - self.glast)
            #self.update_Hinv(x - self.xlast, g - self.glast)
        self.xlast = x.copy()
        self.glast = g.copy()
        return f, g

    def find_saddle_point(self, x0, dxL=1e-5, maxres=5e-3, threepoint=False, bowlescape=True, V0=None, **kwargs):
        d = len(x0)
        
        f0, g0 = self.f_update(x0)
        #f0, g0 = self.f(x0)

        Htrue = NumericalHessian(self.f, x0, g0, dxL, threepoint)

        I = np.eye(self.d)
        if self.molecule:
            T = atoms_tr_projection(x0)
            _, ntr = T.shape
        else:
            T = np.empty((self.d, 0))
            ntr = 0

        P = self.H
        if P is None:
            if self.H0 is not None:
                self.H = self.H0
                P = self.H
            elif self.v is not None:
                P = I - 2 * np.outer(self.v, self.v)
            else:
                u = g0 / np.linalg.norm(g0)
                P = I - 2 * np.outer(u, u)
        H = Htrue

        if self.v is None or np.abs(self.lam) < 1e-8:
            if self.H is not None:
                lams, vecs = eigh(self.H)
                self.v = vecs[:, 0]
            else:
                self.v = np.random.normal(size=d)
                self.v /= np.linalg.norm(self.v)

        v0 = self.v

        lams, Vs, AVs = self.minmode(H, maxres, P, T, V0=V0, shift=self.shift, **kwargs)
        self.calls += H.calls

        #Proj = I - T @ T.T
        if ntr > 0:
            Tnull = scipy.linalg.null_space(T.T)
            Proj = Tnull @ Tnull.T
        else:
            Proj = np.eye(self.d)

        if self.H is None:
            #lam0 = lams[-(ntr + 1)]
            #lam0 = np.sqrt(np.average(lams[:-ntr]**2))
            if ntr == 0:
                lam0 = np.average(np.abs(lams))
            else:
                lam0 = np.average(np.abs(lams[:-ntr]))
            #self.H = lam0 * I + (self.shift - lam0) * T @ T.T
            self.H = lam0 * Proj + self.shift * T @ T.T
            self.Hinv = Proj / lam0 + T @ T.T / self.shift
            #self.H = lam0 * (Proj @ I @ Proj) + 1000 * T @ T.T

        self.update_H(Vs, AVs)
        self.xlast = x0.copy()
        self.glast = g0.copy()

        lams_all, vecs_all = eigh(self.H)
        self.shift = self.shift_factor * lams_all[-(ntr + 1)]

        self.H = Proj @ (self.H) @ Proj + self.shift * (T @ T.T)
        #self.Hinv = Proj @ self.Hinv @ Proj + (T @ T.T) / self.shift

        _, nvecs = Vs.shape

        self.lam = lams[0]
        self.v = Vs[:, 0]

        gpar = np.dot(g0, self.v) * self.v
        gperp = g0 - gpar
        
        if bowlescape:
            if self.lam < 0:
                g = g0 - 2 * gpar
            else:
                g = -gpar
                g *= np.linalg.norm(g0) / np.linalg.norm(g)
        else:
            g = g0 - 2 * gpar

        self.calls += 1
        return f0, g

    def update_H(self, S, Y, method='TS-BFGS', symm=2):
        self.Hlast = self.H.copy()
        if len(S.shape) == 1:
            if np.linalg.norm(S) < 1e-8:
                return
            S = S.reshape((self.d, 1))
            Y = Y.reshape((self.d, 1))

#        if np.linalg.norm(S.T @ Y) < 1e-8:
#            return

        _, nvecs = S.shape

        B = self.H
        I = np.eye(self.d)

        # Symmetrize Y^T S
        if symm == 0:
            Ytilde = Y + S @ scipy.linalg.lstsq(S.T @ S, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
            #Ytilde = Y + S @ np.linalg.lstsq(S.T @ S, np.tril(S.T @ Y - Y.T @ S, -1).T, rcond=None)[0]
        elif symm == 1:
            Ytilde = Y + Y @ scipy.linalg.lstsq(S.T @ Y, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
            #Ytilde = Y + Y @ np.linalg.lstsq(S.T @ Y, np.tril(S.T @ Y - Y.T @ S, -1).T, rcond=None)[0]
        elif symm == 2:
            Ytilde = Y + symmetrize_Y2(S, Y)
        else:
            raise RuntimeError

        if method == 'BFGS':
            BS = B @ S
            X1, _, _, _ = np.linalg.lstsq(Ytilde.T @ S, Ytilde.T, rcond=None)
            X2, _, _, _ = np.linalg.lstsq(S.T @ BS, BS.T, rcond=None)
            self.H += Ytilde @ X1 - BS @ X2

        elif method == 'sequential Bofill':
            for i in range(nvecs - 1, -1, -1):
                s = S[:, i]
                y = Y[:, i]
                Bs = B @ s
                Bsy = Bs - y
                phi = (Bsy.T @ s)**2 / ((Bsy.T @ Bsy) * (s.T @ s))
                phi = np.sqrt(phi)
                dH_SR1 = -np.outer(Bsy, Bsy) / (Bsy.T @ s)
                dH_BFGS = np.outer(y, y) / (s.T @ y) - np.outer(Bs, Bs) / (s.T @ Bs)
                self.H += phi * dH_SR1 + (1 - phi) * dH_BFGS

        elif method == 'scalar Bofill':
            BS = B @ S
            BSY = B @ S - Ytilde
            phi = np.sqrt((BSY[:, 0].T @ S[:, 0])**2 / (BSY[:, 0].T @ BSY[:, 0] * S[:, 0].T @ S[:, 0]))

            # SR1 part
            X1, _, _, _ = np.linalg.lstsq(BSY.T @ S, BSY.T, rcond=None)
            dH_SR1 = -BSY @ X1

            # BFGS part
            X2, _, _, _ = np.linalg.lstsq(S.T @ Ytilde, Ytilde.T, rcond=None)
            X3, _, _, _ = np.linalg.lstsq(S.T @ BS, BS.T, rcond=None)
            dH_BFGS = Ytilde @ X2 - BS @ X3
            self.H += phi * dH_SR1 + (1 - phi) * dH_BFGS

        elif method == 'matrix Bofill':
            BS = B @ S
            BSY = BS - Ytilde
            sqrtBSYTBSY = np.real(sqrtm(BSY.T @ BSY))

            X, _, _, _ = np.linalg.lstsq(BSY.T @ BSY, BSY.T @ S, rcond=None)
            Phi = S.T @ BSY @ X
            sqrtPhi = np.real(sqrtm(Phi))

            # SR1 part
            X1, _, _, _ = np.linalg.lstsq(BSY.T @ S, sqrtPhi @ BSY.T, rcond=None)
            dH_SR1 = -BSY @ X1

            I_sqrtPhi = np.eye(nvecs) - sqrtPhi

            # BFGS part
            X2, _, _, _ = np.linalg.lstsq(S.T @ Ytilde, I_sqrtPhi @ Ytilde.T, rcond=None)
            X3, _, _, _ = np.linalg.lstsq(S.T @ BS, I_sqrtPhi @ BS.T, rcond=None)
            dH_BFGS = Ytilde @ X2 - BS @ X3
            self.H += dH_SR1 + dH_BFGS

        elif method == 'TS-BFGS':
            #lams_B, vecs_B = np.linalg.eigh(B)
            lams_B, vecs_B = eigh(B)
            #absB = vecs_B @ np.diag(np.abs(lams_B)) @ vecs_B.T
            #absB = vecs_B @ (np.abs(lams_B[:, np.newaxis]) * vecs_B.T)
            J = Ytilde - B @ S
            YTS = Ytilde.T @ S
            absBS = vecs_B @ (np.abs(lams_B[:, np.newaxis]) * (vecs_B.T @ S))
            STabsBS = S.T @ absBS
            UT, _, _, _ = scipy.linalg.lstsq(YTS @ YTS.T + STabsBS @ STabsBS, YTS.T @ Ytilde.T + STabsBS @ absBS.T)
            U = UT.T
            UJT = U @ J.T
            self.H += (UJT + UJT.T) - U @ (J.T @ S) @ U.T
        
        elif method == 'PSB':
            J = Ytilde - B @ S
            UT, _, _, _ = scipy.linalg.lstsq(S.T @ S, S.T)
            U = UT.T
            self.H += J @ U.T + U @ J.T - U @ J.T @ S @ J.T

        elif method == 'Powell':
            self.H += (Ytilde - B @ S) @ S.T

        elif method == 'SR1':
            BS = B @ S
            YBS = Ytilde - BS
            X1, _, _, _ = np.linalg.lstsq(YBS.T @ S, YBS.T, rcond=None)
            self.H += YBS @ X1

        # Probably don't use this.
        elif method == 'TS-DFP':
            absYlast = np.zeros_like(Ytilde)
            absY = Ytilde.copy()
            J = Ytilde - B @ S
            Bplus = B.copy()
            U = None
            while np.linalg.norm(absY - absYlast) > 1e-6:
                Bpluslast = Bplus.copy()
                absYlast = absY.copy()
                lams, vecs = np.linalg.eigh(Bplus)
                absY = vecs @ np.diag(np.sign(lams)) @ vecs.T @ Ytilde
                print(np.sum(lams < 0), np.linalg.norm(absY - absYlast))
                UT, _, _, _ = np.linalg.lstsq(absY.T @ S, absY.T, rcond=None)
                U = UT.T
                Bplus = (2 * Bpluslast + B + J @ U.T  + U @ J.T - U @ J.T @ S @ U.T) / 3.

            if U is not None:
                self.H += J @ U.T + U @ J.T - U @ J.T @ S @ U.T

        elif method == 'TS-SR1':
            J = Ytilde - B @ S
            UT, _, _, _ = scipy.linalg.lstsq(S.T @ S, S.T)
            U = UT.T
            Hplus = self.H + J @ U.T + U @ J.T - U @ J.T @ S @ U.T
            while True:
                Hplusold = Hplus.copy()
                X = (Hplus - self.H) @ J
                UT, _, _, _ = scipy.linalg.lstsq(S.T @ X, X.T)
                U = UT.T
                Hplus = self.H + U @ J.T + J @ U.T - U @ J.T @ S @ U.T
                Hdiffnorm = np.linalg.norm(Hplus - Hplusold)
                print('TEST:', Hdiffnorm)
                if Hdiffnorm < 1e-4:
                    break
            self.H = Hplus

        elif method == 'test':
            #lams_B, vecs_B = np.linalg.eigh(B)
            lams_B, vecs_B = eigh(B)
            absB = vecs_B @ np.diag(np.abs(lams_B)) @ vecs_B.T
            J = Ytilde - B @ S
            #UT, _, _, _ = scipy.linalg.lstsq(Ytilde.T @ S @ S.T @ Ytilde + S.T @ absB @ S @ S.T @ absB @ S, S.T @ Ytilde @ Ytilde.T + S.T @ absB @ S @ S.T @ absB)

            #denom = Ytilde.T @ S + S.T @ absB @ S
            #denomYT, _, _, _ = scipy.linalg.lstsq(denom, Ytilde.T)
            #denomBS, _, _, _ = scipy.linalg.lstsq(denom, S.T @ absB)
            #M = Ytilde @ denomYT + absB @ S @ denomBS
            ##print(M)
            #M = 0.5 * (M + M.T)

            #lams_M, vecs_M = scipy.linalg.eigh(M)
            #absM = vecs_M @ np.diag(lams_M) @ vecs_M.T
            absM = absB

            SBS = S.T @ absB @ S
            lams_SBS, vecs_SBS = scipy.linalg.eigh(SBS)
            sqrtSBS = vecs_SBS @ np.diag(np.sqrt(lams_SBS)) @ vecs_SBS.T

            #Bplus = B.copy()
            UT, _, _, _ = scipy.linalg.lstsq(SBS, S.T @ absB)
            U = UT.T
            Bplus = B + (U @ J.T + J @ U.T) - U @ (J.T @ S) @ U.T

            for i in range(500):
                lams_Bplus, vecs_Bplus = scipy.linalg.eigh(Bplus)
                absBplus = vecs_Bplus @ np.diag(np.abs(lams_Bplus)) @ vecs_Bplus.T
                SBplusS = S.T @ absBplus @ S
                lams_plus, vecs_plus = scipy.linalg.eigh(SBplusS)
                sqrtplus = vecs_plus @ np.diag(np.sqrt(lams_plus)) @ vecs_plus.T
                UT, _, _, _ = scipy.linalg.lstsq(SBS @ sqrtplus + SBplusS @ sqrtSBS, sqrtSBS @ S.T @ absBplus + sqrtplus @ S.T @ absB)
                U = UT.T
                Bpluslast = Bplus.copy()
                Bplus = (Bplus + B + (U @ J.T + J @ U.T) - U @ (J.T @ S) @ U.T) / 2.
                print('test:', np.linalg.norm(Bplus - Bpluslast))
                if np.linalg.norm(Bplus - Bpluslast) < 1e-10:
                    break
            self.H = Bplus.copy()
#
            #UT, _, _, _ = scipy.linalg.lstsq(S.T @ absM @ S, S.T @ absM)
#           # UT, _, _, _ = scipy.linalg.lstsq(Ytilde.T @ S + S.T @ absB @ S, S.T @ Ytilde @ Ytilde.T + S.T @ absB @ S @ S.T @ absB)
            #U = UT.T
            #self.H += (U @ J.T + J @ U.T) - U @ (J.T @ S) @ U.T


        # Symmetrize H
        #self.H += np.tril(self.H.T - self.H, -1)
        self.H -= np.tril(self.H.T - self.H, -1).T
        #self.H = 0.5 * (self.H + self.H.T)

        print('DEBUG:', np.linalg.norm(self.H @ S - Ytilde))

        #print(self.H @ S)
        #print(Ytilde)
        #print(self.H @ S - Ytilde)
        #raise RuntimeError


#        lams_H, vecs_H = np.linalg.eigh(self.H)
#        print(lams_H, np.linalg.norm(vecs_H[:, -1] @ S), np.linalg.norm(self.H @ S - Ytilde))

    def update_Hinv(self, S, Y, method='TS-BFGS', symm=1):
        self.Hlast = self.H.copy()
        if len(S.shape) == 1:
            S = S.reshape((self.d, 1))
            Y = Y.reshape((self.d, 1))

        if np.linalg.norm(S.T @ Y) < 1e-8:
            return

        _, nvecs = S.shape

        Binv = self.Hinv
        I = np.eye(self.d)

        # Symmetrize Y^T S
        if symm == 0:
            Ytilde = Y + S @ scipy.linalg.lstsq(S.T @ S, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
            #Ytilde = Y + S @ np.linalg.lstsq(S.T @ S, np.tril(S.T @ Y - Y.T @ S, -1).T, rcond=None)[0]
        elif symm == 1:
            Ytilde = Y + Y @ scipy.linalg.lstsq(S.T @ Y, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
            #Ytilde = Y + Y @ np.linalg.lstsq(S.T @ Y, np.tril(S.T @ Y - Y.T @ S, -1).T, rcond=None)[0]
        elif symm == 2:
            Ytilde = Y + symmetrize_Y2(S, Y)
        else:
            raise RuntimeError

        if method == 'TS-BFGS':
            #absS = S.copy()
            #for i in range(nvecs):
            #    absS[:, i] *= np.sign(S[:, i] @ Ytilde[:, i])

            lams, vecs = eigh(Binv)
            #absBinv = vecs @ np.diag(np.abs(lams)) @ vecs.T
            absBinvY = vecs @ (np.abs(lams)[:, np.newaxis] * (vecs.T @ Ytilde))

            #YTS = Ytilde.T @ S
            #lams_YTS, vecs_YTS = eigh(YTS)
            #absYTS = vecs_YTS @ np.diag(np.abs(lams_YTS)) @ vecs_YTS.T

            J = S - Binv @ Ytilde
            #UT, _, _, _ = scipy.linalg.lstsq(vecs_YTS @ np.diag(lams_YTS**2) @ vecs_YTS.T, absYTS @ S.T)
            #UT, _, _, _ = scipy.linalg.lstsq(Ytilde.T @ absS, absS.T)
            #UT, _, _, _ = scipy.linalg.lstsq(Ytilde.T @ absBinv @ Ytilde, Ytilde.T @ absBinv)
            #UT, _, _, _ = scipy.linalg.lstsq(Ytilde.T @ absBinv @ Ytilde, Ytilde.T @ absBinv)
            #YBY = Ytilde.T @ absBinv @ Ytilde
            YBY = Ytilde.T @ absBinvY
            UT, _, _, _ = scipy.linalg.lstsq(YBY @ YBY, YBY @ absBinvY.T)
            #UT, _, _, _ = scipy.linalg.lstsq(Ytilde.T @ S @ S.T @ Ytilde, S.T @ S @ Ytilde.T)
            #UT, _, _, _ = scipy.linalg.lstsq(Ytilde.T @ S, S.T)
            U = UT.T
            self.Hinv += (J @ U.T + U @ J.T) - U @ (J.T @ Ytilde) @ U.T
        else:
            raise ValueError(method)

        #print('TEST:', np.linalg.norm(self.Hinv @ Ytilde - S))
        #print(np.hstack((self.Hinv @ Ytilde, S)))
        # Symmetrize Hinv
        #self.Hinv += np.tril(self.Hinv.T - self.Hinv, -1)
        self.Hinv -= np.tril(self.Hinv.T - self.Hinv, -1).T
        #self.Hinv = 0.5 * (self.Hinv + self.Hinv.T)

        #self.H = scipy.linalg.pinv(self.Hinv)



    def p_rfo(self, x0, maxiter, ftol, **kwargs):
        maxres = kwargs.pop('maxres', None)
        f, _ = self.find_saddle_point(x0, maxres=maxres, **kwargs)
        x = x0.copy()
        g = self.glast.copy()

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
        A2 = np.zeros((self.d, self.d))
        B1 = np.eye(2)
        B2 = np.eye(self.d)

        dq = np.zeros(self.d)

        R = 0.1
        Rmax = 0.2

        xlast = x.copy()
        lastfailed = False

        while True:
            alpha = 1.
            
            H = self.H
            lams, vecs = np.linalg.eigh(H)

            if self.molecule:
                trvecs = atoms_tr_projection(x)
                H = self.H + 2 * lams[-1] * trvecs @ trvecs.T
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
                B2[1:, 1:] = np.eye(self.d - 1) * alpha
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
#            f, g = self.f(x + dx)
            f, g = self.f_update(x + dx)

            r = (f - flast) / df_pred
            if (r < re_l) or (r > re_u):
                print("Scaling back R:", r, R, dx_norm)
                R = dx_norm / Sfdown
            elif (ri_l <= r <= ri_u) and abs(dx_norm - R) <= eps1:
                R = min(R * np.sqrt(Sf), Rmax)

            if r < Lb or r > Ub:
#                if lastfailed:
#                    print("Search failed, backtracking:", r, R, dx_norm)
##                    self.find_saddle_point(x, maxres=maxres, nlan=50, **kwargs)
                f = flast
                g = glast.copy()
                lastfailed = True
                continue

            x += dx
            flast = f
            xlast = x.copy()
            glast = g.copy()

            self.find_saddle_point(x, maxres=maxres, **kwargs)

            print(f, np.linalg.norm(g), R, np.sign(lams[0]) * np.sqrt(np.abs(lams[0])) * 3276.4983)
            lastfailed = False

            if np.linalg.norm(self.glast) < ftol:
                return x

        
    def rfo(self, x0, maxiter, ftol, **kwargs):
        f, _ = self.find_saddle_point(x0, maxres=1e-3, **kwargs)
        x = x0.copy()
        lam = self.lam
        AH = np.zeros((self.d + 1, self.d + 1))
        dx_max = 0.1

        while True:
            AH[1:, 1:] = self.H
            AH[1:, 0] = AH[0, 1:] = self.glast
            lam_rfo, vecs_rfo = np.linalg.eigh(AH)
            dx = vecs_rfo[1:, 0] / vecs_rfo[0, 0]
            dx *= min(1.0, dx_max / np.linalg.norm(dx))
            x += dx
            f, _ = self.find_saddle_point(x, maxres=1e-3, **kwargs)
            gnorm = np.linalg.norm(self.glast)
            print(f, gnorm, np.sign(self.lam) * np.sqrt(np.abs(self.lam)) * 3276.4983)
            if gnorm < ftol:
                return x

    def berny(self, x0, maxiter, ftol, nqn=0, qntol=0.05,
              r_trust=0.2, inc_factr=0.8, dec_factr=0.8, dec_lb=0., dec_ub=5.,
              inc_lb=0.8, inc_ub=1.2, **kwargs):
        #maxres = None
        #if 'maxres' in kwargs:
        #    maxres = kwargs.pop('maxres')

        f1, _ = self.find_saddle_point(x0, **kwargs)
        #if maxres is not None:
        #    kwargs['maxres'] = maxres

        g1 = self.glast.copy()
        f, g = f1, g1
        x = x0.copy()
        xlast = x.copy()
        I = np.eye(self.d)

        #evnext = True
        evnext = False

        r_trust2 = r_trust**2

        gnormlast = np.linalg.norm(g)

        n = 1
        xi = 2.
        while True:
            lams, vecs = eigh(self.H)
#            lams = np.clip(lams, self.lam, 1000)  # XXX: This may be necessary for convergence!

            L = np.abs(lams)
            L[0] *= -1
            #dx = -vecs @ np.diag(1 / L) @ vecs.T @ g
            dx = -vecs @ ((vecs.T @ g) / L)

            dx_rms = np.sqrt(np.average(dx**2))

            dx_mag = np.linalg.norm(dx)
            dx_max = np.max(np.abs(dx))

            bound_clip = False
            if dx_mag > r_trust:
                xi /= 2.
                #bound_clip = True
                rfo_counter = 0
                xilower = 0
                xiupper = None
                Vg = vecs.T @ g
                LL = L * L
                LLLL = LL * LL
                VgLL = Vg * LL
                r_trust2 = r_trust**2
                while True:
                    if xiupper is None:
                        xi *= 2
                    else:
                        xi = (xilower + xiupper) / 2.
                    #dx_mag2 = Vg.T @ (VgLL / (LLLL + 2 * xi * LL + xi**2))
                    #dx = -vecs @ (L * Vg/ (LL + xi))
                    dx = -vecs @ (Vg * (L / (LL + xi)))
                    dx_mag = np.linalg.norm(dx)
                    dx_max = np.max(np.abs(dx))
                    #dx_mag = np.max(np.abs(dx))
                    #dx_mag2 = dx_mag**2
                    if abs(dx_mag - r_trust) < 1e-14 * r_trust:
                        bound_clip = True
                        print(xi, rfo_counter)
                        break

                    if dx_mag > r_trust:
                        xilower = xi
                    else:
                        xiupper = xi
                    rfo_counter += 1


            f0, g0 = f, g
            H0 = self.H.copy()

            df_pred = g.T @ dx + (dx.T @ self.H @ dx) /2.
            H0proj = dx.T @ H0 @ dx / dx_mag**2
    
            ##if lams[0] > 0 or dx_mag > 0.99 * dx_max or (nqn > 0 and n % nqn == 0):
            #if lams[0] > 0 or dx_rms > qntol or (nqn > 0 and n % nqn == 0):
            if lams[0] > 0 or evnext or (nqn > 0 and n % nqn == 0):
                f1, _ = self.find_saddle_point(x + dx, **kwargs)
                g1 = self.glast.copy()
                n = 1
            else:
                f1last, g1last = f1, g1
                f1, g1 = self.f_update(x + dx)
                n += 1
    
            if np.linalg.norm(g1) < ftol:
                return x + dx
    
            df = f1 - f0
            ratio = df_pred / df

            # Test 2 begins here
            gp0 = g0.T @ dx #/ dx_mag
            gp1 = g1.T @ dx #/ dx_mag
            H0dx = H0 @ dx #/ dx_mag
            H1dx = self.H @ dx #/ dx_mag
            df = f1 - f0
            deltag = gp1 - gp0
            rad = 2 * (-6 * df**2 + 6 * df * gp0 - gp0**2 + 6 * df * gp1 - 4 * gp0 * gp1 - gp1**2)
            radg = 2 * (-12 * df * deltag + 6 * (df * H0dx + deltag * gp0) - 2 * gp0 * H0dx + 6 * (df * H1dx + deltag * gp1) - 4 * (H0dx * gp1 + gp0 * H1dx) - 2 * gp1 * H1dx)

            method = None
            alpha = -1
            print('radicand:', rad)
#            if rad > 0:
#                method = 'quartic'
##                sign = -np.sign(gp0)
#                r = np.sqrt(rad)
#                alpha1 = (-6 * df + 4 * gp0 + 2 * gp1 + r) / (12 * df - 6 * (gp0 + gp1))
#                alpha2 = (6 * df - 4 * gp0 - 2 * gp1 + r) / (12 * df - 6 * (gp0 + gp1))
#                if abs(alpha1) < abs(alpha2):
#                    sign = 1
#                else:
#                    sign = -1
#                rg = 0.5 * radg / r
#
#                a = (gp1 - gp0 + sign * r) / 2.
#                b = -2 * df + 2 * gp0 - sign * r
#                c = (6 * df - 5 * gp0 - gp1 + sign * r) / 2.
#                d = gp0
#                e = f0
#                
#                ag = (H1dx - H0dx + sign * rg) / 2.
#                bg = -2 * deltag + 2 * H0dx - sign * rg
#                cg = (6 * deltag - 5 * H0dx - H1dx + sign * rg) / 2.
#                dg = H0dx
#                eg = gp0
#
#                coeffs = [4 * a, 3 * b, 2 * c, d]
#
#                print(coeffs)
#                roots = np.roots(coeffs)
#                
#                real_roots = []
#                for root in roots:
#                    if root.imag == 0 and 0 < root.real < 1.:
#                        alpha = root.real
#                        break

#            gp0 = g0.T @ dx
#            gp1 = g1.T @ dx
#            H0dx = H0 @ dx
#            H1dx = self.H @ dx
#            dxH0dx = dx.T @ H0dx
#            dxH1dx = dx.T @ H1dx
#
#            a = 3 * f0 - 3 * f1 + 2 * gp0 + gp1 + 0.5 * dxH0dx
#            b = -4 * f0 + 4 * f1 - 3 * gp0 - gp1 - dxH0dx
#            c = 0.5 * dxH0dx
#            d = gp0
#            e = f0
#
#            ag = 3 * g0 - 3 * g1 + 2 * H0dx + H1dx
#            bg = -4 * g0 + 4 * g1 - 3 * H0dx - H1dx
#            cg = 0.
#            dg = H0dx
#            eg = g0
#
#            coeffs = [4 * a, 3 * b, 2 * c, d]
#            roots = np.roots(coeffs)
#            
#            real_roots = []
#            for root in roots:
#                if root.imag == 0 and 0 < root < dx_max / dx_mag:
#                    real_roots.append(root.real)
#            print('quartic roots', real_roots)
#            if real_roots:
#                if len(real_roots) == 3:
#                    alpha = real_roots[1]
#                else:
#                    alpha = real_roots[-1]
#                method = 'quartic'


            if alpha < 0:
#                if gp0 * gp1 < 0:
                method = 'cubic'
                a = 2 * f0 - 2 * f1 + gp0 + gp1
                b = -3 * f0 + 3 * f1 - 2 * gp0 - gp1
                c = gp0
                d = f0

                ag = 2 * g0 - 2 * g1 + H0dx + H1dx
                bg = -3 * g0 + 3 * g1 - 2 * H0dx - H1dx
                cg = H0dx
                dg = g0

                roots = np.roots([3 * a, 2 * b, c])

                real_roots = []
                for root in roots:
                    if root.imag == 0 and 0 < root < r_trust / dx_mag:
                        real_roots.append(root.real)
                print('cubic fit roots', real_roots)
                if real_roots:
                    alpha = real_roots[0]

            if alpha < 0:
                method = 'quadratic'
                if g0.T @ g1 < 0:
                    alpha = gp0 / (gp0 - gp1)
                else:
                    alpha = 1

            alpha = min(alpha, r_trust / dx_mag)


            if method is None or alpha < 0:
                alpha = 1
                f = f1
                g = g1
            elif method == 'quartic':
                f = a * alpha**4 + b * alpha**3 + c * alpha**2 + d * alpha + e
                #g = g0 + alpha * (g1 - g0)
                g = ag * alpha**4 + bg * alpha**3 + cg * alpha**2 + dg * alpha + eg
            elif method == 'cubic':
                f = a * alpha**3 + b * alpha**2 + c * alpha + d
                #g = g0 + alpha * (g1 - g0)
                g = ag * alpha**3 + bg * alpha**2 + cg * alpha + dg
            elif method == 'quadratic':
                f = f0 + alpha * (f1 - f0)
                g = g0 + alpha * (g1 - g0)
            else:
                f = f0 + alpha * (f1 - f0)
                g = g0 + alpha * (g1 - g0)

            x += alpha * dx
            ratio = df_pred / df
            evnext = False

            if ratio < dec_lb or ratio > dec_ub:
                r_trust = dx_mag * dec_factr
                f, g = self.f_update(x)
            elif bound_clip and inc_lb < ratio < inc_ub:
                r_trust /= inc_factr
#            elif bound_clip and inc_lb < ratio < inc_ub:
#                r_trust /= inc_factr

            gnorm = np.linalg.norm(g)
            gnormlast = gnorm

            print(f1, np.linalg.norm(g1), ratio, alpha, dx_mag / r_trust, r_trust, method, lams[0])

    def gediis(self, x0, maxcalls, ftol, nqn=0, nhist=5,
               dx_max=0.2, inc_factr=0.8, dec_factr=0.8, dec_lb=0., dec_ub=5.,
               inc_lb=0.8, inc_ub=1.2, gnorm_ev_thr=2., nrandom=0, **kwargs):
        self.calls = 0
        f1, _ = self.find_saddle_point(x0, nrandom=nrandom, **kwargs)
        g1 = self.glast.copy()
        gnorm = np.linalg.norm(g1)
        gnormlast = 0.
        evnext = True

        G = np.zeros((self.d, nhist), dtype=np.float64)
        R = np.zeros((self.d, nhist), dtype=np.float64)
        E = np.zeros(nhist, dtype=np.float64)
        HG = np.zeros((self.d, nhist), dtype=np.float64)
        R[:, 0] = x0.copy()
        G[:, 0] = g1.copy()
        E[0] = f1
        HG[:, 0], _, _, _ = scipy.linalg.lstsq(self.H, g1)

        H = np.zeros((self.d, self.d, nhist), dtype=np.float64)
        H[:, :, 0] = self.H.copy()

        f, g = f1, g1
        I = np.eye(self.d)

        xi = 0.5
        c = np.array([1.])

        n = 1
        neval = 0
        lam_last = 1
        while True:
            print(c)
            lams, vecs = eigh(self.H)
            allow_increase = True

            L = np.abs(lams)

            if lams[0] > 0:
                print('positive eigenvalue found')
                #bound_clip = True
                evnext = True
                #allow_increase = False
                #c[:-1] = 0.
                #c[-1] = 1.
                nclimb = 2
                #L[:nclimb] *= -1
                L[:nclimb] = -1
                L[nclimb:] = 0
                dx = -vecs[:, :nclimb] @ np.diag(1 / L[:nclimb]) @ vecs[:, :nclimb].T @ G[:, :n] @ c

                #dx = vecs[:, 0] * np.sign(vecs[:, 0].T @ G[:, :n] @ c)
                #nclimb = self.d
                #dx = vecs[:, :nclimb] @ np.diag(1 / lams[:nclimb]) @ vecs[:, :nclimb].T @ G[:, :n] @ c
                #dx *= dx_max / np.linalg.norm(dx)
                #dx_mag = dx_max
                #dx_rms = np.sqrt(np.average(dx**2))
            else:
                L[0] *= -1
                dx = -vecs @ np.diag(1 / L) @ vecs.T @ G[:, :n] @ c

            dx_rms = np.sqrt(np.average(dx**2))

            dx_mag = np.linalg.norm(dx)

                
            bound_clip = False
            if dx_mag > dx_max:
                xi /= 2.
                bound_clip = True
                rfo_counter = 0
                xilower = 0
                xiupper = None
                #L = np.abs(lams)
                #L[0] *= -1
                while True:
                    if xiupper is None:
                        xi *= 2
                    else:
                        xi = (xilower + xiupper) / 2.
                    #Htilde = vecs @ np.diag(L + xi / L) @ vecs.T
                    #dx, _, _, _ = scipy.linalg.lstsq(Htilde, -G[:, :n] @ c)
                    dx = -vecs @ np.diag(L / (L * L + xi)) @ vecs.T @ G[:, :n] @ c
                    dx_mag = np.linalg.norm(dx)
                    if abs(dx_mag - dx_max) < 1e-8 * dx_max:
                        print(xi, rfo_counter)
                        break

                    if dx_mag > dx_max:
                        xilower = xi
                    else:
                        xiupper = xi
                    rfo_counter += 1
            
            dr = (R[:, :n] @ c - R[:, n - 1]) + dx
            df_pred = dr @ G[:, n - 1] + 0.5 * dr @ self.H @ dr
            #dR = (R[:, :n] @ c + dx)[:, np.newaxis] - R[:, :n]
            #df_pred = np.diag(dR.T @ (G[:, :n] + self.H @ dR / 2.)) @ c

            if evnext or (nqn > 0 and neval % nqn == 0):
                V0 = dx[:, np.newaxis].copy()
                f1, _ = self.find_saddle_point(R[:, :n] @ c + dx, V0=V0, nrandom=nrandom, **kwargs)
                g1 = self.glast.copy()
                neval = 0
                evnext = False
            else:
                f1last, g1last = f1, g1
                f1, g1 = self.f_update(R[:, :n] @ c + dx)

            if self.calls >= maxcalls:
                return R[:, :n] @ c + dx

            neval += 1

            if np.linalg.norm(g1) < ftol:
                return R[:, :n] @ c + dx

            #df = f1 - E[:n] @ c
            df = f1 - E[n - 1]
            ratio = df_pred / df

            gnormlast = gnorm
            gnorm = np.linalg.norm(g1)
            #if gnorm > gnormlast:
            #    allow_increase = False

            if ratio < dec_lb or ratio > dec_ub:
                print(f1, np.linalg.norm(g1), ratio, dx_mag / dx_max, dx_max, lams[0])
                dx_max = dx_mag * dec_factr
                evnext = True
                c[:-1] = 0.
                c[-1] = 1.
                continue

            #elif (bound_clip or abs(dx_mag - dx_max) < 1e-3 * dx_max) and inc_lb < ratio < inc_ub:
            elif allow_increase and (bound_clip or abs(dx_mag - dx_max) < 1e-3 * dx_max) and inc_lb < ratio < inc_ub:
                dx_max /= inc_factr

            x = R[:, :n] @ c + dx

            if n >= nhist:
                G[:, :-1] = G[:, 1:].copy()
                R[:, :-1] = R[:, 1:].copy()
                E[:-1] = E[1:].copy()
                #H[:, :, :-1] = H[:, :, 1:].copy()
                HG[:, :-1] = HG[:, 1:].copy()


            n = min(n + 1, nhist)

            lams, vecs = eigh(self.H)
            L = abs(lams)
            if lams[0] > 0:
                L *= -1
            else:
                L[0] *= -1
            HG[:, n - 1] = vecs @ np.diag(1 / L) @ vecs.T @ g1

            G[:, n - 1] = g1.copy()
            R[:, n - 1] = x.copy()
            E[n - 1] = f1
            #H[:, :, n - 1] = self.H.copy()
            #HG[:, n - 1], _, _, _ = scipy.linalg.lstsq(self.H, g1)
            #HG = vecs @ np.diag(1 / L) @ vecs.T @ G[:, :n]

            #HG, _, _, _ = scipy.linalg.lstsq(self.H, G[:, :n])
            #HG = np.zeros((self.d, n), dtype=np.float64)
            #for i in range(n):
            #    HG[:, i], _, _, _ = scipy.linalg.lstsq(H[:, :, i], G[:, i])
            #GTG = HG.T @ HG
            GTG = HG[:, :n].T @ HG[:, :n]
            #GTG = G[:, :n].T @ G[:, :n]
            ONES = np.ones(n)
            gnorm_max = np.infty
            c = np.zeros(n)
            c[-1] = 1.
            for mask in mask_gen(n):
                Y, _, _, _ = scipy.linalg.lstsq(GTG, ONES - mask)
                c = (Y / (Y @ (ONES - mask))) @ (np.eye(n) - np.diag(mask))
                #if np.all(c >= 0.): # and np.all(c[-1] > c[:-1]):
                if np.all(c >= 0.) and c[-1] > 1. / n:
                    break

            print(f1, np.linalg.norm(g1), ratio, dx_mag/dx_max, dx_max, lams[0])


    def quasi_newton(self, x0, maxiter, ftol, maxls=10, **kwargs):
        f, _ = self.find_saddle_point(x0, **kwargs)
        c2 = 0.9
        dx_max = 0.025

        x = x0.copy()
        g = self.glast.copy()
        lam = self.lam

        self.converged = False
        while True:
            if self.lam >= 0:
                dx = np.dot(g, self.v) * self.v
                dx *= np.linalg.norm(g) / ((self.lam + 1e-8) * np.linalg.norm(dx))
            else:
                lams, vecs = np.linalg.eigh(self.H)
                dx = np.zeros_like(x)
                for i, ilam in enumerate(lams):
                    factr = -1 if i == 0 else 1
                    dx -= factr * (np.dot(g, vecs[:, i]) / (abs(ilam) + 1e-8)) * vecs[:, i]

            # Scale step size
            dx_mag = np.linalg.norm(dx)
            dx *= min(1.0, dx_max / dx_mag)
            dx_mag = min(dx_mag, dx_max)

            gdx = np.dot(g, dx)

#            f1, g1 = self.f(x + dx)
            f1, g1 = self.f_update(x + dx)
            if np.linalg.norm(g1) < ftol:
                return x + dx
            g1dx = np.dot(g1, dx)

            taulow, tauhigh = 0., 1.
            tau = 1.
            
            # Only do a backtracking line search if we passed a zero
            if np.abs(dx_mag - dx_max) > 1e-5 and np.sign(gdx) != np.sign(g1dx):
                gdxlow = gdx
                gdxhigh = g1dx
                k = 0
                while np.sign(gdx) * g1dx < -c2 * np.abs(gdx):
                    if tauhigh - taulow < 1e-2:
                        tau = (tauhigh + taulow) / 2.
                        f1, g1 = self.f(x + tau * dx)
                        break

                    tau = (taulow * gdxhigh - tauhigh * gdxlow) / (gdxhigh - gdxlow)
                    if tau - taulow < 1e-2 * (tauhigh - taulow) or tauhigh - tau < 1e-2 * (tauhigh - taulow):
                        tau = (tauhigh + taulow) / 2.
                        f1, g1 = self.f(x + tau * dx)
                        break
                    f1, g1 = self.f(x + tau * dx)
                    if np.linalg.norm(g1) < ftol:
                        return x + tau * dx
                    k += 1
                    g1dx = np.dot(g1, dx)
                    print(f1, gdx, taulow, gdxlow, tauhigh, gdxhigh, tau, g1dx)

                    if np.sign(g1dx) == np.sign(gdxlow):
                        taulow = tau
                        gdxlow = g1dx
                    else:
                        tauhigh = tau
                        gdxhigh = g1dx


#            ls_failed = False
#
#            if np.abs(dx_mag - dx_max) > 1e-5:
#                gdxlow = gdx
#                gdxhigh = g1dx
#                k = 0
#                while np.abs(g1dx) > c2 * np.abs(gdx) and not ls_failed:
#                    tau = (taulow * gdxhigh - tauhigh * gdxlow) / (gdxhigh - gdxlow)
#                    # if we hit the max step size, but the extrapolation wants us to keep going,
#                    # abort with the max step size.
#                    if tau < 1e-8:
#                        tau = tauhigh * 1.5
#                    tau = min(tau, tauhigh * 1.5, dx_max / dx_mag)
#                    if np.abs(tauhigh * dx_mag - dx_max) < 1e-4 and tau - tauhigh > -1e-4:
#                        ls_failed = True
#                        tau = dx_max / dx_mag
#                        break
#                    f1, g1 = self.f(x + tau * dx)
#                    k += 1
#                    g1dx = np.dot(g1, dx)
#                    print(f1, gdx, taulow, gdxlow, tauhigh, gdxhigh, tau, g1dx, dx_max / dx_mag, lam0, lam1, self.lam)
#                    if (gdx < 0 and g1dx > 0) or (gdx > 0 and g1dx < 0):
#                        tauhigh = tau
#                        gdxhigh = g1dx
#                    elif (gdx < 0 and gdxhigh > 0) or (gdx > 0 and gdxhigh < 0):
#                        taulow = tau
#                        gdxlow = g1dx
#                    else:
#                        if tau < tauhigh:
#                            taulow = tau
#                            gdxlow = g1dx
#                        else:
#                            taulow = tauhigh
#                            gdxlow = gdxhigh
#                            tauhigh = tau
#                            gdxhigh = g1dx
#                    if k == maxls:
#                        break

            dx *= tau

            x += dx
            f, g = f1, g1
            self.find_saddle_point(x, **kwargs)
            if self.converged:
                return x

            print(f, np.linalg.norm(g), tau, np.sign(self.lam) * np.sqrt(np.abs(self.lam)) * 3276.4983)

    def lbfgs(self, x0, maxiter, ftol, inicurv=1., maxls=10, exact_diag=False, **kwargs):
        invcurv = 1./inicurv
        f, g = self.find_saddle_point(x0, **kwargs)
        c2 = 0.9
        dx_max = 0.1

        x = x0.copy()

        self.converged = False
        m = 5
        ys = np.zeros((m, self.d))
        ss = np.zeros((m, self.d))
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

            lam0 = self.lam

            f1, g1 = self.find_saddle_point(x + dx, **kwargs)
            g1dx = np.dot(g1, dx)
            gdxlow = gdx
            gdxhigh = g1dx
            if self.converged:
                return x + dx
            lam1 = self.lam
#            if lam0 < 0 and lam1 > 0:
#                f1, g1 = self.find_saddle_point(x + dx, maxres=0., **kwargs)
#                print("Approx lam: {}, exact lam: {}".format(lam1, self.lam))

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
                    f1, g1 = self.find_saddle_point(x + tau * dx, **kwargs)
                    k += 1
                    g1dx = np.dot(g1, dx)
                    print(f1, gdx, taulow, gdxlow, tauhigh, gdxhigh, tau, g1dx, dx_max / dx_mag, lam0, lam1, self.lam)
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
                    if (lam0 < 0 and self.lam > 0) or (lam0 > 0 and self.lam < 0):
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

            print(f, np.linalg.norm(g), tau, np.sign(self.lam) * np.sqrt(np.abs(self.lam)) * 3276.4983)

def mask_gen(n):
    mask = np.zeros(n, dtype=int)
    yield mask.copy()
    while True:
        mask[0] += 1
        for i in range(n - 1):
            mask[i + 1] += mask[i] // 2
            mask[i] %= 2
        if mask[-1] == 1:
            return
        yield mask.copy()
