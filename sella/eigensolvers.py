#!/usr/bin/env python

import numpy as np

from scipy.linalg import eigh, lstsq, solve, null_space

from .cython_routines import ortho
from .hessian_update import symmetrize_Y, update_H


def exact(A, maxres=None, P=None):
    if isinstance(A, np.ndarray):
        lams, vecs = eigh(A)
    else:
        n, _ = A.shape
        if P is None:
            P = np.eye(n)
            vecs_P = np.eye(n)
        else:
            _, vecs_P, _ = exact(P)

        # Construct numerical version of A in case it is a LinearOperator.
        # This should be more or less exact if A is a numpy array already.
        B = np.zeros((n, n))
        for i in range(n):
            v = vecs_P[i]
            B += np.outer(v, A.dot(v))
        B = 0.5 * (B + B.T)
        lams, vecs = eigh(B)
    return lams, vecs, lams[np.newaxis, :] * vecs


def lobpcg(A, v0, maxres, P=None):

    if maxres <= 0:
        return exact(A, maxres, P)

    # Use identity matrix as preconditioner if none provided
    if P is None:
        N = I.copy()
    else:
        N = P

    N_thetas, N_vecs, _ = exact(N, 0)
    X0 = N_vecs[:, :3]
    N_theta = N_thetas[0]

#    X0 = np.array([v0]).T

    n, nev = X0.shape
    I = np.eye(n)

    # Relative convergence tolerance
    if maxres is None:
        maxres = np.sqrt(1e-15) * n

    # Orthogonalize initial guess vectors
    X = ortho(X0, np.empty((n, 0)))
    U = X.copy()

    # Initial Ritz pairs
    AX = A @ X
    AU = AX.copy()
    Atilde = X.conj().T @ AX
    thetas, Y = eigh(Atilde)

    # Update X and calculate residuals R
    X = X @ Y
    AX = AX @ Y
    Atilde = X.conj().T @ AX
    thetas, Y = eigh(Atilde)
    X = X @ Y[:, :nev]
    AX = AX @ Y[:, :nev]
    R = AX - X @ np.diag(thetas)
    RI = R.copy()

    # P begins empty
    P = np.empty((n, 0))
    AP = np.empty((n, 0))
    for k in range(n):
        # Find new search directions and orthogonalize
        Htilde, _, _, _ = lstsq(N - 1.15 * N_theta * I, RI)
#        Htilde, _ = bicg(N - 1.15 * N_theta * I, RI)
        H = ortho(Htilde, np.hstack((X, P)))
        U = np.hstack((U, H))

        # New set of guess vectors
        S = np.hstack((X, H, P)).copy()

        # Calculate action of A on search directions
        AH = A @ H
        AU = np.hstack((AU, AH))
        AS = np.hstack((AX, AH, AP))

        # Updated Ritz pairs
        Atilde = S.conj().T @ AS
        thetas, Y = eigh(Atilde)

        # Leftmost Ritz vectors becomes new X
        X = S @ Y[:, :nev]
        AX = AS @ Y[:, :nev]

        # Update residuals
        R = AX - X @ np.diag(thetas[:nev])

        # Check which if any vectors are converged
        converged = np.linalg.norm(R, axis=0) < maxres * np.abs(thetas[:nev])
        print(np.linalg.norm(R[:, 0]), thetas[0], np.linalg.norm(R[:, 0]) / thetas[0])

        if all(converged):
            print("LOBPCG converged in {} iterations".format(k))
            return thetas, S @ Y, AS @ Y

        # Indices of unconverged vectors
        iconv = [i for i in range(nev) if not converged[i]]

        RI = R[:, iconv].copy()
        Ytilde = Y[:, iconv].copy()

        # Zero the components belonging to X
        Ytilde[:nev, :] = 0.

        # Strict reorthogonalization with repeated MGS
        YI = ortho(Ytilde, Y[:, :nev])
        P = S @ YI
        AP = AS @ YI
    print('Warning: LOBPCG may not have converged')
    return thetas, S @ Y, AS @ Y


def davidson(A, maxres, P, vref=None, refine=False, harmonic=False):
    n, _ = A.shape

    if maxres <= 0:
        return exact(A, maxres, P)

    I = np.eye(n)

    P_lams, P_vecs, _ = exact(P, 0)
    nneg = max(2, np.sum(P_lams < 0) + 1)

    V = ortho(P_vecs[:, :nneg])

    AV = A.dot(V)

    method = 2
    seeking = 0
    while True:
        Atilde = V.T @ (symmetrize_Y(V, AV, symm=method))
        lams, vecs = eigh(Atilde, V.T @ V)
        nneg = max(2, np.sum(lams < 0) + 1)
        # Rotate our subspace V to be diagonal in A.
        # This is not strictly necessary but it makes our lives easier later
        AV = AV @ vecs
        V = V @ vecs
        vecs = np.eye(V.shape[1])

        # a hack for the optbench.org eigensolver convergence test
        if vref is not None:
            v0 = V @ vecs[:, 0]
            print(np.abs(v0 @ vref))
            if np.abs(v0 @ vref) > 0.99:
                return lams, V, AV

        Ytilde = symmetrize_Y(V, AV, symm=method)
        R = Ytilde @ vecs[:, :nneg] - V @ vecs[:, :nneg] * lams[np.newaxis, :nneg]
        Rnorm = np.linalg.norm(R, axis=0)
        print(Rnorm, lams[:nneg], Rnorm / lams[:nneg], seeking)

        # Loop over all Ritz values of interest
        for seeking, (rinorm, thetai) in enumerate(zip(Rnorm, lams)):
            # Take the first Ritz value that is not converged, and use it
            # to extend V
            if rinorm >= maxres * np.abs(thetai):
                ri = R[:, seeking]
                ui = V @ vecs[:, seeking]
                thetai = lams[seeking]
                break
        # If they all seem converged, then we are done
        else:
            return lams, V, AV

        if refine:
            Rtilde = AV - thetai * V
            _, C = eigh(Rtilde.T @ Rtilde)
            ui = V @ C[:, 0]
            yi = AV @ C[:, 0]
            thetai = ui @ yi
            ri = yi - thetai * ui
        if harmonic:
            Yshift = Ytilde - thetai * V
            Clams, C = eigh(np.diag(lams - thetai), Yshift.T @ Yshift)
            for i, cilam in enumerate(Clams):
                if cilam > 0:
                    break
            ui = V @ C[:, i - 1]
            yi = AV @ C[:, i - 1]
            thetai = ui @ yi
            ri = yi - thetai * ui

        Pproj = P - thetai * I
        Pprojr = solve(Pproj, ri)
        PprojV = solve(Pproj, V)
        alpha = solve(V.T @ PprojV, V.T @ Pprojr)
        ti = solve(Pproj, (V @ alpha - ri))

        t = ortho(ti, V)

        # Davidson failed to find a new search direction
        if t.shape[1] == 0:
            # Do Lanczos instead
            t = ortho(AV[:, -1], V)
            # If Lanczos also fails to find a new search direction,
            # just give up and return the current Ritz pairs
            if t.shape[1] == 0:
                return lams, V, AV

        V = np.hstack([V, t])
        AV = np.hstack([AV, A.dot(t)])
    else:
        return lams, V, AV

def eig_test(A, maxres, P, vref=None):
    n, _ = A.shape

    if maxres <= 0:
        return exact(A, maxres, P)

    I = np.eye(n)

    P_lams, P_vecs, _ = exact(P, 0)
    nneg = max(2, np.sum(P_lams < 0) + 1)

    V = ortho(P_vecs[:, :nneg])

    AV = A.dot(V)

    method = 2
    seeking = 0
    while True:
        #Atilde = V.T @ (symmetrize_Y(V, AV, symm=method))
        Atilde = V.T @ AV
        lams, vecs = eigh(Atilde, V.T @ V)
        nneg = max(2, np.sum(lams < 0) + 1)
        # Rotate our subspace V to be diagonal in A.
        # This is not strictly necessary but it makes our lives easier later
        #AV = AV @ vecs
        #V = V @ vecs
        #vecs = np.eye(V.shape[1])


        #Ytilde = symmetrize_Y(V, AV, symm=method)
        Ytilde = AV
        R = Ytilde @ vecs[:, :nneg] - V @ vecs[:, :nneg] * lams[np.newaxis, :nneg]
        Rnorm = np.linalg.norm(R, axis=0)
        print(Rnorm, lams[:nneg], Rnorm / lams[:nneg], seeking)

        # a hack for the optbench.org eigensolver convergence test
        if vref is not None:
            Rtilde = Ytilde - lams[0] * V
            _, C = eigh(Rtilde.T @ Rtilde)
            v0 = V @ C[:, 0]
            #v0 = V @ vecs[:, 0]
            print(np.abs(v0 @ vref))
            if np.abs(v0 @ vref) > 0.99:
                return lams, V, AV

        # Loop over all Ritz values of interest
        for seeking, (rinorm, thetai) in enumerate(zip(Rnorm, lams)):
            # Take the first Ritz value that is not converged, and use it
            # to extend V
            if rinorm >= maxres * np.abs(thetai):
                ri = R[:, seeking]
                ui = V @ vecs[:, seeking]
                thetai = lams[seeking]
                break
        # If they all seem converged, then we are done
        else:
            return lams, V, AV

        t = ortho(expand(V, Ytilde, P, lams, vecs, thetai, 'lanczos', seeking), V)

        # Davidson failed to find a new search direction
        if t.shape[1] == 0:
            # Do Lanczos instead
            t = ortho(AV[:, -1], V)
            # If Lanczos also fails to find a new search direction,
            # just give up and return the current Ritz pairs
            if t.shape[1] == 0:
                return lams, V, AV

        V = np.hstack([V, t])
        AV = np.hstack([AV, A.dot(t)])
    else:
        return lams, V, AV

def expand(V, Y, P, lams, vecs, shift, method='modified_jacobi_davidson', seeking=0):
    R = Y @ vecs - V @ vecs * lams[np.newaxis, :]
    Pshift = P - shift * np.eye(V.shape[0])
    if P is None:
        P = I.copy()
    if method == 'lanczos':
        return Y @ vecs[:, seeking]
    elif method == 'preconditioned_lanczos':
        return np.linalg.solve(Pshift, Y @ vecs[:, seeking])
    elif method == 'cg':
        return np.linalg.solve(P, R[:, seeking])
    elif method == 'davidson':
        return np.linalg.solve(Pshift, R[:, seeking])
    elif method == 'jacobi_davidson':
        vi = V @ vecs[:, seeking]
        Pprojr = solve(Pshift, R[:, seeking])
        Pprojv = solve(Pshift, vi)
        alpha = vi.T @ Pprojr / (vi.T @ Pprojv)
        return Pprojv * alpha - Pprojr
    elif method == 'modified_jacobi_davidson':
        Pprojr = solve(Pshift, R[:, seeking])
        PprojV = solve(Pshift, V @ vecs)
        alpha = solve((V @ vecs).T @ PprojV, (V @ vecs).T @ Pprojr)
        return solve(Pshift, ((V @ vecs) @ alpha - R[:, seeking]))


def lanczos(A, maxres, P, vref=None):
    n, _ = A.shape

    if maxres <= 0:
        return exact(A, maxres, P)

    I = np.eye(n)

    P_lams, P_vecs, _ = exact(P, 0)
    nneg = max(2, np.sum(P_lams < 0) + 1)

    V = ortho(P_vecs[:, :nneg])

    AV = A.dot(V)

    Ytilde = symmetrize_Y(V, AV, symm=2)
    Atilde = V.T @ Ytilde
    lams, vecs = eigh(Atilde, V.T @ V)
    nneg = max(2, np.sum(lams < 0) + 1)
    R = Ytilde @ vecs[:, :nneg] - V @ vecs[:, :nneg] * lams[np.newaxis, :nneg]

    rlast = R[:, 0].copy()
    dlast = -rlast

    method = 2
    seeking = 0
    while True:
        Ytilde = symmetrize_Y(V, AV, symm=method)
        Atilde = V.T @ Ytilde
        lams, vecs = eigh(Atilde, V.T @ V)
        nneg = max(2, np.sum(lams < 0) + 1)
        # hold onto this for later
        # Rotate our subspace V to be diagonal in A.
        # This is not strictly necessary but it makes our lives easier later
        #AV = AV @ vecs
        #V = V @ vecs

        # a hack for the optbench.org eigensolver convergence test
        if vref is not None:
            print(np.abs(V @ vecs[:, 0] @ vref))
            if np.abs(V @ vecs[:, 0] @ vref) > 0.99:
                return lams, V, AV

        #Ytilde = symmetrize_Y(V, AV, symm=method)
        R = Ytilde @ vecs[:, :nneg] - V @ vecs[:, :nneg] * lams[np.newaxis, :nneg]
        Rnorm = np.linalg.norm(R, axis=0)
        print(Rnorm, lams[:nneg], Rnorm / lams[:nneg], seeking)

        # Loop over all Ritz values of interest
        for seeking, (rinorm, thetai) in enumerate(zip(Rnorm, lams)):
            # Take the first Ritz value that is not converged, and use it
            # to extend V
            if rinorm >= maxres * np.abs(thetai):
                ri = R[:, seeking]
                ui = V[:, seeking]
                break
        # If they all seem converged, then we are done
        else:
            return lams, V, AV
        if vref is not None:
            seeking = 0
        t = expand(V, AV, P=P, method='modified_jacobi_davidson', seeking=seeking)

        t = ortho(t, V)

        V = np.hstack([V, t])
        AV = np.hstack([AV, A.dot(t)])
    else:
        return lams, V, AV



def spam(A, maxres, P, vref=None):
    n, _ = A.shape

    if maxres <= 0:
        return exact(A, maxres, P)

    I = np.eye(n)

    P_lams, V, _ = exact(P, 0)
    W = P @ V
    Akhat = np.zeros((n, n))

    nneg = max(2, np.sum(P_lams < 0) + 1)

    method = 2
    seeking = 0
    for i in range(n + 1):
        if i > 0:
            Ytilde = symmetrize_Y(V[:, :i], W[:, :i], symm=method)
            M = V[:, :i].T @ Ytilde
            thetas, X = eigh(M)
            # a hack for the optbench.org eigensolver convergence test
            if vref is not None:
                v0 = V[:, :i] @ X[:, 0]
                print(np.abs(v0 @ vref))
                if np.abs(v0 @ vref) > 0.99:
                    return thetas, V[:, :i], W[:, :i]

            nneg = max(2, np.sum(thetas < 0) + 1)
            R = Ytilde @ X[:, :nneg] - V[:, :i] @ X[:, :nneg] * thetas[np.newaxis, :nneg]
            Rnorm = np.linalg.norm(R, axis=0)
            print(Rnorm, thetas[:nneg], Rnorm / thetas[:nneg], seeking)

            for seeking, (rinorm, theta) in enumerate(zip(Rnorm, thetas)):
                if rinorm > maxres * np.abs(theta):
                    break
            else:
                return thetas, V[:, :i], W[:, :i]

            V[:, i:] = null_space(V[:, :i].T)
            W[:, i:] = P @ V[:, i:]
            Ak = P + update_H(P, V[:, :i], W[:, :i], method='test')
        else:
            Ytilde = W[:, :i]
            Ak = P

        #Akhat[:, :i] = V.T @ Ytilde
        #Akhat[:i, i:] = Akhat[i:, :i].T
        #Akhat[i:, i:] = W[:, i:].T @ V[:, i:]
        #Ak = V @ Akhat @ V.T
        lams, vecs = np.linalg.eigh(Ak)
        V[:, i] = ortho(vecs[:, seeking], V[:, :i]).ravel()
        W[:, i] = A.dot(V[:, i])
    else:
        return lams, V, AV

def lobpcg2(A, maxres, T, vref=None):
    n, _ = A.shape

    if maxres <= 0:
        return exact(A, maxres, P)

    I = np.eye(n)

    T_lams, T_vecs, _ = exact(T, 0)
    nneg = max(2, np.sum(T_lams < 0) + 1)

    x = ortho(T_vecs[:, :nneg])
    X = A.dot(x)

    S = x.copy()
    AS = X.copy()

    p = np.empty((n, 0))
    P = np.empty((n, 0))

    theta = np.diag(x.T @ X)
    w = X - theta[np.newaxis, :] * x
    w /= np.linalg.norm(w, axis=0)[np.newaxis, :]

    method = 2
    seeking = 0
    locked = np.empty((n, 0))
    while True:
        #W = A.dot(w)

        Snew = ortho(w, S)
        print(w.shape, Snew.shape)
        #if Snew.shape[1] == 0:
        #    return lams, V, AV
        ASnew = A.dot(Snew)
        S = np.hstack((S, Snew))
        AS = np.hstack((AS, ASnew))
        W = AS @ S.T @ w

        V = np.hstack((x, w, p))
        AV = np.hstack((X, W, P))
        AVtilde = symmetrize_Y(V, AV, symm=method)

        SA = V.T @ AVtilde
        SB = V.T @ V
        lams, vecs = eigh(SA, SB)
        n = nneg
        nneg = max(2, np.sum(lams < 0) + 1)

        x = V @ vecs[:, :nneg]
        xnorm = np.linalg.norm(x, axis=0)
        x /= xnorm[np.newaxis, :]

        X = AV @ vecs[:, :nneg]
        X /= xnorm[np.newaxis, :]

        R = X - x * lams[np.newaxis, :nneg]
        Rnorm = np.linalg.norm(R, axis=0)

        print(Rnorm, lams[:nneg], Rnorm / lams[:nneg], seeking)

        if np.all(Rnorm < maxres * np.abs(lams[:nneg])):
            return lams, V, AV

        p = V[:, :n] @ vecs[:n, :nneg] + V[:, 2*n:] @ vecs[2*n:, :nneg]
        pnorm = np.linalg.norm(p, axis=0)
        p /= pnorm[np.newaxis, :]


        P = AV[:, :n] @ vecs[:n, :nneg] + AV[:, 2*n:] @ vecs[2*n:, :nneg]
        P /= pnorm[np.newaxis, :]

        w = np.zeros_like(R)
        for i, (thetai, ri) in enumerate(zip(lams[:nneg], R.T)):
            w[:, i] = np.linalg.solve(T - thetai * I, ri)

        w /= np.linalg.norm(w, axis=0)[np.newaxis, :]
    else:
        return lams, V, AV
