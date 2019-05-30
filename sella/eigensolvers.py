#!/usr/bin/env python

import numpy as np

from scipy.linalg import eigh, lstsq, solve

from .cython_routines import ortho
from .hessian_update import symmetrize_Y


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


def davidson(A, maxres, P, vref=None):
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

        # a hack for the optbench.org eigensolver convergence test
        if vref is not None:
            print(np.abs(V[:, 0] @ vref))
            if np.abs(V[:, 0] @ vref) > 0.99:
                return lams, V, AV

        Ytilde = symmetrize_Y(V, AV, symm=method)
        R = Ytilde[:, :nneg] - V[:, :nneg] * lams[np.newaxis, :nneg]
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


def lanczos(A, maxres, P):
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
        lams, vecs = eigh(Atilde)
        nneg = max(2, np.sum(lams < 0) + 1)
        # Rotate our subspace V to be diagonal in A.
        # This is not strictly necessary but it makes our lives easier later
        AV = AV @ vecs
        V = V @ vecs

        Ytilde = symmetrize_Y(V, AV, symm=method)
        R = Ytilde[:, :nneg] - V[:, :nneg] * lams[np.newaxis, :nneg]
        Rnorm = np.linalg.norm(R, axis=0)
        print(Rnorm, lams[:nneg], Rnorm / lams[:nneg], seeking)

        # Loop over all Ritz values of interest
        for seeking, (rinorm, thetai) in enumerate(zip(Rnorm, lams)):
            # Take the first Ritz value that is not converged, and use it
            # to extend V
            if rinorm >= maxres * np.abs(thetai):
                break
        # If they all seem converged, then we are done
        else:
            return lams, V, AV

        t = ortho(AV[:, seeking], V)

        # If Lanczos also fails to find a new search direction,
        # just give up and return the current Ritz pairs
        if t.shape[1] == 0:
            return lams, V, AV

        V = np.hstack([V, t])
        AV = np.hstack([AV, A.dot(t)])
    else:
        return lams, V, AV
