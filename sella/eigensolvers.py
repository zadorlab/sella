import numpy as np

from scipy.linalg import eigh, solve

from sella.utilities.math import modified_gram_schmidt
from .hessian_update import symmetrize_Y


def exact(A, gamma=None, P=None):
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


def rayleigh_ritz(A, gamma, P, B=None, v0=None, vref=None, vreftol=0.99,
                  method='jd0', maxiter=None):
    n, _ = A.shape

    if B is None:
        B = np.eye(n)

    if maxiter is None:
        maxiter = 2 * n + 1

    if gamma <= 0:
        return exact(A, gamma, P)

    if v0 is not None:
        V = modified_gram_schmidt(v0.reshape((-1, 1)))
    else:
        P_lams, P_vecs, _ = exact(P, 0)
        nneg = max(1, np.sum(P_lams < 0))
        V = modified_gram_schmidt(P_vecs[:, :nneg])
        v0 = V[:, 0]

    AV = A.dot(V)

    symm = 2
    seeking = 0
    while True:
        Atilde = V.T @ (symmetrize_Y(V, AV, symm=symm))
        lams, vecs = eigh(Atilde, V.T @ B @ V)
        nneg = max(1, np.sum(lams < 0))
        # Rotate our subspace V to be diagonal in A.
        # This is not strictly necessary but it makes our lives easier later
        AV = AV @ vecs
        V = V @ vecs
        vecs = np.eye(V.shape[1])
        if V.shape[1] >= maxiter:
            return lams, V, AV

        Ytilde = symmetrize_Y(V, AV, symm=symm)
        R = (Ytilde @ vecs[:, :nneg]
             - B @ V @ vecs[:, :nneg] * lams[np.newaxis, :nneg])
        Rnorm = np.linalg.norm(R, axis=0)

        # a hack for the optbench.org eigensolver convergence test
        if vref is not None:
            x0 = V @ vecs[:, 0]
            print(np.abs(x0 @ vref))
            if np.abs(x0 @ vref) > vreftol:
                print("Dot product between your v0 and the final answer:",
                      np.abs(v0 @ x0) / np.linalg.norm(v0))
                return lams, V, AV

        # Loop over all Ritz values of interest
        for seeking, (rinorm, thetai) in enumerate(zip(Rnorm, lams)):
            # Take the first Ritz value that is not converged, and use it
            # to extend V
            if V.shape[1] == 1 or rinorm >= gamma * np.abs(thetai):
                ri = R[:, seeking]
                thetai = lams[seeking]
                break
        # If they all seem converged, then we are done
        else:
            return lams, V, AV

        t = expand(V, Ytilde, P, B, lams, vecs, thetai, method, seeking)
        t /= np.linalg.norm(t)
        if np.linalg.norm(t - V @ V.T @ t) < 1e-2:  # pragma: no cover
            # Do Lanczos instead
            t = ri / np.linalg.norm(ri)

        t = modified_gram_schmidt(t[:, np.newaxis], V)

        # Davidson failed to find a new search direction
        if t.shape[1] == 0:  # pragma: no cover
            # Do Lanczos instead
            for rj in R.T:
                t = modified_gram_schmidt(rj[:, np.newaxis], V)
                if t.shape[1] == 1:
                    break
            else:
                t = modified_gram_schmidt(np.random.normal(size=(n, 1)), V)
                if t.shape[1] == 0:
                    return lams, V, AV

        V = np.hstack([V, t])
        AV = np.hstack([AV, A.dot(t)])


def expand(V, Y, P, B, lams, vecs, shift, method='jd0', seeking=0):
    d, n = V.shape
    R = Y @ vecs - B @ V @ vecs * lams[np.newaxis, :]
    Pshift = P - shift * B
    if method == 'lanczos':
        return R[:, seeking]
    elif method == 'gd':
        return np.linalg.solve(Pshift, R[:, seeking])
    elif method == 'jd0_alt':
        vi = V @ vecs[:, seeking]
        Pprojr = solve(Pshift, R[:, seeking])
        Pprojv = solve(Pshift, vi)
        alpha = vi.T @ Pprojr / (vi.T @ Pprojv)
        return Pprojv * alpha - Pprojr
    elif method == 'jd0':
        vi = V @ vecs[:, seeking]
        Aaug = np.block([[Pshift, vi[:, np.newaxis]], [vi, 0]])
        raug = np.zeros(d + 1)
        raug[:d] = R[:, seeking]
        z = solve(Aaug, -raug)
        return z[:d]
    elif method == 'mjd0_alt':
        Pprojr = solve(Pshift, R[:, seeking])
        PprojV = solve(Pshift, V @ vecs)
        alpha = solve((V @ vecs).T @ PprojV, (V @ vecs).T @ Pprojr)
        return solve(Pshift, ((V @ vecs) @ alpha - R[:, seeking]))
    elif method == 'mjd0':
        Vrot = V @ vecs
        Aaug = np.block([[Pshift, Vrot], [Vrot.T, np.zeros((n, n))]])
        raug = np.zeros(d + n)
        raug[:d] = R[:, seeking]
        z = solve(Aaug, -raug)
        return z[:d]
    else:  # pragma: no cover
        raise ValueError("Unknown diagonalization method {}".format(method))
