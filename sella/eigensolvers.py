import numpy as np

from scipy.linalg import eigh, solve, solve_triangular
from scipy.sparse.linalg import LinearOperator

from sella.utilities.math import modified_gram_schmidt
from .hessian_update import symmetrize_Y


def exact(A, gamma=None, P=None, B=None):
    """Exact dense symmetric or generalized-symmetric eigendecomposition.

    For an ndarray ``A`` this is a direct ``eigh``. For a LinearOperator, ``A``
    is first materialized as a symmetric dense matrix in the eigenbasis of the
    preconditioner ``P`` (identity when ``P`` is None). ``B`` optionally
    supplies the positive-definite generalized metric. Returns
    ``(eigenvalues, eigenvectors, A @ eigenvectors)``. ``gamma`` is accepted
    only for a common signature with ``rayleigh_ritz`` and is unused here.
    """
    if isinstance(A, np.ndarray):
        Adense = A
    else:
        n, _ = A.shape
        if P is None:
            P = np.eye(n)
            vecs_P = np.eye(n)
        else:
            _, vecs_P, _ = exact(P)

        # Construct numerical version of A in case it is a LinearOperator.
        # This should be more or less exact if A is a numpy array already.
        Adense = np.zeros((n, n))
        for i in range(n):
            v = vecs_P[i]
            Adense += np.outer(v, A.dot(v))
        Adense = 0.5 * (Adense + Adense.T)
    lams, vecs = eigh(Adense, B)
    return lams, vecs, Adense @ vecs


def _orthogonal_dense_probe(V):
    """Return a deterministic dense unit vector outside ``range(V)``."""
    Q, _ = np.linalg.qr(V, mode='reduced')
    grid = np.arange(1, V.shape[0] + 1, dtype=float)
    probes = (
        np.sin(np.sqrt(2.0) * grid) + np.cos(np.sqrt(3.0) * grid),
        np.sin(np.sqrt(5.0) * grid) + np.cos(np.sqrt(7.0) * grid),
    )
    for probe in probes:
        probe = probe - Q @ (Q.T @ probe)
        norm = np.linalg.norm(probe)
        if norm > 1e-12:
            return probe / norm
    return None


def _mark_secant_rank_cleanup(operator, seen=None):
    """Mark numerical Hessians reached through lightweight wrappers."""
    if seen is None:
        seen = set()
    identifier = id(operator)
    if identifier in seen:
        return
    seen.add(identifier)

    if hasattr(operator, 'requires_secant_rank_cleanup'):
        operator.requires_secant_rank_cleanup = True
    source = getattr(operator, '_sella_source_operator', None)
    if source is not None:
        _mark_secant_rank_cleanup(source, seen)
    for matrix in getattr(operator, 'matrices', ()):
        _mark_secant_rank_cleanup(matrix, seen)


def _generalized_to_standard(operator, chol):
    """Return ``L^-1 operator L^-T`` for ``B = L L^T``."""
    if isinstance(operator, np.ndarray):
        left = solve_triangular(chol, operator, lower=True)
        return solve_triangular(chol, left.T, lower=True).T

    n = operator.shape[0]

    def matvec(vector):
        x = solve_triangular(chol.T, np.asarray(vector).ravel(), lower=False)
        Ax = operator.dot(x)
        return solve_triangular(chol, Ax, lower=True)

    transformed = LinearOperator(
        (n, n), matvec=matvec, dtype=np.result_type(operator.dtype, chol.dtype)
    )
    transformed._sella_source_operator = operator
    return transformed


def rayleigh_ritz(A, gamma, P, B=None, v0=None, vref=None, vreftol=0.99,
                  method='jd0', maxiter=None):
    """Iterative (generalized) eigensolver for the lowest eigenpairs of ``A``.

    A Davidson / Jacobi-Davidson scheme: build an orthonormal subspace ``V``
    (seeded from ``v0``, else from the negative-eigenvalue directions of the
    preconditioner ``P``), solve the projected generalized problem against ``B``
    (identity when None), and expand ``V`` toward the least-converged Ritz
    vector (via ``expand`` with the given ``method``) until a residual falls
    below ``gamma * |theta|`` or the subspace reaches ``min(n, maxiter)``. Falls
    back to ``exact`` when ``gamma <= 0``. Returns ``(lams, V, AV)``.
    ``vref``/``vreftol`` are an early-exit hook against a reference eigenvector.
    """
    n, _ = A.shape

    if maxiter is None:
        maxiter = 2 * n + 1

    if gamma <= 0:
        return exact(A, gamma, P, B=B)

    if B is not None:
        chol = np.linalg.cholesky(B)
        A_standard = _generalized_to_standard(A, chol)
        P_standard = _generalized_to_standard(P, chol)
        v0_standard = None if v0 is None else chol.T @ v0
        vref_standard = (
            None if vref is None
            else solve_triangular(chol, vref, lower=True)
        )
        lams, vecs_standard, Avecs_standard = rayleigh_ritz(
            A_standard, gamma, P_standard, v0=v0_standard,
            vref=vref_standard, vreftol=vreftol, method=method,
            maxiter=maxiter,
        )
        vecs = solve_triangular(
            chol.T, vecs_standard, lower=False
        )
        return lams, vecs, chol @ Avecs_standard

    B = np.eye(n)

    if v0 is not None and np.linalg.norm(v0) >= 1e-12:
        V = modified_gram_schmidt(v0.reshape((-1, 1)))
    else:
        P_lams, P_vecs, _ = exact(P, 0)
        nneg = max(1, np.sum(P_lams < 0))
        V = modified_gram_schmidt(P_vecs[:, :nneg])

    AV = A.dot(V)

    symm = 2
    escape_used = False
    while True:
        Atilde = V.T @ (symmetrize_Y(V, AV, symm=symm))
        lams, vecs = eigh(Atilde, V.T @ B @ V)
        nneg = max(1, np.sum(lams < 0))
        # Rotate our subspace V to be diagonal in A.
        # This is not strictly necessary but it makes our lives easier later
        AV = AV @ vecs
        V = V @ vecs
        vecs = np.eye(V.shape[1])
        if V.shape[1] >= min(n, maxiter):
            return lams, V, AV

        Ytilde = symmetrize_Y(V, AV, symm=symm)
        R = Ytilde - B @ V * lams[np.newaxis, :]
        Rnorm = np.linalg.norm(R, axis=0)

        # a hack for the optbench.org eigensolver convergence test
        if vref is not None:
            x0 = V @ vecs[:, 0]
            if np.abs(x0 @ vref) > vreftol:
                return lams, V, AV

        # Loop over all Ritz values of interest
        for seeking, (rinorm, thetai) in enumerate(
            zip(Rnorm[:nneg], lams[:nneg])
        ):
            # Take the first Ritz value that is not converged, and use it
            # to extend V
            if rinorm > gamma * np.abs(thetai):
                ri = R[:, seeking]
                thetai = lams[seeking]
                break
        else:
            if not escape_used and V.shape[1] < min(n, maxiter):
                probe = _orthogonal_dense_probe(V)
                if probe is not None:
                    Aprobe = A.dot(probe)
                    _mark_secant_rank_cleanup(A)
                    mixed = V.copy()
                    Amixed = AV.copy()
                    mixed[:, 0] += probe
                    Amixed[:, 0] += Aprobe
                    V, Rmix = np.linalg.qr(mixed, mode='reduced')
                    AV = np.linalg.solve(Rmix.T, Amixed.T).T
                    escape_used = True
                    continue
            return lams, V, AV

        t = expand(V, Ytilde, P, B, lams, vecs, thetai, method, seeking)
        t /= np.linalg.norm(t)
        if np.linalg.norm(t - V @ (V.T @ t)) < 1e-2:  # pragma: no cover
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
        escape_used = True


def expand(V, Y, P, B, lams, vecs, shift, method='jd0', seeking=0):
    """Compute the subspace-expansion (correction) vector for one Ritz pair.

    Given the current subspace ``V`` (with ``Y = A @ V``), the Ritz values
    ``lams`` and rotation ``vecs``, returns the search direction that extends
    ``V`` toward the ``seeking``-th Ritz pair, using the ``shift`` and
    preconditioner ``P`` (with metric ``B``). ``method`` selects the correction:
    'lanczos' (bare residual), 'gd' (generalized Davidson), 'jd0'/'mjd0'
    (single/multi-vector Jacobi-Davidson), plus their '_alt' variants.
    """
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
        denom = vi.T @ Pprojv
        if abs(denom) < 1e-12:
            # Fallback when denominator is near zero
            return Pprojr
        alpha = vi.T @ Pprojr / denom
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
