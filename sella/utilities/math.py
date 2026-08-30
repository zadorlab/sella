import numpy as np


def _orthonormalize(X, Y, eps1, eps2, maxiter):
    """Orthonormalize the columns of X against the orthonormal basis Y."""
    nrows, ncols = X.shape
    output = np.empty((nrows, ncols), dtype=np.float64)
    naccepted = 0

    for column in X.T:
        vector = column.astype(np.float64, copy=True)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm):
            raise ValueError("Input contains non-finite values.")
        if norm < eps2:
            continue
        vector /= norm

        for _ in range(maxiter):
            if Y.shape[1]:
                vector -= Y @ (Y.T @ vector)
            if naccepted:
                basis = output[:, :naccepted]
                vector -= basis @ (basis.T @ vector)

            norm = np.linalg.norm(vector)
            if not np.isfinite(norm):
                raise RuntimeError("Modified Gram-Schmidt failed.")
            if norm < eps2:
                break
            vector /= norm

            if abs(1.0 - norm) <= eps1:
                output[:, naccepted] = vector
                naccepted += 1
                break
        else:
            raise RuntimeError("Modified Gram-Schmidt failed to converge.")

    return output[:, :naccepted]


def modified_gram_schmidt(Xin, Yin=None, eps1=1e-15, eps2=1e-6,
                          maxiter=100):
    """Return an orthonormal basis for ``Xin`` orthogonal to ``Yin``.

    Linearly dependent columns are omitted. When supplied, ``Yin`` is copied
    and orthonormalized before it is used as the fixed basis.
    """
    X = np.asarray(Xin)
    if X.ndim != 2:
        raise ValueError("Xin must be a two-dimensional array.")
    if eps1 < 0 or eps2 < 0:
        raise ValueError("eps1 and eps2 must be non-negative.")
    if maxiter < 1:
        raise ValueError("maxiter must be at least one.")

    if Yin is None:
        Y = np.empty((X.shape[0], 0), dtype=np.float64)
    else:
        Yin = np.asarray(Yin)
        if Yin.ndim != 2:
            raise ValueError("Yin must be a two-dimensional array.")
        if Yin.shape[0] != X.shape[0]:
            raise ValueError("Xin and Yin must have the same number of rows.")
        Y = Yin.astype(np.float64, copy=False)
        gram = Y.T @ Y
        identity = np.eye(Y.shape[1])
        if not np.allclose(gram, identity, rtol=1e-12, atol=1e-12):
            empty_basis = np.empty((X.shape[0], 0), dtype=np.float64)
            Y = _orthonormalize(Y, empty_basis, eps1, eps2, maxiter)

    return _orthonormalize(X, Y, eps1, eps2, maxiter)


def pseudo_inverse(A, eps=1e-6):
    """Compute an SVD and a truncated Moore-Penrose pseudoinverse.

    Returns ``(U, singular_values, VT, Ainv, rank)`` for compatibility with
    Sella's former Cython implementation.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")

    U, singular_values, VT = np.linalg.svd(A, full_matrices=True)
    retained = np.flatnonzero(singular_values >= eps)
    if retained.size:
        Ainv = ((VT[retained].T / singular_values[retained])
                @ U[:, retained].T)
    else:
        Ainv = np.zeros(A.T.shape, dtype=np.float64)

    return U, singular_values, VT, Ainv, retained.size
