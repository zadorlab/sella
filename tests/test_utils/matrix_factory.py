import numpy as np

def get_matrix(n, m, pd=False, symm=False, rng=None):
    """Generates a random n-by-m matrix"""
    if rng is None:
        rng = np.random.RandomState(1)
    A = rng.normal(size=(n, m))
    if symm:
        assert n == m
        A = 0.5 * (A + A.T)
    if pd:
        assert n == m
        lams, vecs = np.linalg.eigh(A)
        A = vecs @ (np.abs(lams)[:, np.newaxis] * vecs.T)
    return A
