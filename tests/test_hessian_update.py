import pytest
import numpy as np
from scipy.linalg import eigh, lstsq

from test_utils import get_matrix

from sella.hessian_update import update_H, _MS_TS_BFGS, _MS_PSB


@pytest.mark.parametrize("dim,subdim,method,symm, pd",
                         [(10, 1, 'TS-BFGS', 2, False),
                          (10, 2, 'TS-BFGS', 0, False),
                          (10, 2, 'TS-BFGS', 1, False),
                          (10, 2, 'TS-BFGS', 2, False),
                          (10, 2, 'BFGS', 2, False),
                          (10, 2, 'PSB', 2, False),
                          (10, 2, 'DFP', 2, False),
                          (10, 2, 'SR1', 2, False),
                          (10, 2, 'Greenstadt', 2, False),
                          (10, 2, 'BFGS_auto', 2, False),
                          (10, 2, 'BFGS_auto', 2, True),
                          ])
def test_update_H(dim, subdim, method, symm, pd):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-6, rtol=1e-6)

    B = get_matrix(dim, dim, pd, True, rng=rng)
    H = get_matrix(dim, dim, pd, True, rng=rng)

    S = get_matrix(dim, subdim, rng=rng)
    Y = H @ S

    B1 = update_H(None, S, Y, method=method, symm=symm)
    np.testing.assert_allclose(B1 @ S, Y, **tol)

    B2 = update_H(B, S, Y, method=method, symm=symm)
    np.testing.assert_allclose(B2 @ S, Y, **tol)

    if subdim == 1:
        B3 = update_H(B, S.ravel(), Y.ravel(), method=method, symm=symm)
        np.testing.assert_allclose(B2, B3, **tol)

        B4 = update_H(B, S.ravel() / 1e12, Y.ravel() / 1e12, method=method,
                      symm=symm)
        np.testing.assert_allclose(B, B4, atol=0, rtol=0)


def test_rank_one_ts_bfgs_matches_general_formula():
    """The rank-one fast path must match the old multi-secant expression."""
    rng = np.random.RandomState(2)
    dim = 50
    A = rng.normal(size=(dim, dim))
    B = 0.5 * (A + A.T)
    S = rng.normal(size=(dim, 1))
    Y = rng.normal(size=(dim, 1))
    lams, vecs = eigh(B)

    J = Y - B @ S
    X1 = S.T @ Y @ Y.T
    absBS = vecs @ (np.abs(lams[:, np.newaxis]) * (vecs.T @ S))
    X2 = S.T @ absBS @ absBS.T
    XS = (X1 + X2) @ S
    if np.linalg.cond(XS) > 1e12:
        expected = _MS_PSB(B, S, Y)
    else:
        U = lstsq(XS, X1 + X2)[0].T
        UJT = U @ J.T
        expected = (UJT + UJT.T) - U @ (J.T @ S) @ U.T

    actual = _MS_TS_BFGS(B, S, Y, lams, vecs)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
