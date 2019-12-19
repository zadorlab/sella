import pytest
import numpy as np

from test_utils import get_matrix

from sella.hessian_update import update_H


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
