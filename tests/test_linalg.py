import pytest

import numpy as np
from scipy.stats import ortho_group

from sella.linalg import NumericalHessian

from test_utils import poly_factory


@pytest.mark.parametrize("dim,subdim,order,threepoint",
                         [(3, None, 1, False),
                          (3, None, 1, True),
                          (5, 3, 2, True),
                          (10, None, 4, True),
                          (10, 6, 4, False)])
def test_NumericalHessian(dim, subdim, order, threepoint, eta=1e-6, atol=1e-4):
    rng = np.random.RandomState(2)
    tol = dict(rtol=atol, atol=eta**2)

    x = rng.normal(size=dim)

    poly1 = poly_factory(dim, order, rng)
    _, g1, h1 = poly1(x)

    poly2 = poly_factory(dim, order, rng)
    _, g2, h2 = poly2(x)

    if subdim is None:
        U = None
        subdim = dim
        g1proj = g1
        xproj = x
    else:
        U = ortho_group.rvs(dim, random_state=rng)[:, :subdim]
        h1 = U.T @ h1 @ U
        h2 = U.T @ h2 @ U
        g1proj = U.T @ g1
        xproj = U.T @ x

    Hkwargs = dict(x0=x, eta=eta, threepoint=threepoint, Uproj=U)

    H1 = NumericalHessian(lambda x: poly1(x)[:2], g0=g1, **Hkwargs)

    # M1: some random matrix
    M1 = rng.normal(size=(subdim, subdim))

    H2 = H1 + NumericalHessian(lambda x: poly2(x)[:2], g0=g2, **Hkwargs) + M1
    H3 = h1 + h2 + M1

    # Make first column orthogonal to g1
    M1[:, 0] = xproj - g1proj * (xproj @ g1proj) / (g1proj @ g1proj)

    # Make second column orthogonal to g1 and x
    M1[:, 1] -= M1[:, 0] * (M1[:, 1] @ M1[:, 0]) / (M1[:, 0] @ M1[:, 0])
    M1[:, 1] -= g1proj * (M1[:, 1] @ g1proj) / (g1proj @ g1proj)

    np.testing.assert_allclose(H2.T.dot(M1), H3.T @ M1, **tol)
