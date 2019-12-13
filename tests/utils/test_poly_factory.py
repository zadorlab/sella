import pytest

import numpy as np

from .poly_factory import poly_factory


@pytest.mark.parametrize("dim,order", [(1, 1), (2, 2), (10, 5)])
def test_poly_factory(dim, order, eta=1e-6, atol=1e-4):
    rng = np.random.RandomState(1)

    tol = dict(atol=atol, rtol=eta**2)

    my_poly = poly_factory(dim, order)
    x0 = rng.normal(size=dim)
    f0, g0, h0 = my_poly(x0)

    g_numer = np.zeros_like(g0)
    h_numer = np.zeros_like(h0)
    for i in range(dim):
        x = x0.copy()
        x[i] += eta
        fplus, gplus, _ = my_poly(x)
        x[i] = x0[i] - eta
        fminus, gminus, _ = my_poly(x)
        g_numer[i] = (fplus - fminus) / (2 * eta)
        h_numer[i] = (gplus - gminus) / (2 * eta)

    np.testing.assert_allclose(g0, g_numer, **tol)
    np.testing.assert_allclose(h0, h_numer, **tol)
