import pytest
import numpy as np

from sella.utilities.math import pseudo_inverse, modified_gram_schmidt

from test_utils import get_matrix

import pyximport
pyximport.install(language_level=3)
from math_wrappers import wrappers

# TODO: figure out why m > n crashes
@pytest.mark.parametrize("n,m,eps",
                         [(3, 3, 1e-10),
                          (100, 3, 1e-6),
                          ])
def test_mppi(n, m, eps):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-6, rtol=1e-6)

    A = get_matrix(n, m, rng=rng)
    U1, s1, VT1, Ainv, nsing1 = pseudo_inverse(A.copy(), eps=eps)

    A_test = U1[:, :nsing1] @ np.diag(s1) @ VT1[:nsing1, :]
    np.testing.assert_allclose(A_test, A, **tol)

    Ainv_test = np.linalg.pinv(A)
    np.testing.assert_allclose(Ainv_test, Ainv, **tol)

    nsingB = nsing1 - 1
    B = U1[:, :nsingB] @ np.diag(s1[:nsingB]) @ VT1[:nsingB, :]
    U2, s2, VT2, Binv, nsing2 = pseudo_inverse(B.copy(), eps=eps)


@pytest.mark.parametrize("n,mx,my,eps1,eps2,maxiter",
                         [(3, 2, 1, 1e-15, 1e-6, 100),
                          (100, 50, 25, 1e-15, 1e-6, 100),
                          ])
def test_modified_gram_schmidt(n, mx, my, eps1, eps2, maxiter):
    rng = np.random.RandomState(2)

    tol = dict(atol=1e-6, rtol=1e-6)
    mgskw = dict(eps1=eps1, eps2=eps2, maxiter=maxiter)

    X = get_matrix(n, mx, rng=rng)

    Xout1 = modified_gram_schmidt(X, **mgskw)
    _, nxout1 = Xout1.shape

    np.testing.assert_allclose(Xout1.T @ Xout1, np.eye(nxout1), **tol)
    np.testing.assert_allclose(np.linalg.det(X.T @ X),
                               np.linalg.det(X.T @ Xout1)**2, **tol)


    Y = get_matrix(n, my, rng=rng)
    Xout2 = modified_gram_schmidt(X, Y, **mgskw)
    _, nxout2 = Xout2.shape

    np.testing.assert_allclose(Xout2.T @ Xout2, np.eye(nxout2), **tol)
    np.testing.assert_allclose(Xout2.T @ Y, np.zeros((nxout2, my)), **tol)

    X[:, 1] = X[:, 0]

    Xout3 = modified_gram_schmidt(X, **mgskw)
    _, nxout3 = Xout3.shape
    assert nxout3 == nxout1 - 1

    np.testing.assert_allclose(Xout2.T @ Xout2, np.eye(nxout2), **tol)


@pytest.mark.parametrize('rngstate,length',
                         [(0, 1),
                          (1, 2),
                          (2, 10),
                          (3, 1024),
                          (4, 0)])
def test_normalize(rngstate, length):
    rng = np.random.RandomState(rngstate)
    x = rng.normal(size=(length,))
    wrappers['normalize'](x)

    if length > 0:
        assert abs(np.linalg.norm(x) - 1.) < 1e-14

@pytest.mark.parametrize('rngstate,length,scale',
                         [(0, 1, 1.),
                          (1, 3, 0.5),
                          (2, 100, 4.),
                          (3, 10, 0.),
                          (4, 0, 1.)])
def test_vec_sum(rngstate, length, scale):
    rng = np.random.RandomState(rngstate)
    x = rng.normal(size=(length,))
    y = rng.normal(size=(length,))
    z = np.zeros(length)
    err = wrappers['vec_sum'](x, y, z, scale)
    assert err == 0
    np.testing.assert_allclose(z, x + scale * y)

    if length > 0:
        assert wrappers['vec_sum'](x, y[:length-1], z, scale) == -1
        assert wrappers['vec_sum'](x, y, z[:length-1], scale) == -1

@pytest.mark.parametrize('rngstate,n,m',
                         [(0, 1, 1),
                          (1, 5, 5),
                          (2, 3, 7),
                          (3, 8, 4),
                          (6, 100, 100)])
def test_symmetrize(rngstate, n, m):
    rng = np.random.RandomState(rngstate)
    X = get_matrix(n, m, rng=rng)
    minnm = min(n, m)
    Y = X[:minnm, :minnm]
    wrappers['symmetrize'](Y)
    np.testing.assert_allclose(Y, Y.T)


@pytest.mark.parametrize('rngstate,scale',
                         [(0, 1.),
                          (1, 0.1),
                          (2, 100.)])
def test_skew(rngstate, scale):
    rng = np.random.RandomState(rngstate)
    x = rng.normal(size=(3,))
    Y = get_matrix(3, 3, rng=rng)
    wrappers['skew'](x, Y, scale)
    np.testing.assert_allclose(scale * np.cross(np.eye(3), x), Y)

@pytest.mark.parametrize('rngstate,n,mx,my',
                         [(2, 10, 2, 4)])
def test_mgs(rngstate, n, mx, my):
    rng = np.random.RandomState(rngstate)
    X = get_matrix(n, mx, rng=rng)
    assert wrappers['mgs'](X, None, maxiter=1) < 0
    X = get_matrix(n, mx, rng=rng)
    assert wrappers['mgs'](X, None, eps2=1e10) == 0
    X = get_matrix(n, mx, rng=rng)
    Y = get_matrix(n, my, rng=rng)
    assert wrappers['mgs'](X, Y, eps2=1e10) == 0
    Y = get_matrix(n, my, rng=rng)
    my2 = wrappers['mgs'](Y, None)
    assert my2 >= 0
    np.testing.assert_allclose(Y[:, :my2].T @ Y[:, :my2], np.eye(my2),
                               atol=1e-10)
    X = get_matrix(n, mx, rng=rng)
    mx2 = wrappers['mgs'](X, Y)
    assert mx2 >= 0
    np.testing.assert_allclose(X[:, :mx2].T @ X[:, :mx2], np.eye(mx2),
                               atol=1e-10)
    np.testing.assert_allclose(X[:, :mx2].T @ Y[:, :my2], np.zeros((mx2, my2)),
                               atol=1e-10)
    assert wrappers['mgs'](X, Y[:n-1]) < 0

