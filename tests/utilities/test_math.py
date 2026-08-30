import pytest
import numpy as np

from sella.utilities.math import pseudo_inverse, modified_gram_schmidt

@pytest.mark.parametrize("n,m,eps",
                         [(3, 3, 1e-10),
                          (3, 5, 1e-10),
                          (100, 3, 1e-6),
                          ])
def test_mppi(n, m, eps):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-6, rtol=1e-6)

    A = rng.normal(size=(n, m))
    U1, s1, VT1, Ainv, nsing1 = pseudo_inverse(A.copy(), eps=eps)

    A_test = U1[:, :nsing1] @ np.diag(s1) @ VT1[:nsing1, :]
    np.testing.assert_allclose(A_test, A, **tol)

    Ainv_test = np.linalg.pinv(A)
    np.testing.assert_allclose(Ainv_test, Ainv, **tol)

    nsingB = nsing1 - 1
    B = U1[:, :nsingB] @ np.diag(s1[:nsingB]) @ VT1[:nsingB, :]
    U2, s2, VT2, Binv, nsing2 = pseudo_inverse(B.copy(), eps=eps)
    assert nsing2 == nsingB
    np.testing.assert_allclose(B @ Binv @ B, B, **tol)


def test_pseudo_inverse_zero_rank():
    A = np.zeros((4, 2))
    U, s, VT, Ainv, rank = pseudo_inverse(A)

    assert U.shape == (4, 4)
    assert s.shape == (2,)
    assert VT.shape == (2, 2)
    assert Ainv.shape == (2, 4)
    assert rank == 0
    np.testing.assert_array_equal(Ainv, 0.0)


@pytest.mark.parametrize("n,mx,my,eps1,eps2,maxiter",
                         [(3, 2, 1, 1e-15, 1e-6, 100),
                          (100, 50, 25, 1e-15, 1e-6, 100),
                          ])
def test_modified_gram_schmidt(n, mx, my, eps1, eps2, maxiter):
    rng = np.random.RandomState(2)

    tol = dict(atol=1e-6, rtol=1e-6)
    mgskw = dict(eps1=eps1, eps2=eps2, maxiter=maxiter)

    X = rng.normal(size=(n, mx))

    Xout1 = modified_gram_schmidt(X, **mgskw)
    _, nxout1 = Xout1.shape

    np.testing.assert_allclose(Xout1.T @ Xout1, np.eye(nxout1), **tol)
    np.testing.assert_allclose(np.linalg.det(X.T @ X),
                               np.linalg.det(X.T @ Xout1)**2, **tol)


    Y = rng.normal(size=(n, my))
    Xout2 = modified_gram_schmidt(X, Y, **mgskw)
    _, nxout2 = Xout2.shape

    np.testing.assert_allclose(Xout2.T @ Xout2, np.eye(nxout2), **tol)
    np.testing.assert_allclose(Xout2.T @ Y, np.zeros((nxout2, my)), **tol)

    X[:, 1] = X[:, 0]

    Xout3 = modified_gram_schmidt(X, **mgskw)
    _, nxout3 = Xout3.shape
    assert nxout3 == nxout1 - 1

    np.testing.assert_allclose(Xout2.T @ Xout2, np.eye(nxout2), **tol)


def test_modified_gram_schmidt_empty_and_invalid_inputs():
    output = modified_gram_schmidt(np.empty((5, 0)))
    assert output.shape == (5, 0)

    with pytest.raises(ValueError, match="two-dimensional"):
        modified_gram_schmidt(np.ones(5))
    with pytest.raises(ValueError, match="same number of rows"):
        modified_gram_schmidt(np.ones((5, 1)), np.ones((4, 1)))


def test_modified_gram_schmidt_maxiter():
    rng = np.random.RandomState(3)
    with pytest.raises(RuntimeError, match="converge"):
        modified_gram_schmidt(rng.normal(size=(10, 2)), maxiter=1)


def test_modified_gram_schmidt_orthonormal_basis_fast_path():
    rng = np.random.RandomState(4)
    Y, _ = np.linalg.qr(rng.normal(size=(20, 5)))
    Y_before = Y.copy()

    output = modified_gram_schmidt(rng.normal(size=(20, 2)), Y)

    np.testing.assert_array_equal(Y, Y_before)
    np.testing.assert_allclose(output.T @ output, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(output.T @ Y, 0.0, atol=1e-12)
