import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from sella.eigensolvers import exact, rayleigh_ritz
from sella.linalg import NumericalHessian

from test_utils import poly_factory, get_matrix


@pytest.mark.parametrize("dim,order,eta,threepoint",
                         [(10, 4, 1e-6, True),
                          (10, 4, 1e-6, False)])
def test_exact(dim, order, eta, threepoint):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-4, rtol=eta**2)

    poly = poly_factory(dim, order, rng=rng)
    x = rng.normal(size=dim)

    _, g, h = poly(x)

    H = NumericalHessian(lambda x: poly(x)[:2], g0=g, x0=x,
                         eta=eta, threepoint=threepoint)

    l1, V1, AV1 = exact(h)
    l2, V2, AV2 = exact(H)

    np.testing.assert_allclose(l1, l2, **tol)
    np.testing.assert_allclose(np.abs(V1.T @ V2), np.eye(dim), **tol)
    np.testing.assert_allclose(h @ V1, AV1, **tol)
    np.testing.assert_allclose(h @ V2, AV2, **tol)

    P = h + get_matrix(dim, dim, rng=rng) * 1e-3
    l3, V3, AV3 = exact(H, P=P)

    np.testing.assert_allclose(l1, l3, **tol)
    np.testing.assert_allclose(np.abs(V1.T @ V2), np.eye(dim), **tol)


@pytest.mark.parametrize("dim,order,eta,threepoint,gamma,method,maxiter",
                         [(10, 4, 1e-6, False, 0., 'jd0', None),
                          (10, 4, 1e-6, False, 1e-32, 'jd0', 3),
                          (10, 4, 1e-6, True, 1e-1, 'jd0', None),
                          (10, 4, 1e-6, False, 1e-1, 'jd0', None),
                          (10, 4, 1e-6, False, 1e-1, 'lanczos', None),
                          (10, 4, 1e-6, False, 1e-1, 'gd', None),
                          (10, 4, 1e-6, False, 1e-1, 'jd0_alt', None),
                          (10, 4, 1e-6, False, 1e-1, 'mjd0_alt', None),
                          (10, 4, 1e-6, False, 1e-1, 'mjd0', None),
                          ])
def test_rayleigh_ritz(dim, order, eta, threepoint, gamma, method, maxiter):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-4, rtol=eta**2)

    poly = poly_factory(dim, order, rng=rng)
    x = rng.normal(size=dim)

    _, g, h = poly(x)
    H = NumericalHessian(lambda x: poly(x)[:2], g0=g, x0=x,
                         eta=eta, threepoint=threepoint)

    l1, V1, AV1 = rayleigh_ritz(H, gamma, np.eye(dim), method=method,
                                maxiter=maxiter)
    np.testing.assert_allclose(l1, np.linalg.eigh(V1.T @ AV1)[0], **tol)

    v0 = rng.normal(size=dim)
    rayleigh_ritz(H, gamma, np.eye(dim), method=method, v0=v0,
                  maxiter=maxiter, vref=np.linalg.eigh(h)[1][:, 0])


@pytest.mark.parametrize(
    "diagonal",
    [np.array([0.0, 1.0, 2.0]), np.zeros(3), np.array([-1.0, 0.0, 2.0])],
)
def test_rayleigh_ritz_accepts_exact_zero_residual(diagonal):
    A = np.diag(diagonal)
    lams, vecs, avecs = rayleigh_ritz(A, 0.1, np.eye(3), method='jd0')

    assert np.all(np.isfinite(lams))
    assert np.all(np.isfinite(vecs))
    np.testing.assert_allclose(A @ vecs, avecs, atol=1e-14)
    np.testing.assert_allclose(
        avecs[:, 0] - vecs[:, 0] * lams[0], 0.0, atol=1e-14
    )


def test_rayleigh_ritz_exact_seed_does_not_hide_lower_mode():
    A = np.diag([2.0, 4.0, -1.0])
    lams, vecs, avecs = rayleigh_ritz(
        A, 0.1, np.eye(3), method='jd0'
    )

    assert lams[0] == pytest.approx(-1.0)
    np.testing.assert_allclose(A @ vecs, avecs, atol=1e-14)


class _CountingDiagonal(LinearOperator):
    def __init__(self, diagonal):
        self.diagonal = np.asarray(diagonal)
        self.calls = 0
        self.requires_secant_rank_cleanup = False
        super().__init__(np.dtype(float), (len(diagonal), len(diagonal)))

    def _matvec(self, vector):
        self.calls += 1
        return self.diagonal * np.asarray(vector).ravel()


def test_exact_seed_escape_avoids_full_coordinate_scan():
    diagonal = np.arange(1.0, 31.0)
    diagonal[-1] = -10.0
    A = _CountingDiagonal(diagonal)

    lams, _, _ = rayleigh_ritz(A, 0.1, np.eye(30), method='jd0')

    assert lams[0] < -9.0
    assert A.calls <= 10


def test_flat_operator_escape_uses_only_one_extra_probe():
    A = _CountingDiagonal(np.zeros(30))

    rayleigh_ritz(A, 0.1, np.eye(30), method='jd0')

    assert A.calls == 2
    assert A.requires_secant_rank_cleanup


def test_ordinary_expansion_does_not_request_secant_rank_cleanup():
    A = _CountingDiagonal([-2.0, 1.0, 3.0])

    rayleigh_ritz(
        A, 0.1, np.eye(3), method='jd0', v0=np.ones(3)
    )

    assert not A.requires_secant_rank_cleanup


def test_escape_marks_numerical_hessian_through_matrix_sum():
    def flat_func(x):
        return 0.0, np.zeros_like(x)

    H = NumericalHessian(
        flat_func, x0=np.zeros(3), g0=np.zeros(3), eta=1e-5
    )

    rayleigh_ritz(H + np.zeros((3, 3)), 0.1, np.eye(3), method='jd0')

    assert H.calls == 2
    assert H.requires_secant_rank_cleanup


def test_rayleigh_ritz_zero_seed_uses_preconditioner_seed():
    A = np.diag([-2.0, 1.0, 3.0])
    lams, _, _ = rayleigh_ritz(
        A, 0.1, np.eye(3), method='jd0', v0=np.zeros(3)
    )

    assert lams[0] == pytest.approx(-2.0)


def test_exact_generalized_eigenproblem_honors_metric():
    A = np.diag([2.0, 3.0, 5.0])
    B = np.diag([1.0, 4.0, 2.0])
    expected = np.array([0.75, 2.0, 2.5])

    for operator in (A, aslinearoperator(A)):
        lams, vecs, avecs = rayleigh_ritz(
            operator, 0.0, np.eye(3), B=B
        )

        np.testing.assert_allclose(lams, expected, atol=1e-14)
        np.testing.assert_allclose(A @ vecs, avecs, atol=1e-14)
        np.testing.assert_allclose(
            avecs, B @ vecs * lams[np.newaxis, :], atol=1e-14
        )


def test_iterative_generalized_eigenproblem_finds_hidden_lower_mode():
    rng = np.random.default_rng(41)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    metric_values = np.array([0.5, 2.0, 5.0, 11.0])
    eigenvalues = np.array([1.0, 2.0, 3.0, -4.0])
    B = Q @ np.diag(metric_values) @ Q.T
    A = Q @ np.diag(eigenvalues * metric_values) @ Q.T
    P = Q @ np.diag([1.0, 2.0, 3.0, 4.0]) @ Q.T

    lams, vecs, avecs = rayleigh_ritz(
        aslinearoperator(A), 1e-3, P, B=B, method='jd0'
    )

    assert lams[0] == pytest.approx(-4.0, abs=1e-3)
    np.testing.assert_allclose(
        avecs[:, 0], B @ vecs[:, 0] * lams[0], atol=1e-3
    )
