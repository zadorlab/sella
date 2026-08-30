import numpy as np

from sella.linalg import ApproximateHessian
from sella.optimize.restricted_step import TrustRegion
from sella.optimize.stepper import (
    PartitionedRationalFunctionOptimization,
    QuasiNewton,
    RationalFunctionOptimization,
)


class _StationaryQuadraticPES:
    dim = 3

    def __init__(self):
        self.H = ApproximateHessian(3, 3, np.eye(3))

    def get_g(self):
        return np.zeros(3)

    def get_scons(self):
        return np.zeros(3)

    def get_H(self):
        return self.H

    def get_fast_restricted_step_data(self, *args):
        return None

    def get_Ufree(self):
        return np.eye(3)

    def get_HL_projected(self, basis):
        return self.H.project(basis)


class _QuadraticPES(_StationaryQuadraticPES):
    def __init__(self, gradient, hessian):
        self.g = np.asarray(gradient)
        self.H = ApproximateHessian(3, 3, np.asarray(hessian))

    def get_g(self):
        return self.g


def test_trust_region_accepts_zero_step_at_stationary_point():
    step, magnitude = TrustRegion(
        _StationaryQuadraticPES(), order=0, delta=0.1
    ).get_s()

    np.testing.assert_array_equal(step, np.zeros(3))
    assert magnitude == 0.0


def _assert_step_derivative(stepper, alpha):
    delta = 1e-6
    _, analytic = stepper.get_s(alpha)
    plus = stepper.get_s(alpha + delta)[0]
    minus = stepper.get_s(alpha - delta)[0]
    numeric = (plus - minus) / (2 * delta)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-7)


def test_quasi_newton_ascent_step_derivative():
    H = ApproximateHessian(4, 4, np.diag([-2.0, -0.7, 1.3, 2.8]))
    g = np.array([0.3, -0.2, 0.4, 0.1])
    _assert_step_derivative(QuasiNewton(g, H, order=2), alpha=0.4)


def test_rfo_step_derivative():
    H = ApproximateHessian(4, 4, np.diag([-2.0, -0.7, 1.3, 2.8]))
    g = np.array([0.3, -0.2, 0.4, 0.1])
    _assert_step_derivative(
        RationalFunctionOptimization(g, H, order=1), alpha=0.4
    )


def test_partitioned_rfo_step_derivative():
    H = ApproximateHessian(4, 4, np.diag([-2.0, -0.7, 1.3, 2.8]))
    g = np.array([0.3, -0.2, 0.4, 0.1])
    _assert_step_derivative(
        PartitionedRationalFunctionOptimization(g, H, order=1), alpha=0.4
    )


def test_only_order_one_partitioned_rfo_is_newton_safe():
    H = ApproximateHessian(4, 4, np.diag([-2.0, -0.7, 1.3, 2.8]))
    g = np.array([0.3, -0.2, 0.4, 0.1])

    assert PartitionedRationalFunctionOptimization(
        g, H, order=1
    ).newton_safe
    assert not PartitionedRationalFunctionOptimization(
        g, H, order=2
    ).newton_safe


def test_newton_safe_restricted_step_periodically_bisects():
    class StalledNewtonStepper:
        alpha0 = 1.0
        alphamin = 0.0
        alphamax = 1.0
        slope = 1.0
        newton_safe = True

    restricted = TrustRegion.__new__(TrustRegion)
    restricted.delta = 0.25
    restricted.tol = 1e-10
    restricted.maxiter = 1000
    restricted.stepper = StalledNewtonStepper()
    restricted.eval = lambda alpha: (
        np.array([alpha]), alpha, 1e12,
    )

    step, magnitude = restricted.get_s()

    np.testing.assert_allclose(step, [0.25], atol=1e-10)
    assert magnitude == restricted.delta


def test_restricted_step_exposes_prfo_spectrum_and_basis():
    pes = _QuadraticPES(
        [0.3, -0.2, 0.4], np.diag([-2.0, -0.7, 1.3])
    )
    restricted = TrustRegion(
        pes, order=1, delta=10.0,
        method=PartitionedRationalFunctionOptimization,
    )

    assert restricted.projection_basis is not None
    np.testing.assert_allclose(
        restricted.projected_eigenvalues,
        [-2.0, -0.7, 1.3],
    )


def test_partitioned_rfo_weak_pole_matches_full_eigensolve():
    hessian = ApproximateHessian(
        3, 3, np.diag([-600.0, -560.0, 20.0])
    )
    gradient = np.array([4e-4, 4e-5, 5e-4])
    prfo = PartitionedRationalFunctionOptimization(
        gradient, hessian, order=1
    )

    vmax = hessian.evecs[:, :1]
    vmin = hessian.evecs[:, 1:]
    full_max = RationalFunctionOptimization(
        vmax.T @ gradient, hessian.project(vmax), order=1
    )
    full_min = RationalFunctionOptimization(
        vmin.T @ gradient, hessian.project(vmin), order=0
    )

    alpha = 0.03
    expected_step = (
        vmax @ full_max.get_s(alpha)[0]
        + vmin @ full_min.get_s(alpha)[0]
    )
    expected_derivative = (
        vmax @ full_max.get_s(alpha)[1]
        + vmin @ full_min.get_s(alpha)[1]
    )
    step, derivative = prfo.get_s(alpha)

    np.testing.assert_allclose(step, expected_step, rtol=1e-12)
    np.testing.assert_allclose(
        derivative, expected_derivative, rtol=1e-12, atol=1e-12
    )


def test_partitioned_rfo_matches_full_augmented_eigensolves():
    rng = np.random.default_rng(12)
    for size, order in ((12, 1), (80, 1), (120, 60)):
        matrix = rng.normal(size=(size, size))
        matrix = 0.5 * (matrix + matrix.T)
        gradient = rng.normal(size=size)
        H = ApproximateHessian(size, size, matrix)
        prfo = PartitionedRationalFunctionOptimization(
            gradient, H, order=order
        )

        eigenvalues = H.evals
        eigenvectors = H.evecs
        vmax = eigenvectors[:, :order]
        vmin = eigenvectors[:, order:]
        full_max = RationalFunctionOptimization(
            vmax.T @ gradient,
            ApproximateHessian(
                order, 0, np.diag(eigenvalues[:order])
            ),
            order=order,
        )
        full_min = RationalFunctionOptimization(
            vmin.T @ gradient,
            ApproximateHessian(
                size - order, 0, np.diag(eigenvalues[order:])
            ),
            order=0,
        )

        for alpha in (1.0, 0.4, 0.05):
            smax, dsmax = full_max.get_s(alpha)
            smin, dsmin = full_min.get_s(alpha)
            expected = (
                vmax @ smax + vmin @ smin,
                vmax @ dsmax + vmin @ dsmin,
            )
            actual = prfo.get_s(alpha)
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
            np.testing.assert_allclose(actual[1], expected[1], atol=1e-11)


def test_partitioned_rfo_handles_degenerate_stationary_hessian():
    H = ApproximateHessian(3, 3, np.diag([0.0, 0.0, 1.0]))
    prfo = PartitionedRationalFunctionOptimization(
        np.zeros(3), H, order=1
    )

    step, derivative = prfo.get_s(0.4)

    np.testing.assert_allclose(step, 0.0)
    np.testing.assert_allclose(derivative, 0.0)


def test_interior_rfo_reaches_trust_region_boundary():
    pes = _QuadraticPES(
        [0.3, -0.2, 0.4], np.diag([-2.0, -0.7, 1.3])
    )
    delta = 0.01
    step, reported = TrustRegion(
        pes, order=1, delta=delta, method=RationalFunctionOptimization
    ).get_s()

    np.testing.assert_allclose(np.linalg.norm(step), delta, atol=1e-10)
    np.testing.assert_allclose(reported, delta, atol=1e-10)
