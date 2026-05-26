"""
General tests for core Sella functionality.

These tests exercise key functionality including:
- Hessian operations and arithmetic
- Internal coordinate calculations
- Trust region optimization steps
- Eigensolvers
"""
import pytest
import numpy as np

from ase.build import molecule
from ase.calculators.emt import EMT

from sella.linalg import ApproximateHessian, NumericalHessian, SparseInternalHessians
from sella.internal import Internals
from sella.peswrapper import PES, InternalPES
from sella.eigensolvers import exact, rayleigh_ritz


class TestApproximateHessian:
    """Test ApproximateHessian operations."""

    def test_hessian_arithmetic(self):
        """Test that ApproximateHessian supports addition with arrays."""
        dim = 5
        ncart = 5
        rng = np.random.RandomState(42)

        # Create initialized Hessian
        H1 = ApproximateHessian(dim, ncart, update_method='BFGS')
        # Initialize with a step
        s = rng.normal(size=dim)
        y = rng.normal(size=dim)
        s /= np.linalg.norm(s)
        y /= np.linalg.norm(y)
        H1.update(s, y)
        assert H1.initialized

        # Add to an array
        M = rng.normal(size=(dim, dim))
        M = 0.5 * (M + M.T)

        result = H1 + M
        expected = H1.B + M
        # Result is an ApproximateHessian, compare underlying matrices
        np.testing.assert_allclose(result.B, expected, atol=1e-10)

    def test_hessian_addition_with_uninitialized(self):
        """Test adding two Hessians where one is uninitialized."""
        dim = 5
        ncart = 5
        rng = np.random.RandomState(42)

        H1 = ApproximateHessian(dim, ncart, update_method='BFGS')
        H2 = ApproximateHessian(dim, ncart, update_method='BFGS')

        # Initialize H1 only
        s = rng.normal(size=dim)
        y = rng.normal(size=dim)
        H1.update(s, y)

        # Adding uninitialized H2 should work
        result = H1 + H2
        # Result should be H1.B + H2.B (diagonal for uninitialized)
        assert result is not None

    def test_eigendecomposition(self):
        """Test eigenvalue decomposition of ApproximateHessian."""
        dim = 6
        ncart = 6
        rng = np.random.RandomState(42)

        H = ApproximateHessian(dim, ncart, update_method='BFGS')

        # Initialize with multiple updates
        for _ in range(3):
            s = rng.normal(size=dim)
            y = rng.normal(size=dim)
            H.update(s, y)

        # Access eigenvalues
        evals = H.evals
        evecs = H.evecs
        assert evals is not None
        assert evecs is not None
        assert len(evals) == dim

        # Verify eigendecomposition
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        np.testing.assert_allclose(H.B, reconstructed, atol=1e-10)


class TestSparseInternalHessians:
    """Test SparseInternalHessians functionality."""

    def test_numpy_array_conversion(self):
        """Test that SparseInternalHessians can be converted to numpy array."""
        # Create a simple molecule
        atoms = molecule('H2O')
        internal = Internals(atoms)
        internal.find_all_bonds()
        internal.find_all_angles()

        # Get the Hessian
        hess = internal.hessian()
        assert isinstance(hess, SparseInternalHessians)

        # Convert to numpy array
        arr = np.asarray(hess)
        assert isinstance(arr, np.ndarray)

        # Check shape consistency
        n = len(internal.calc())
        assert arr.shape == (n, 3 * len(atoms), 3 * len(atoms))


class TestInternals:
    """Test internal coordinate functionality."""

    def test_basic_internal_coords(self):
        """Test basic internal coordinate creation and calculation."""
        atoms = molecule('CH4')
        internal = Internals(atoms)

        # Find standard internals
        internal.find_all_bonds()
        internal.find_all_angles()

        # Calculate internal coords
        q = internal.calc()
        assert len(q) > 0
        assert not np.any(np.isnan(q))

        # Calculate Jacobian
        jac = internal.jacobian()
        assert jac.shape[0] == len(q)
        assert jac.shape[1] == 3 * len(atoms)

    def test_water_molecule(self):
        """Test internal coordinates for water molecule."""
        atoms = molecule('H2O')
        internal = Internals(atoms)

        internal.find_all_bonds()
        internal.find_all_angles()

        q = internal.calc()
        jac = internal.jacobian()

        # Should have 2 bonds and 1 angle
        assert len(q) >= 3

        # Values should be finite
        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(jac))


class TestPES:
    """Test PES wrapper functionality."""

    def test_pes_basic_operations(self):
        """Test basic PES operations."""
        atoms = molecule('H2O')
        atoms.calc = EMT()

        pes = PES(atoms)
        pes.kick(0., diag=True, gamma=0.1)

        # Should be able to get gradient and Hessian
        g = pes.get_g()
        assert g is not None
        assert len(g) == 3 * len(atoms)

        H = pes.get_H()
        assert H is not None

    def test_internal_pes_operations(self):
        """Test InternalPES operations."""
        atoms = molecule('H2O')
        atoms.calc = EMT()

        # Create internals first
        internal = Internals(atoms)
        internal.find_all_bonds()
        internal.find_all_angles()

        pes = InternalPES(atoms, internal)
        pes.kick(0., diag=True, gamma=0.1)

        # Get projection matrices
        Ufree = pes.get_Ufree()
        Ucons = pes.get_Ucons()

        # Free and constrained spaces should be orthogonal
        overlap = Ufree.T @ Ucons
        np.testing.assert_allclose(overlap, 0, atol=1e-10)


class TestEigensolvers:
    """Test eigensolver functionality."""

    def test_exact_eigensolver(self):
        """Test exact diagonalization."""
        dim = 5
        rng = np.random.RandomState(42)

        # Create symmetric matrix
        A = rng.normal(size=(dim, dim))
        A = 0.5 * (A + A.T)

        lams, vecs, Avecs = exact(A)

        # Check eigenvalue equation
        for i in range(dim):
            np.testing.assert_allclose(
                A @ vecs[:, i], lams[i] * vecs[:, i], atol=1e-10
            )

        # Check that Avecs is correct
        np.testing.assert_allclose(Avecs, lams[np.newaxis, :] * vecs, atol=1e-10)

    def test_rayleigh_ritz_small_gamma(self):
        """Test Rayleigh-Ritz eigensolver with reasonable parameters."""
        dim = 6
        rng = np.random.RandomState(42)

        # Create positive definite matrix (easier for convergence)
        A = rng.normal(size=(dim, dim))
        A = A.T @ A + 0.1 * np.eye(dim)

        P = np.eye(dim)
        gamma = 0.1  # Larger gamma for easier convergence

        lams, vecs, Avecs = rayleigh_ritz(A, gamma, P, maxiter=20)

        # Should have found some eigenvalues
        assert len(lams) > 0


class TestNumericalHessian:
    """Test numerical Hessian functionality."""

    def test_matvec_with_zero_vector(self):
        """Test that matvec handles zero input correctly."""
        def simple_func(x):
            return 0.5 * np.sum(x**2), x

        x0 = np.array([1.0, 2.0, 3.0])
        g0 = x0.copy()

        H = NumericalHessian(simple_func, x0, g0, eta=1e-5)

        # Apply to zero vector
        result = H @ np.zeros(3)
        np.testing.assert_allclose(result, np.zeros(3), atol=1e-14)

    def test_matvec_symmetry(self):
        """Test that numerical Hessian is approximately symmetric."""
        def quadratic_func(x):
            A = np.array([[2.0, 0.5], [0.5, 3.0]])
            return 0.5 * x @ A @ x, A @ x

        x0 = np.array([1.0, 1.0])
        _, g0 = quadratic_func(x0)

        H = NumericalHessian(quadratic_func, x0, g0, eta=1e-5, threepoint=True)

        # Build full matrix
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])

        H11 = (H @ e1)[0]
        H12 = (H @ e1)[1]
        H21 = (H @ e2)[0]
        H22 = (H @ e2)[1]

        # Check symmetry
        np.testing.assert_allclose(H12, H21, rtol=1e-5)
