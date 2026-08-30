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

from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT

from sella import Sella
from sella.linalg import ApproximateHessian, NumericalHessian, SparseInternalHessians
from sella.internal import Constraints, Dihedral, Internals
from sella.peswrapper import PES, InternalPES
from sella.eigensolvers import exact, rayleigh_ritz
from sella.optimize import stepper as stepper_module


class TestApproximateHessian:
    """Test ApproximateHessian operations."""

    def test_dense_initialized_false_is_respected(self):
        H0 = 2.0 * np.eye(2)
        H = ApproximateHessian(2, 2, H0, initialized=False)

        assert H.initialized is False
        np.testing.assert_allclose(H.asarray(), H0)

        H.update(np.zeros(2), np.ones(2))
        assert H.initialized is True
        np.testing.assert_allclose(H.asarray(), H0)

        H.set_B(np.eye(2))
        assert H.initialized is True

    def test_tiny_uninitialized_update_is_finite_identity(self):
        H = ApproximateHessian(3, 3)

        H.update(np.zeros(3), np.ones(3))

        assert H.initialized
        np.testing.assert_allclose(H.asarray(), np.eye(3))
        assert np.isfinite(H.asarray()).all()

    def test_tiny_gpu_resident_update_is_noop(self):
        class FakeGpuArray:
            shape = (3, 3)

        H = ApproximateHessian(3, 3, np.eye(3), initialized=True)
        fake_gpu = FakeGpuArray()
        H.B = None
        H._cpu_current = False
        H._B_gpu = fake_gpu
        H._evals_gpu = object()
        H._evecs_gpu = object()

        H.update(np.zeros(3), np.ones(3))

        assert H.initialized
        assert H.B is None
        assert H._B_gpu is fake_gpu
        assert H._cpu_current is False

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

    def test_stepper_eigh_falls_back_after_lapack_error(self, monkeypatch):
        """PRFO should retry a robust LAPACK driver after dsyevr failures."""
        real_eigh = stepper_module.eigh
        calls = []

        def flaky_eigh(A, *args, **kwargs):
            calls.append(kwargs.get('driver'))
            if kwargs.get('driver') is None:
                raise np.linalg.LinAlgError("Internal Error.")
            return real_eigh(A, *args, **kwargs)

        monkeypatch.setattr(stepper_module, 'eigh', flaky_eigh)
        A = np.array([[2.0, 0.1], [0.1, -1.0]])

        vals, vecs = stepper_module._eigh_symmetric(A)

        assert calls[:2] == [None, 'evd']
        np.testing.assert_allclose(
            vecs @ np.diag(vals) @ vecs.T,
            A,
            atol=1e-12,
        )

    def test_stepper_eigh_rejects_asymmetric_matrix(self):
        A = np.array([[1.0, 1e-4], [0.0, 2.0]])

        with pytest.raises(ValueError, match="non-symmetric"):
            stepper_module._eigh_symmetric(A)


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

    def test_hessian_rdot_mat_matches_matrix_product(self):
        """Direct HVP contractions should match hessian_rdot(v) @ mat."""
        water1 = molecule('H2O')
        water2 = molecule('H2O')
        water2.positions += [5.0, 0.0, 0.0]
        atoms = water1 + water2

        internal = Internals(atoms, allow_fragments=True)
        internal.find_all_bonds()
        internal.find_all_angles()
        internal.find_all_dihedrals()

        rng = np.random.RandomState(7)
        v = rng.normal(size=internal.ndof)
        mat = rng.normal(size=(internal.ndof, 3))

        expected = np.asarray(internal.hessian_rdot(v) @ mat)
        actual = internal.hessian_rdot_mat(v, mat)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

        w = rng.normal(size=internal.ndof)
        expected_vec = np.asarray(internal.hessian_rdot(v) @ w).ravel()
        actual_vec = internal.hessian_rdot_mat(v, w)
        np.testing.assert_allclose(actual_vec, expected_vec, atol=1e-12)

        internal._active['angles'][0] = False
        expected_inactive = np.asarray(internal.hessian_rdot(v) @ mat)
        actual_inactive = internal.hessian_rdot_mat(v, mat)
        np.testing.assert_allclose(actual_inactive, expected_inactive,
                                   atol=1e-12)

    def test_constraint_wrap_respects_inactive_inequality_offsets(self):
        """Inactive constraints before a dihedral must not shift wrap offsets."""
        atoms = Atoms(
            'CCCC',
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.2],
            ],
        )
        current = Dihedral((0, 1, 2, 3)).calc(atoms)
        cons = Constraints(atoms)
        cons.fix_bond((0, 1), target=10.0, comparator='lt')
        cons.fix_dihedral((0, 1, 2, 3),
                          target=np.degrees(current) + 350.0)

        cons.disable_satisfied_inequalities()

        residual = cons.residual()
        expected = (
            (current - (current + np.deg2rad(350.0)) + np.pi)
            % (2 * np.pi) - np.pi
        )
        assert cons._active_mask == [False, True]
        np.testing.assert_allclose(residual, [expected], atol=1e-12)


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


def test_sella_reuses_matching_restricted_step_spectrum():
    basis = np.eye(3)

    class FakeHessian:
        B = np.eye(3)
        _B_gpu = None
        evals = np.array([-0.5, 1.0, 2.0])

    class FakePES:
        H = FakeHessian()

        def get_Unred(self):
            return basis

        def get_HL_projected(self, _):
            raise AssertionError("matching PRFO spectrum should be reused")

    opt = object.__new__(Sella)
    opt.pes = FakePES()
    opt.eig = True
    opt.ord = 1
    opt.nsteps_since_diag = 3
    opt.nsteps_per_diag = 3
    opt.diag_every_n = np.inf
    opt._last_step_basis = basis
    opt._last_step_eigenvalues = np.array([-0.5, 1.0, 2.0])

    assert not opt._should_diagonalize()
    opt._last_step_eigenvalues[0] = 0.5
    assert opt._should_diagonalize()

    opt._last_step_basis = basis.copy()
    with pytest.raises(AssertionError, match="should be reused"):
        opt._should_diagonalize()


class TestLinearMolecule:
    """Test that linear molecules (e.g. N2) don't produce NaN in rotation
    constraints.

    Linear molecules have degenerate eigenvalues in the quaternion-based
    rotation parameterization, which previously caused NaN in second
    derivatives and zeroed Jacobians due to jnp.sign(0)==0.
    """

    def test_n2_cartesian(self):
        """Test N2 optimization in Cartesian coordinates."""
        atoms = molecule('N2')
        atoms.calc = EMT()
        opt = Sella(atoms, order=0, logfile=None)
        opt.run(fmax=0.01, steps=100)
        assert opt.converged()

    def test_n2_internal(self):
        """Test N2 optimization with internal coordinates (TRICs)."""
        atoms = molecule('N2')
        atoms.calc = EMT()
        opt = Sella(atoms, order=0, internal=True, logfile=None)
        opt.run(fmax=0.01, steps=100)
        assert opt.converged()
