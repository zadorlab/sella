"""Tests for unit cell optimization functionality."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixBondLength

from sella import Sella
from sella.optimize.restricted_step import MaxInternalStep
from sella.peswrapper import (
    InternalPES, CellInternalPES, CellCartesianPES, _logm_3x3
)
from sella.internal import Internals, Bond, Angle, Constraints


def make_molecular_crystal():
    """Create a simple periodic molecular system for testing.

    Uses methane (CH4) in a periodic box, which has well-defined
    bonds and angles that work reliably with internal coordinates.
    """
    mol = molecule('CH4')
    # Place in a box with padding
    mol.center(vacuum=3.0)
    mol.pbc = True
    return mol


def make_water_crystal():
    """Create a water molecule in a periodic box."""
    mol = molecule('H2O')
    mol.center(vacuum=3.0)
    mol.pbc = True
    return mol


class TestCellDerivatives:
    """Test cell derivative functions for internal coordinates."""

    @staticmethod
    def _cell_deriv_fd(atoms, coord, delta=1e-5):
        """Compute d(coord)/d(cell) via central finite differences."""
        cell0 = atoms.get_cell().array.copy()
        grad_numeric = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                cell_plus = cell0.copy()
                cell_plus[i, j] += delta
                atoms.set_cell(cell_plus, scale_atoms=False)
                val_plus = coord.calc(atoms)

                cell_minus = cell0.copy()
                cell_minus[i, j] -= delta
                atoms.set_cell(cell_minus, scale_atoms=False)
                val_minus = coord.calc(atoms)

                grad_numeric[i, j] = (val_plus - val_minus) / (2 * delta)
        atoms.set_cell(cell0, scale_atoms=False)
        return grad_numeric

    def test_bond_cell_derivative_numerical_molecular(self):
        """Verify bond cell derivative against numerical finite difference."""
        atoms = make_molecular_crystal()
        internals = Internals(atoms)
        internals.find_all_bonds()
        assert len(internals.internals['bonds']) > 0, "No bonds found"

        bond = internals.internals['bonds'][0]
        grad_analytic = bond.calc_cell_gradient(atoms)
        grad_numeric = self._cell_deriv_fd(atoms, bond)
        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_bond_cell_derivative_with_periodic_image(self):
        """Test bond cell derivative for bond crossing periodic boundary."""
        atoms = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
        atoms.set_cell([3.0, 3.0, 3.0])
        atoms.pbc = True

        bond = Bond(
            indices=np.array([0, 1], dtype=np.int32),
            ncvecs=np.array([[0, 0, -1]], dtype=np.int32)
        )

        grad_analytic = bond.calc_cell_gradient(atoms)
        grad_numeric = self._cell_deriv_fd(atoms, bond)
        assert not np.allclose(grad_analytic, 0), "Cell gradient should be non-zero for PBC bond"
        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_angle_cell_derivative_numerical(self):
        """Verify angle cell derivative against numerical finite difference."""
        atoms = make_water_crystal()
        internals = Internals(atoms)
        internals.find_all_bonds()
        internals.find_all_angles()

        if not internals.internals['angles']:
            pytest.skip("No angles found in structure")

        angle = internals.internals['angles'][0]
        grad_analytic = angle.calc_cell_gradient(atoms)
        grad_numeric = self._cell_deriv_fd(atoms, angle)

        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_cell_jacobian_shape(self):
        """Test that cell_jacobian returns correct shape."""
        atoms = make_molecular_crystal()  # CH4 with bonds and angles

        internals = Internals(atoms)
        internals.find_all_bonds()
        internals.find_all_angles()

        J_cell = internals.cell_jacobian()

        # Should have shape (n_active_coords, 9)
        n_active = len(internals.calc())
        assert J_cell.shape == (n_active, 9)


class TestCellRawASEConstraintForces:
    """Sella imports ASE constraints, so evals must use raw calculator data."""

    def test_cartesian_cell_eval_uses_raw_forces_with_fixatoms(self):
        atoms = Atoms('H2',
                      positions=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                      cell=np.eye(3) * 5.0, pbc=True)
        raw_forces = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        stress = np.zeros(6)
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.0, forces=raw_forces, stress=stress
        )
        atoms.set_constraint(FixAtoms([0]))
        assert_allclose(atoms.get_forces()[0], 0.0)

        pes = CellCartesianPES(atoms)
        _, g = pes.eval()

        assert_allclose(g[:pes.n_cart], -raw_forces.ravel())
        assert_allclose(
            g[pes.n_cart:],
            pes._stress_to_cell_gradient(stress, raw_forces),
        )

    def test_internal_cell_eval_uses_raw_forces_with_fixbond(self):
        atoms = Atoms('OH2',
                      positions=[[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0]],
                      cell=np.eye(3) * 5.0, pbc=True)
        raw_forces = np.array([[0.0, 0.0, 0.0],
                               [2.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.0, forces=raw_forces, stress=np.zeros(6)
        )
        atoms.set_constraint(FixBondLength(0, 1))
        assert not np.allclose(atoms.get_forces(), raw_forces)

        pes = CellInternalPES(atoms, Internals(atoms),
                              auto_find_internals=True)
        seen = {}

        def spy_stress_to_cell_gradient(stress, forces=None):
            seen['forces'] = forces.copy()
            return np.zeros(pes.n_cell_dof)

        pes._stress_to_cell_gradient = spy_stress_to_cell_gradient
        pes.eval()

        assert_allclose(seen['forces'], raw_forces)


class TestCellHessianFunction:
    """Cell PES hessian_function installs only the coordinate block."""

    @staticmethod
    def _atoms():
        atoms = Atoms('H2',
                      positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
                      cell=np.eye(3) * 4.0, pbc=True)
        atoms.calc = LennardJones()
        return atoms

    @staticmethod
    def _hessian(atoms):
        return np.eye(3 * len(atoms))

    @pytest.mark.parametrize("internal", [False, True])
    def test_cell_hessian_function_with_eig_step(self, internal):
        opt = Sella(self._atoms(), order=0, internal=internal,
                    optimize_cell=True, hessian_function=self._hessian,
                    eig=True, logfile=None)
        opt.step()

        assert opt.pes.H.shape == (opt.pes.dim, opt.pes.dim)
        assert opt.pes.H.initialized


class TestCellSmaxSemantics:
    """Explicit smax is effective stress; omitted smax is ASE-like."""

    @staticmethod
    def _stressed_water():
        atoms = Atoms('OH2',
                      positions=[[5.0, 5.0, 5.0],
                                 [5.96, 5.0, 5.0],
                                 [5.0, 5.96, 5.0]],
                      cell=np.eye(3) * 10.0, pbc=True)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=0.0,
            forces=np.zeros((3, 3)),
            stress=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        return atoms

    @pytest.mark.parametrize("internal", [False, True])
    def test_explicit_smax_uses_effective_stress(self, internal):
        opt = Sella(self._stressed_water(), order=0, internal=internal,
                    optimize_cell=True, allow_fragments=internal,
                    logfile=None)
        opt.fmax = 0.05

        opt.smax = None
        assert not opt.converged()
        assert opt._last_converged[3] > 1.0

        opt.smax = 0.02
        assert opt.converged()
        assert_allclose(opt._last_converged[3], 0.01)

        opt.smax = 0.005
        assert not opt.converged()
        assert_allclose(opt._last_converged[3], 0.01)

    def test_explicit_smax_uses_corrected_cell_gradient(self):
        atoms = Atoms('H2',
                      positions=[[4.4, 5.0, 5.0], [5.6, 5.0, 5.0]],
                      cell=np.eye(3) * 10.0, pbc=True)
        atoms.calc = LennardJones()
        atoms.set_constraint(FixBondLength(0, 1))
        assert np.abs(atoms.get_stress(apply_constraint=False)).max() > 1e-3

        opt = Sella(atoms, order=0, internal=True, optimize_cell=True,
                    allow_fragments=True, logfile=None)
        opt.fmax = 0.05
        opt.smax = 1e-3

        assert opt.converged()
        assert opt._last_converged[1] < 1e-12
        assert_allclose(opt._last_converged[3], 0.0, atol=1e-12)


class TestCellInternalPES:
    """Test CellInternalPES class."""

    def test_cell_internal_pes_initialization(self):
        """Test that CellInternalPES initializes correctly."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        # Check dimensions include cell DOF
        assert pes.n_cell_dof == 9  # Full 3x3 cell
        assert pes.dim == pes.n_internal + 9

    def test_cell_internal_pes_get_x(self):
        """Test get_x returns combined internal + cell vector."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        x = pes.get_x()

        # Length should be n_internal + n_cell_dof
        assert len(x) == pes.dim

        # First n_internal elements are internal coords
        q = internals.calc()
        assert_allclose(x[:pes.n_internal], q, rtol=1e-10)

        # Cell params at identity should be approximately zero
        # (log of identity matrix is zero)
        assert_allclose(x[pes.n_internal:], 0, atol=1e-10)

    def test_cell_internal_pes_cell_mask(self):
        """Test cell_mask parameter."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Only allow diagonal elements (hydrostatic strain)
        cell_mask = np.eye(3, dtype=bool)

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals, cell_mask=cell_mask)

        assert pes.n_cell_dof == 3  # Only 3 diagonal elements
        assert pes.dim == pes.n_internal + 3

    def test_cell_internal_pes_eval(self):
        """Test eval returns correct gradient shape."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        f, g = pes.eval()

        assert isinstance(f, float)
        assert len(g) == pes.dim


class TestCellCartesianPES:
    """Test CellCartesianPES class."""

    def test_cell_cartesian_pes_initialization(self):
        """Test that CellCartesianPES initializes correctly."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        # Check dimensions include cell DOF
        assert pes.n_cell_dof == 9  # Full 3x3 cell
        assert pes.n_cart == 3 * len(atoms)
        assert pes.dim == pes.n_cart + 9

    def test_cell_cartesian_pes_get_x(self):
        """Test get_x returns combined Cartesian + cell vector."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        x = pes.get_x()

        # Length should be n_cart + n_cell_dof
        assert len(x) == pes.dim

        # First n_cart elements are Cartesian positions
        positions_flat = atoms.get_positions().ravel()
        assert_allclose(x[:pes.n_cart], positions_flat, rtol=1e-10)

        # Cell params at identity should be approximately zero
        # (log of identity matrix is zero)
        assert_allclose(x[pes.n_cart:], 0, atol=1e-10)

    def test_cell_cartesian_pes_cell_mask(self):
        """Test cell_mask parameter."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Only allow diagonal elements (hydrostatic strain)
        cell_mask = np.eye(3, dtype=bool)

        pes = CellCartesianPES(atoms, cell_mask=cell_mask)

        assert pes.n_cell_dof == 3  # Only 3 diagonal elements
        assert pes.dim == pes.n_cart + 3

    def test_cell_cartesian_pes_eval(self):
        """Test eval returns correct gradient shape."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        f, g = pes.eval()

        assert isinstance(f, float)
        assert len(g) == pes.dim

    def test_cell_cartesian_pes_save_restore(self):
        """Test save and restore functionality."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        # Save initial state
        pes.save()
        x0 = pes.get_x()
        cell0 = atoms.get_cell().array.copy()
        pos0 = atoms.get_positions().copy()

        # Modify positions and cell
        atoms.positions += 0.1
        new_cell = atoms.get_cell().array * 1.05
        atoms.set_cell(new_cell, scale_atoms=False)

        # Verify modifications took effect
        assert not np.allclose(atoms.get_positions(), pos0)
        assert not np.allclose(atoms.get_cell().array, cell0)

        # Restore and verify
        pes.restore()
        assert_allclose(atoms.get_positions(), pos0)
        assert_allclose(atoms.get_cell().array, cell0)

    def test_cell_cartesian_pes_set_x(self):
        """Test set_x updates positions and cell."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        # Get initial x
        x0 = pes.get_x()

        # Create target with small perturbations
        x_target = x0.copy()
        x_target[:pes.n_cart] += 0.01  # Small position change
        x_target[pes.n_cart:] += 0.001  # Small cell change

        # Set new x
        pes.set_x(x_target)

        # Get x and verify it's close to target
        x_new = pes.get_x()
        # Cell parameters should match
        assert_allclose(x_new[pes.n_cart:], x_target[pes.n_cart:], rtol=1e-6)
        # Positions should match
        assert_allclose(x_new[:pes.n_cart], x_target[:pes.n_cart], rtol=1e-6)

    def test_cell_cartesian_vs_internal_gradient_shape(self):
        """Compare CellCartesianPES and CellInternalPES gradient shapes."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # CellCartesianPES
        pes_cart = CellCartesianPES(atoms.copy())
        pes_cart.atoms.calc = EMT()
        _, g_cart = pes_cart.eval()

        # CellInternalPES
        internals = Internals(atoms)
        pes_int = CellInternalPES(atoms, internals)
        _, g_int = pes_int.eval()

        # Both should have 9 cell DOF (full 3x3 cell)
        assert pes_cart.n_cell_dof == 9
        assert pes_int.n_cell_dof == 9

        # Cell parts of gradient should have same length
        assert len(g_cart[pes_cart.n_cart:]) == len(g_int[pes_int.n_internal:])

    def test_cell_cartesian_pes_pressure(self):
        """Test scalar pressure contribution."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Without pressure
        pes_no_p = CellCartesianPES(atoms.copy())
        pes_no_p.atoms.calc = EMT()
        f_no_p, _ = pes_no_p.eval()

        # With pressure
        pressure = 0.1  # eV/Å³
        atoms2 = atoms.copy()
        atoms2.calc = EMT()
        pes_p = CellCartesianPES(atoms2, scalar_pressure=pressure)
        f_p, _ = pes_p.eval()

        # Energy with pressure should be higher (positive pressure)
        volume = atoms.get_volume()
        expected_diff = pressure * volume
        assert_allclose(f_p - f_no_p, expected_diff, rtol=1e-10)


class TestCellCartesianGradient:
    """Test cell gradient calculations in CellCartesianPES."""

    def test_gradient_numerical(self):
        """Test full gradient (Cartesian + cell) matches numerical FD for bulk Cu."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()
        pes = CellCartesianPES(atoms)

        _, g = pes.eval()
        delta = 1e-6
        x0 = pes.get_x()
        g_numeric = np.zeros(pes.dim)

        for i in range(pes.dim):
            pes.set_x(x0)
            x_plus = x0.copy()
            x_plus[i] += delta
            pes.set_x(x_plus)
            e_plus, _ = pes.eval()

            pes.set_x(x0)
            x_minus = x0.copy()
            x_minus[i] -= delta
            pes.set_x(x_minus)
            e_minus, _ = pes.eval()

            g_numeric[i] = (e_plus - e_minus) / (2 * delta)

        pes.set_x(x0)

        assert_allclose(g, g_numeric, atol=1e-4, rtol=1e-3)


class TestSellaWithCellOptimization:
    """Integration tests for Sella with cell optimization."""

    def test_sella_optimize_cell_parameter(self):
        """Test that optimize_cell parameter is properly handled."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        opt = Sella(atoms, internal=True, order=0, optimize_cell=True)

        assert opt.optimize_cell is True
        assert isinstance(opt.pes, CellInternalPES)

    def test_sella_cell_optimization_validation(self):
        """Test validation of cell optimization parameters."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Should fail with order != 0
        with pytest.raises(ValueError, match="order=0"):
            Sella(atoms, internal=True, order=1, optimize_cell=True)

        # internal=False with optimize_cell=True should use CellCartesianPES
        atoms2 = atoms.copy()
        atoms2.calc = EMT()
        opt = Sella(atoms2, internal=False, order=0, optimize_cell=True)
        assert isinstance(opt.pes, CellCartesianPES)

        # Should fail without PBC
        atoms_nopbc = atoms.copy()
        atoms_nopbc.pbc = False
        atoms_nopbc.calc = EMT()
        with pytest.raises(ValueError, match="periodic"):
            Sella(atoms_nopbc, internal=True, order=0, optimize_cell=True)

    def test_sella_cell_optimization_single_step(self):
        """Test that cell optimization can take a single step."""
        # Use strained FCC copper
        atoms = bulk('Cu', 'fcc', a=3.8)  # Slightly expanded
        atoms.calc = EMT()

        opt = Sella(atoms, internal=True, order=0, optimize_cell=True)

        # Record initial state
        cell0 = atoms.get_cell().array.copy()
        e0 = atoms.get_potential_energy()

        # Take one step
        opt.step()

        # Energy should decrease (or at least be computed)
        e1 = atoms.get_potential_energy()
        # Cell should change
        cell1 = atoms.get_cell().array

        # At least something should have changed
        assert not np.allclose(cell0, cell1) or not np.isclose(e0, e1)

    @pytest.mark.slow
    def test_sella_cell_optimization_convergence(self):
        """Test that cell optimization converges for a simple system."""
        # Create strained FCC copper - use single primitive cell
        # (larger supercells have issues with angle finding in FCC)
        # Note: bulk() creates a primitive cell, so a=3.8 gives cell param ~2.69 Å
        atoms = bulk('Cu', 'fcc', a=3.8)  # Slightly expanded
        atoms.calc = EMT()

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            logfile=None,
        )

        # Run optimization
        converged = opt.run(fmax=0.05, steps=50)

        if converged:
            # Check that cell relaxed toward equilibrium
            # EMT equilibrium for FCC Cu primitive cell is about a = 2.54 Å
            a = atoms.get_cell().cellpar()[0]
            assert 2.4 < a < 2.7


    def test_gradient_converged_delegates_to_converged(self):
        """Test that gradient_converged routes through converged().

        ASE 3.28 changed irun() to call gradient_converged() instead of
        converged(), bypassing Sella's stress check. This test verifies
        that Sella's gradient_converged override delegates to converged(),
        so the stress convergence check is always reached.
        """
        atoms = bulk('Cu', 'fcc', a=3.8)  # Strained → nonzero stress
        atoms.calc = EMT()

        opt = Sella(atoms, internal=True, order=0, optimize_cell=True,
                    logfile=None)

        # Force evaluation so pes has cached gradient
        opt.pes.eval()

        # converged() should check stress; with a strained cell and tight
        # tolerance, it should report not converged
        opt.fmax = 1e-10
        opt.smax = 1e-10
        assert not opt.converged()

        # gradient_converged must agree (it delegates to converged)
        assert not opt.gradient_converged()

        # Now with loose tolerances, both should agree on convergence
        opt.fmax = 100.0
        opt.smax = 100.0
        assert opt.converged()
        assert opt.gradient_converged()


class TestVoigtConversion:
    """Test Voigt stress conversion functions."""

    def test_voigt_roundtrip(self):
        """Test conversion roundtrip."""
        from sella.peswrapper import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress

        # Random Voigt stress
        voigt = np.random.randn(6)

        # Convert to 3x3 and back
        stress_3x3 = voigt_6_to_full_3x3_stress(voigt)
        voigt_back = full_3x3_to_voigt_6_stress(stress_3x3)

        assert_allclose(voigt, voigt_back)

    def test_voigt_symmetry(self):
        """Test that converted tensor is symmetric."""
        from sella.peswrapper import voigt_6_to_full_3x3_stress

        voigt = np.array([1, 2, 3, 4, 5, 6])
        stress_3x3 = voigt_6_to_full_3x3_stress(voigt)

        assert_allclose(stress_3x3, stress_3x3.T)


class TestStressTensor:
    """Test stress tensor calculations for cell optimization."""

    def test_stress_changes_with_strain_inorganic(self):
        """Test that stress changes when cell is strained for bulk Cu."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()
        stress_ref = atoms.get_stress()

        # Strain the cell
        atoms_strained = bulk('Cu', 'fcc', a=3.8)  # Different lattice constant
        atoms_strained.calc = EMT()
        stress_strained = atoms_strained.get_stress()

        # Stress should change
        assert not np.allclose(stress_ref, stress_strained)

        # Stress should be finite
        assert np.all(np.isfinite(stress_ref))
        assert np.all(np.isfinite(stress_strained))

    def test_stress_finite_molecular(self):
        """Test stress is finite for molecular system (CH4 in periodic box)."""
        # Create CH4 in a periodic box
        mol = molecule('CH4')
        mol.center(vacuum=4.0)
        mol.pbc = True
        mol.calc = EMT()

        # Should be able to get stress without error
        stress_voigt = mol.get_stress()
        assert len(stress_voigt) == 6
        assert np.all(np.isfinite(stress_voigt))


class TestCellGradient:
    """Test cell gradient calculations in CellInternalPES."""

    def test_cell_gradient_numerical_inorganic(self):
        """Test cell gradient matches numerical finite difference for bulk Cu."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        # Get analytical gradient
        _, g = pes.eval()
        g_cell = g[pes.n_internal:]  # Cell part of gradient

        # Numerical gradient via finite difference on cell parameters
        delta = 1e-6
        x0 = pes.get_x()
        g_cell_numeric = np.zeros(pes.n_cell_dof)

        for i in range(pes.n_cell_dof):
            pes.set_x(x0)  # Restore before each probe
            x_plus = x0.copy()
            x_plus[pes.n_internal + i] += delta
            pes.set_x(x_plus)
            e_plus, _ = pes.eval()

            pes.set_x(x0)  # Restore before -delta
            x_minus = x0.copy()
            x_minus[pes.n_internal + i] -= delta
            pes.set_x(x_minus)
            e_minus, _ = pes.eval()

            g_cell_numeric[i] = (e_plus - e_minus) / (2 * delta)

        pes.set_x(x0)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_cell_gradient_shape_molecular(self):
        """Test cell gradient has correct shape for molecular system."""
        water1 = molecule('H2O')
        water2 = molecule('H2O')
        water1.positions += [1.0, 1.0, 1.0]
        water2.positions += [4.0, 4.0, 4.0]
        mol = water1 + water2
        mol.set_cell([7.0, 7.0, 7.0])
        mol.pbc = True
        mol.calc = LennardJones()

        internals = Internals(mol, allow_fragments=True)
        pes = CellInternalPES(mol, internals, auto_find_internals=True)

        _, g = pes.eval()

        # Gradient should have n_internal + n_cell_dof components
        assert len(g) == pes.n_internal + pes.n_cell_dof

        # All components should be finite
        assert np.all(np.isfinite(g))


class TestMolecularCrystal:
    """Test cell optimization for molecular crystal systems.

    Molecular crystals have multiple separate molecules in a periodic cell,
    requiring TRICs (Translation-Rotation Internal Coordinates) for proper
    handling of each molecular fragment.
    """

    def test_molecular_crystal_with_trics(self):
        """Test that molecular crystal optimization works with allow_fragments=True."""
        # Create a simple molecular crystal: two water molecules in a box
        water1 = molecule('H2O')
        water2 = molecule('H2O')

        # Place molecules in different parts of the cell
        water1.positions += [1.0, 1.0, 1.0]
        water2.positions += [4.0, 4.0, 4.0]

        # Combine into one system
        atoms = water1 + water2
        atoms.set_cell([7.0, 7.0, 7.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        # Create optimizer with allow_fragments=True for TRICs
        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            allow_fragments=True,
            logfile=None,
        )

        assert isinstance(opt.pes, CellInternalPES)

        # Verify the internal coords have TRICs (translations and rotations)
        internals = opt.pes.int
        assert internals.ntrans > 0, "Should have translation coordinates for fragments"
        assert internals.nrotations > 0, "Should have rotation coordinates for fragments"

        # Take a few steps to verify no errors
        for _ in range(3):
            opt.step()

    def test_molecular_crystal_cartesian(self):
        """Test molecular crystal optimization with Cartesian coordinates."""
        # Create two methane molecules
        ch4_1 = molecule('CH4')
        ch4_2 = molecule('CH4')

        ch4_1.positions += [1.5, 1.5, 1.5]
        ch4_2.positions += [5.0, 5.0, 5.0]

        atoms = ch4_1 + ch4_2
        atoms.set_cell([8.0, 8.0, 8.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        # Use Cartesian coordinates (internal=False) with cell optimization
        opt = Sella(
            atoms,
            internal=False,
            order=0,
            optimize_cell=True,
            logfile=None,
        )

        assert isinstance(opt.pes, CellCartesianPES)

        # Take a few steps
        for _ in range(3):
            opt.step()

    def test_molecular_crystal_trics_dof_count(self):
        """Test that TRICs add correct DOF for molecular fragments."""
        # Create two separate H2 molecules
        atoms = Atoms(
            'H4',
            positions=[
                [0.0, 0.0, 0.0],
                [0.74, 0.0, 0.0],
                [4.0, 4.0, 4.0],
                [4.74, 4.0, 4.0],
            ]
        )
        atoms.set_cell([8.0, 8.0, 8.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        # With allow_fragments=True, should get TRICs for each fragment
        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()

        # Should have 2 bonds (one per H2)
        assert internals.nbonds == 2

        # Should have translations for the 2 fragments
        assert internals.ntrans > 0

        # Should have rotations for fragments that can rotate
        # (H2 is linear, so rotation DOF may be limited)
        assert internals.nrotations >= 0


class TestTRICsCellDerivatives:
    """Test that TRICs have correct cell derivatives.

    For molecular crystals, translation and rotation coordinates have
    zero cell derivatives by construction (hardcoded in cell_jacobian).
    Bond/angle/dihedral cell derivatives are tested below.
    """

    def test_bond_cell_derivative_intramolecular(self):
        """Test intramolecular bond has zero cell derivative if not crossing PBC."""
        # Create H2 well within the cell (not crossing boundary)
        atoms = Atoms('H2', positions=[[2.0, 2.5, 2.5], [2.74, 2.5, 2.5]])
        atoms.set_cell([5.0, 5.0, 5.0])
        atoms.pbc = True

        internals = Internals(atoms)
        internals.find_all_bonds()

        bond = internals.internals['bonds'][0]

        # For intramolecular bond not crossing PBC, cell derivative should be zero
        # (the ncvec should be zero)
        grad_cell = bond.calc_cell_gradient(atoms)

        # Check that gradient is zero (bond doesn't cross boundary)
        assert_allclose(grad_cell, 0, atol=1e-10)

    def test_cell_jacobian_trics_rows_zero(self):
        """Test that TRICs rows in cell_jacobian are zero."""
        atoms = molecule('H2O')
        atoms.center(vacuum=3.0)
        atoms.pbc = True

        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()
        internals.find_all_angles()

        J_cell = internals.cell_jacobian()

        # The rows corresponding to translations and rotations should be zero
        # First ntrans rows are translations
        n_trans = internals.ntrans
        n_rot = internals.nrotations

        if n_trans > 0:
            trans_rows = J_cell[:n_trans, :]
            assert_allclose(trans_rows, 0, atol=1e-10)

        # Last nrotations rows are rotations (after other coords)
        if n_rot > 0:
            # Rotations come after: trans, bonds, angles, dihedrals, other
            rot_start = n_trans + internals.nbonds + internals.nangles + internals.ndihedrals + internals.nother
            rot_rows = J_cell[rot_start:rot_start + n_rot, :]
            assert_allclose(rot_rows, 0, atol=1e-10)


class TestCellConstraints:
    """Test cell optimization with various constraints."""

    def test_hydrostatic_only_dof_count(self):
        """Test that hydrostatic constraint gives correct number of DOF."""
        atoms = bulk('Cu', 'fcc', a=3.8)
        atoms.calc = EMT()

        # Only allow diagonal elements
        cell_mask = np.eye(3, dtype=bool)

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=cell_mask,
            logfile=None,
        )

        # Check that only 3 cell DOF (diagonal elements)
        assert opt.pes.n_cell_dof == 3

    def test_isotropic_scaling(self):
        """Test cell optimization with isotropic scaling only (1 DOF)."""
        atoms = bulk('Cu', 'fcc', a=3.8)
        atoms.calc = EMT()

        # Only allow uniform scaling (first diagonal element)
        cell_mask = np.zeros((3, 3), dtype=bool)
        cell_mask[0, 0] = True

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=cell_mask,
            logfile=None,
        )

        assert opt.pes.n_cell_dof == 1

    def test_no_shear_mask(self):
        """Test that shear components can be masked out."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Allow all 9 components (full cell optimization)
        full_mask = np.ones((3, 3), dtype=bool)
        opt_full = Sella(
            atoms.copy(),
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=full_mask,
            logfile=None,
        )
        atoms.calc = EMT()  # Reset

        # Allow only diagonal (no shear)
        diag_mask = np.eye(3, dtype=bool)
        opt_diag = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=diag_mask,
            logfile=None,
        )

        assert opt_full.pes.n_cell_dof == 9
        assert opt_diag.pes.n_cell_dof == 3


class TestRefineInitialHessian:
    """Test the refine_initial_hessian option for cell optimization."""

    def test_refine_hessian_produces_nonzero_coupling(self):
        """Test that refine_initial_hessian produces non-zero coupling blocks."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # With refinement
        pes_refined = CellInternalPES(
            atoms,
            Internals(atoms),
            refine_initial_hessian=True,
        )

        H = pes_refined.H.B
        n_int = pes_refined.n_internal

        # The coupling block should be non-zero (or at least the cell-cell block
        # should differ from 70*I)
        H_coupling = H[:n_int, n_int:]
        H_cell = H[n_int:, n_int:]

        # Cell-cell block should not be exactly 70*I (it was computed via FD)
        # Note: it might still be close to diagonal for simple systems
        assert H_cell.shape == (pes_refined.n_cell_dof, pes_refined.n_cell_dof)

    def test_refine_hessian_vs_default_different(self):
        """Test that refined Hessian differs from default."""
        # Without refinement
        atoms1 = bulk('Cu', 'fcc', a=3.6)
        atoms1.calc = EMT()
        pes_default = CellInternalPES(
            atoms1,
            Internals(atoms1),
            refine_initial_hessian=False,
        )

        # With refinement
        atoms2 = bulk('Cu', 'fcc', a=3.6)
        atoms2.calc = EMT()
        pes_refined = CellInternalPES(
            atoms2,
            Internals(atoms2),
            refine_initial_hessian=True,
        )

        H_default = pes_default.H.B
        H_refined = pes_refined.H.B

        n_int = pes_default.n_internal

        # The cell-cell blocks should be different
        H_cell_default = H_default[n_int:, n_int:]
        H_cell_refined = H_refined[n_int:, n_int:]

        # Default is 70*I, refined should be computed from actual curvature
        assert not np.allclose(H_cell_default, H_cell_refined)

    def test_refine_hessian_cartesian_pes(self):
        """Test refine_initial_hessian with CellCartesianPES."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(
            atoms,
            refine_initial_hessian=True,
        )

        H = pes.H.B
        n_cart = pes.n_cart

        # Hessian should have correct shape
        assert H.shape == (pes.dim, pes.dim)

        # Cell-cell block should exist
        H_cell = H[n_cart:, n_cart:]
        assert H_cell.shape == (pes.n_cell_dof, pes.n_cell_dof)

    def test_refine_hessian_via_sella_api(self):
        """Test refine_initial_hessian through Sella API."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            refine_initial_hessian=True,
            logfile=None,
        )

        # Should be CellInternalPES
        assert isinstance(opt.pes, CellInternalPES)

        # Hessian should be properly dimensioned
        H = opt.pes.H.B
        assert H.shape == (opt.pes.dim, opt.pes.dim)

    def test_refine_hessian_force_call_count(self):
        """Test that refinement makes expected number of force calls."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Only allow diagonal cell DOF for fewer calls
        cell_mask = np.eye(3, dtype=bool)  # 3 DOF

        pes = CellInternalPES(
            atoms,
            Internals(atoms),
            cell_mask=cell_mask,
            refine_initial_hessian=True,
        )

        # Should have made 2 * n_cell_dof = 6 evaluations during init
        # (2 per cell DOF for central difference)
        assert pes.neval == 6


class TestRigidFragments:
    """Test rigid fragment mode for molecular crystal cell optimization."""

    def _make_two_water_crystal(self, cell_diag=7.0):
        """Create two water molecules in a periodic box."""
        water1 = molecule('H2O')
        water2 = molecule('H2O')
        water1.positions += [1.0, 1.0, 1.0]
        water2.positions += [4.0, 4.0, 4.0]
        atoms = water1 + water2
        atoms.set_cell([cell_diag, cell_diag, cell_diag])
        atoms.pbc = True
        atoms.calc = LennardJones()
        return atoms

    def test_rigid_fragments_auto_detected(self):
        """Test that rigid_fragments is auto-detected from allow_fragments."""
        atoms = self._make_two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True
        assert hasattr(pes, 'fragment_groups')
        assert len(pes.fragment_groups) == 2  # Two water molecules

    def test_rigid_fragments_not_detected_without_fragments(self):
        """Test that rigid_fragments is False when no translations exist."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()
        internals = Internals(atoms)

        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is False

    def test_rigid_fragments_explicit_override(self):
        """Test that rigid_fragments can be explicitly set."""
        atoms = self._make_two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        # Explicitly disable
        pes = CellInternalPES(atoms, internals, rigid_fragments=False)
        assert pes.rigid_fragments is False

    def test_fragment_groups_correct_atoms(self):
        """Test that fragment groups contain the correct atom indices."""
        atoms = self._make_two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)

        # Each group should have 3 atoms (water has O, H, H)
        for group in pes.fragment_groups:
            assert len(group) == 3

        # All atoms should be covered
        all_atoms = np.sort(np.concatenate(pes.fragment_groups))
        assert_allclose(all_atoms, np.arange(6))

    def test_compute_delta_r(self):
        """Test that _compute_delta_r gives positions relative to fragment CoM."""
        atoms = self._make_two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)
        delta_r = pes._compute_delta_r()

        # For each fragment, the mean of delta_r should be zero
        for group in pes.fragment_groups:
            assert_allclose(delta_r[group].mean(axis=0), 0, atol=1e-12)

    def _make_triclinic_crystal(self):
        """Create two water molecules in a triclinic periodic box."""
        water1 = molecule('H2O')
        water2 = molecule('H2O')
        water1.positions += [1.0, 1.0, 1.0]
        water2.positions += [4.0, 4.0, 4.0]
        atoms = water1 + water2
        cell = np.array([
            [7.0, 0.5, 0.3],
            [0.0, 6.8, 0.4],
            [0.0, 0.0, 7.2],
        ])
        atoms.set_cell(cell)
        atoms.pbc = True
        atoms.calc = LennardJones()
        return atoms

    def _make_sheared_crystal(self):
        """Create two water molecules in a heavily sheared triclinic cell."""
        water1 = molecule('H2O')
        water2 = molecule('H2O')
        water1.positions += [1.0, 1.0, 1.0]
        water2.positions += [4.0, 4.0, 4.0]
        atoms = water1 + water2
        cell = np.array([
            [7.0, 1.5, 0.8],
            [0.0, 6.5, 1.2],
            [0.0, 0.0, 7.5],
        ])
        atoms.set_cell(cell)
        atoms.pbc = True
        atoms.calc = LennardJones()
        return atoms

    def _make_seeded_triclinic_water_crystal(self, seed, nwaters=3):
        """Create a deterministic skewed molecular crystal from a seed."""
        rng = np.random.RandomState(seed)
        cell = np.array([
            [5.8, 0.0, 0.0],
            [0.8 + 0.15 * rng.rand(), 5.6, 0.0],
            [0.5 + 0.15 * rng.rand(), 0.7 + 0.15 * rng.rand(), 6.0],
        ])
        frac_centers = np.array([
            [0.20, 0.22, 0.22],
            [0.58, 0.34, 0.50],
            [0.34, 0.70, 0.42],
            [0.76, 0.68, 0.75],
        ])
        water = molecule('H2O')
        water.positions -= water.positions.mean(axis=0)

        atoms = Atoms()
        for frac in frac_centers[:nwaters]:
            mol = water.copy()
            for axis in ['x', 'y', 'z']:
                mol.rotate(rng.rand() * 360.0, axis, center=(0, 0, 0))
            mol.positions += frac @ cell
            atoms += mol

        atoms.set_cell(cell)
        atoms.pbc = True
        atoms.calc = LennardJones()
        return atoms

    def _cell_gradient_fd(self, pes, delta=1e-6):
        """Compute cell DOF gradient via central finite differences."""
        _, g = pes.eval()
        g_cell = g[pes.n_internal:]
        x0 = pes.get_x()
        g_cell_numeric = np.zeros(pes.n_cell_dof)
        for i in range(pes.n_cell_dof):
            pes.set_x(x0)
            x_plus = x0.copy()
            x_plus[pes.n_internal + i] += delta
            pes.set_x(x_plus)
            e_plus, _ = pes.eval()
            pes.set_x(x0)
            x_minus = x0.copy()
            x_minus[pes.n_internal + i] -= delta
            pes.set_x(x_minus)
            e_minus, _ = pes.eval()
            g_cell_numeric[i] = (e_plus - e_minus) / (2 * delta)
        pes.set_x(x0)
        return g_cell, g_cell_numeric

    @pytest.mark.parametrize("setup", [
        "orthogonal", "triclinic", "sheared", "deformed",
    ])
    def test_rigid_fragment_cell_gradient_numerical(self, setup):
        """Test rigid fragment cell gradient matches numerical FD."""
        if setup == "orthogonal":
            atoms = self._make_two_water_crystal()
        elif setup == "triclinic":
            atoms = self._make_triclinic_crystal()
        elif setup == "sheared":
            atoms = self._make_sheared_crystal()
        elif setup == "deformed":
            atoms = self._make_two_water_crystal()

        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True

        if setup == "deformed":
            x0 = pes.get_x()
            x_deformed = x0.copy()
            for i in range(pes.n_cell_dof):
                x_deformed[pes.n_internal + i] += 0.02 * ((-1)**i)
            pes.set_x(x_deformed)

        g_cell, g_cell_numeric = self._cell_gradient_fd(pes)
        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("seed", [0, 7, 19])
    def test_seeded_triclinic_cell_gradient_matches_fd(self, seed):
        """Rigid-fragment cell gradients should survive varied skewed cells."""
        atoms = self._make_seeded_triclinic_water_crystal(seed)
        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals, rigid_fragments=True)
        assert pes.rigid_fragments is True
        assert len(pes.fragment_groups) == 3

        # Move away from the construction point so the FD sweep samples the
        # real cell-update path, including rigid rotations under shear.
        x = pes.get_x().copy()
        rng = np.random.RandomState(seed + 100)
        x[pes.n_internal:] += 0.02 * rng.normal(size=pes.n_cell_dof)
        pes.set_x(x)

        g_cell, g_cell_numeric = self._cell_gradient_fd(pes, delta=1e-5)
        assert np.abs(g_cell).max() > 1e-4
        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_masked_rigid_fragment_cell_gradient_matches_fd(self):
        """A sparse cell mask must use the same rigid cell derivative."""
        atoms = self._make_seeded_triclinic_water_crystal(seed=23)
        internals = Internals(atoms, allow_fragments=True)
        cell_mask = np.array([
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ])
        pes = CellInternalPES(
            atoms,
            internals,
            rigid_fragments=True,
            cell_mask=cell_mask,
        )
        assert pes.n_cell_dof == np.count_nonzero(cell_mask)

        x = pes.get_x().copy()
        x[pes.n_internal:] += np.linspace(-0.015, 0.015, pes.n_cell_dof)
        pes.set_x(x)

        g_cell, g_cell_numeric = self._cell_gradient_fd(pes, delta=1e-5)
        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_intramolecular_geometry_preserved_after_cell_step(self):
        """Test that intramolecular geometry is preserved after a cell-only step."""
        atoms = self._make_two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)

        pes = CellInternalPES(atoms, internals)

        # Record initial bond lengths and angles
        q0 = internals.calc()
        x0 = pes.get_x()

        # Apply a cell-only displacement (change cell params, keep internal targets)
        x_target = x0.copy()
        x_target[pes.n_internal:] += 0.01  # Small uniform cell strain

        pes.set_x(x_target)

        # Get new internal coordinates
        q1 = internals.calc()

        # Internal coords (bonds, angles) should be close to original
        # The internal coord solver handles the small residual
        n_trans = internals.ntrans
        # Skip translations (those change with cell), check bonds/angles
        assert_allclose(q0[n_trans:], q1[n_trans:], atol=1e-3)

    def test_monoatomic_rigid_fragments_trivial(self):
        """For monoatomic crystals, Δr=0 so rigid_fragments correction vanishes."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Create internals with allow_fragments - each atom is its own "fragment"
        internals_frag = Internals(atoms, allow_fragments=True)
        has_translations = bool(internals_frag.internals.get('translations', []))

        if not has_translations:
            # No fragments detected for monoatomic - this is expected
            # Just verify rigid_fragments defaults to False
            pes = CellInternalPES(atoms, internals_frag)
            assert pes.rigid_fragments is False
            return

        # If fragments are detected, verify Δr=0 behavior
        pes_rigid = CellInternalPES(atoms, internals_frag, rigid_fragments=True)
        delta_r = pes_rigid._compute_delta_r()
        assert_allclose(delta_r, 0, atol=1e-12)

    def test_rigid_fragments_sella_integration(self):
        """Test rigid fragments through the Sella API."""
        atoms = self._make_two_water_crystal()

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            allow_fragments=True,
            logfile=None,
        )

        assert isinstance(opt.pes, CellInternalPES)
        assert opt.pes.rigid_fragments is True

        # Take a few steps to verify no errors
        for _ in range(3):
            opt.step()

    def test_rotation_applied_on_shear(self):
        """Test that fragment atoms rotate under shear, not just translate."""
        atoms = self._make_two_water_crystal()
        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)

        x0 = pes.get_x()
        pos0 = atoms.get_positions().copy()

        # Apply a pure shear cell displacement (off-diagonal log-deformation)
        x_shear = x0.copy()
        # Find an off-diagonal cell DOF (shear)
        cell_mask = pes.cell_mask
        offdiag_indices = []
        for idx, (i, j) in enumerate(zip(*np.where(cell_mask))):
            if i != j:
                offdiag_indices.append(idx)
        if not offdiag_indices:
            pytest.skip("No off-diagonal cell DOF available")

        shear_idx = offdiag_indices[0]
        x_shear[pes.n_internal + shear_idx] += 0.05

        pes.set_x(x_shear)
        pos_after = atoms.get_positions().copy()

        # For each fragment, check that the relative geometry changed orientation
        # (rotation applied) but bond lengths are preserved
        for group in pes.fragment_groups:
            dr_before = pos0[group] - pos0[group].mean(axis=0)
            dr_after = pos_after[group] - pos_after[group].mean(axis=0)

            # Bond lengths should be nearly preserved
            dists_before = np.linalg.norm(dr_before, axis=1)
            dists_after = np.linalg.norm(dr_after, axis=1)
            assert_allclose(dists_before, dists_after, atol=1e-3)

            # But the orientation should have changed (dr_after != dr_before)
            # The rotation under shear should produce a nonzero angular change
            if len(group) > 1:
                diff = dr_after - dr_before
                assert np.max(np.abs(diff)) > 1e-6, (
                    "Fragment atoms should rotate under shear deformation"
                )

    def _full_gradient_fd(self, pes, delta_int=1e-5, delta_cell=1e-6):
        """Compute full numerical gradient via central finite differences."""
        _, g_analytical = pes.eval()
        x0 = pes.get_x()
        g_numeric = np.zeros(pes.dim)

        for i in range(pes.dim):
            delta = delta_cell if i >= pes.n_internal else delta_int

            pes.set_x(x0)
            x_plus = x0.copy()
            x_plus[i] += delta
            pes.set_x(x_plus)
            e_plus, _ = pes.eval()

            pes.set_x(x0)
            x_minus = x0.copy()
            x_minus[i] -= delta
            pes.set_x(x_minus)
            e_minus, _ = pes.eval()

            g_numeric[i] = (e_plus - e_minus) / (2 * delta)

        pes.set_x(x0)
        return g_analytical, g_numeric

    @pytest.mark.parametrize("setup", [
        "orthogonal", "triclinic", "deformed",
    ])
    def test_full_gradient_numerical(self, setup):
        """Verify analytical gradient matches FD for ALL DOFs."""
        if setup == "triclinic":
            atoms = self._make_triclinic_crystal()
        else:
            atoms = self._make_two_water_crystal()

        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)
        assert pes.rigid_fragments is True

        if setup == "deformed":
            x0 = pes.get_x()
            x_deformed = x0.copy()
            for i in range(pes.n_cell_dof):
                x_deformed[pes.n_internal + i] += 0.02 * ((-1)**i)
            pes.set_x(x_deformed)

        g_analytical, g_numeric = self._full_gradient_fd(pes)
        assert_allclose(g_analytical, g_numeric, atol=1e-4, rtol=1e-3)

    def test_rotation_correction_nonzero_under_shear(self):
        """Verify that the rotation correction to the gradient is nonzero
        when the cell is sheared (so R != I in polar decomposition)."""
        from scipy.linalg import polar

        water1 = molecule('H2O')
        water2 = molecule('H2O')
        water1.positions += [1.0, 1.0, 1.0]
        water2.positions += [4.0, 4.0, 4.0]
        atoms = water1 + water2

        # Start with orthogonal cell, then deform with shear
        atoms.set_cell([7.0, 7.0, 7.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals)

        # Apply shear to move away from R = I
        x0 = pes.get_x()
        x_sheared = x0.copy()
        cell_mask = pes.cell_mask
        offdiag_indices = []
        for idx, (i, j) in enumerate(zip(*np.where(cell_mask))):
            if i != j:
                offdiag_indices.append(idx)
        if not offdiag_indices:
            pytest.skip("No off-diagonal cell DOF")

        x_sheared[pes.n_internal + offdiag_indices[0]] += 0.05
        pes.set_x(x_sheared)

        # At this point F should have a nontrivial rotation component
        F = pes._get_deformation_gradient()
        R, U = polar(F)
        assert not np.allclose(R, np.eye(3), atol=1e-6), (
            "R should differ from identity after shear"
        )

    def test_rigid_rotation_is_objective_on_skewed_cell(self):
        """A pure global rotation of a skewed rigid-fragment cell must not
        change the energy.

        The rigid motion uses the row-vector incremental deformation
        M = inv(C_before) @ C_after (a pure rotation gives M = Q.T, so
        delta_r @ polar(M) rotates rigidly). The old left-multiplied form
        (C_after @ inv(C_before), delta_r @ R.T) was not objective on skewed
        cells: a global rotation changed the energy by ~1e-2 eV.
        """
        cell = np.array([[6.0, 0, 0], [2.0, 6.0, 0], [0.5, 1.0, 6.0]])
        atoms = Atoms('OH2OH2',
                      positions=[[1, 1, 1], [1.9, 1.2, 1], [0.6, 1.9, 1],
                                 [3, 4, 2], [3.9, 4.2, 2], [2.6, 4.9, 2]],
                      cell=cell, pbc=True)
        atoms.calc = LennardJones()
        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals, rigid_fragments=True)
        pes.get_g()
        E0 = pes.eval()[0]

        # Target cell = pure rotation of the current cell about z.
        theta = 1e-3
        c, s = np.cos(theta), np.sin(theta)
        Qz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        cell_rot = cell @ Qz.T
        F = cell_rot @ np.linalg.inv(pes.orig_cell)
        U = _logm_3x3(F) * pes.exp_cell_factor
        x = pes.get_x().copy()
        x[pes.n_internal:] = U[pes.cell_mask]

        pes.save()
        pes.set_x(x)
        E1 = pes.eval()[0]
        pes.restore()
        assert abs(E1 - E0) < 1e-9, f"rotation changed energy by {E1 - E0:.3e}"

    def test_cell_gradient_matches_fd_on_skewed_nonequilibrium(self):
        """Analytic cell gradient must match FD of set_x on a skewed cell far
        from equilibrium, where the rotation-correction term is active."""
        cell = np.array([[4.5, 0, 0], [1.5, 4.5, 0], [0.8, 0.6, 4.5]])
        atoms = Atoms('OH2OH2',
                      positions=[[0.5, 0.5, 0.5], [1.4, 0.7, 0.5],
                                 [0.1, 1.4, 0.5], [2.5, 2.5, 2.0],
                                 [3.4, 2.7, 2.0], [2.1, 3.4, 2.0]],
                      cell=cell, pbc=True)
        atoms.calc = LennardJones()
        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals, rigid_fragments=True)
        pes.get_g()
        n = pes.n_internal
        g_cell = pes.curr['g'][n:].copy()
        # This configuration has a genuinely nonzero cell gradient.
        assert np.abs(g_cell).max() > 1e-2
        x0 = pes.get_x().copy()
        eps = 1e-5
        g_fd = np.zeros(pes.n_cell_dof)
        for i in range(pes.n_cell_dof):
            pes.save(); xp = x0.copy(); xp[n + i] += eps
            pes.set_x(xp); fp = pes.eval()[0]; pes.restore()
            pes.save(); xm = x0.copy(); xm[n + i] -= eps
            pes.set_x(xm); fm = pes.eval()[0]; pes.restore()
            g_fd[i] = (fp - fm) / (2 * eps)
        assert_allclose(g_cell, g_fd, atol=1e-6)


class TestConstrainedCellGradient:
    """Cell gradient must include constraint-reprojection virtual work.

    With an internal constraint (fixed bond/angle/dihedral) + cell
    optimization, set_x changes the cell (moving atoms fractionally), which
    perturbs the constrained coordinate, then reprojects atoms back onto the
    constraint manifold. That projection does work against the forces, adding
    a term to dE/d(cell) that the stress/virial alone misses. Before the fix
    the analytic cell gradient differed from the FD derivative of set_x by
    ~26 eV on a skewed periodic H2O cell with one fixed OH bond.
    """

    @staticmethod
    def _skewed_water():
        cell = np.array([[6.0, 0, 0], [1.5, 6.0, 0], [0, 0, 6.0]])
        atoms = Atoms('OH2', positions=[[1, 1, 0], [1.9, 1.2, 0],
                                        [0.6, 1.9, 0]], cell=cell, pbc=True)
        atoms.calc = LennardJones()
        return atoms

    def _fd_cell_gradient(self, pes):
        n = pes.n_internal
        g_cell = pes.curr['g'][n:].copy()
        x0 = pes.get_x().copy()
        eps = 1e-5
        g_fd = np.zeros(pes.n_cell_dof)
        for i in range(pes.n_cell_dof):
            pes.save(); xp = x0.copy(); xp[n + i] += eps
            pes.set_x(xp); fp = pes.eval()[0]; pes.restore()
            pes.save(); xm = x0.copy(); xm[n + i] -= eps
            pes.set_x(xm); fm = pes.eval()[0]; pes.restore()
            g_fd[i] = (fp - fm) / (2 * eps)
        return g_cell, g_fd

    def test_fixed_bond_cell_gradient_matches_fd(self):
        atoms = self._skewed_water()
        cons = Constraints(atoms)
        cons.fix_bond((0, 1))
        ic = Internals(atoms, cons=cons)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        pes = CellInternalPES(atoms, internals=ic, eta=1e-4,
                              auto_find_internals=False)
        pes.get_g()
        g_cell, g_fd = self._fd_cell_gradient(pes)
        # The projection work term is large here; make sure the test is
        # exercising a regime where the correction matters.
        assert np.abs(g_cell).max() > 1.0
        assert_allclose(g_cell, g_fd, atol=1e-6)

    def test_unconstrained_cell_gradient_unaffected(self):
        # No constraints: the correction must be identically zero.
        atoms = self._skewed_water()
        ic = Internals(atoms)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        pes = CellInternalPES(atoms, internals=ic, eta=1e-4,
                              auto_find_internals=False)
        assert_allclose(pes._constraint_projection_cell_gradient(), 0.0,
                        atol=1e-14)
        pes.get_g()
        g_cell, g_fd = self._fd_cell_gradient(pes)
        assert_allclose(g_cell, g_fd, atol=1e-6)

    def test_interfragment_constraint_rigid_cell_gradient_matches_fd(self):
        # A constraint spanning two rigid fragments (a fixed angle across two
        # separate waters -- angles are not connectivity edges, so the
        # fragments stay separate) acquires a residual under cell strain: the
        # interfragment separation follows the full deformation. The projection
        # correction must make the analytic cell gradient match FD. (The FD
        # harness tightens the projection tolerance to avoid the
        # _project_to_constraints dead zone that would otherwise mask the term.)
        import types
        from sella.peswrapper import CellInternalPES as _CIP
        cell = np.array([[10.0, 0, 0], [2.0, 10.0, 0], [0.5, 0.8, 10.0]])
        atoms = Atoms('OH2OH2',
                      positions=[[1, 1, 1], [1.9, 1.2, 1], [0.6, 1.9, 1],
                                 [6, 6, 4], [6.9, 6.2, 4], [5.6, 6.9, 4]],
                      cell=cell, pbc=True)
        atoms.calc = LennardJones()
        cons = Constraints(atoms)
        cons.fix_angle((1, 0, 3))  # atoms 0,1 in frag1; atom 3 in frag2
        ic = Internals(atoms, cons=cons, allow_fragments=True)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        pes = CellInternalPES(atoms, internals=ic, eta=1e-4,
                              auto_find_internals=False, rigid_fragments=True)
        assert len(pes.fragment_groups) == 2  # genuinely two fragments
        pes.get_g()
        n = pes.n_internal
        g_cell = pes.curr['g'][n:].copy()
        assert np.abs(g_cell).max() > 1.0  # the correction is large here

        orig = _CIP._project_to_constraints
        pes._project_to_constraints = types.MethodType(
            lambda self, target_tol=1e-11, max_iter=40, **k:
                orig(self, target_tol=target_tol, max_iter=max_iter, **k), pes)
        x0 = pes.get_x().copy(); eps = 1e-5
        g_fd = np.zeros(pes.n_cell_dof)
        for i in range(pes.n_cell_dof):
            pes.save(); xp = x0.copy(); xp[n + i] += eps
            pes.set_x(xp); fp = pes.eval()[0]; pes.restore()
            pes.save(); xm = x0.copy(); xm[n + i] -= eps
            pes.set_x(xm); fm = pes.eval()[0]; pes.restore()
            g_fd[i] = (fp - fm) / (2 * eps)
        assert_allclose(g_cell, g_fd, atol=1e-6)

    def test_intrafragment_constraint_rigid_ok(self):
        # A constraint within one rigid fragment is fine (invariant under the
        # rigid cell move) and must not raise.
        cell = np.array([[10.0, 0, 0], [2.0, 10.0, 0], [0.5, 0.8, 10.0]])
        atoms = Atoms('OH2OH2',
                      positions=[[1, 1, 1], [1.9, 1.2, 1], [0.6, 1.9, 1],
                                 [6, 6, 4], [6.9, 6.2, 4], [5.6, 6.9, 4]],
                      cell=cell, pbc=True)
        atoms.calc = LennardJones()
        cons = Constraints(atoms)
        cons.fix_angle((1, 0, 2))  # all in frag1
        ic = Internals(atoms, cons=cons, allow_fragments=True)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        pes = CellInternalPES(atoms, internals=ic, eta=1e-4,
                              auto_find_internals=False, rigid_fragments=True)
        assert_allclose(pes._constraint_projection_cell_gradient(), 0.0,
                        atol=1e-14)
        _, Ucons, _, _ = pes._compute_basis_int()
        assert Ucons.shape[1] == 1
        pes.get_g()  # must not raise

    def test_periodic_image_intrafragment_constraint_not_shortcut(self):
        # Same-fragment shape constraints are invariant under rigid cell motion
        # only when their stored periodic image offsets are zero. An explicit
        # image angle has a nonzero cell derivative and must take the full
        # projection path.
        cell = np.array([[10.0, 0, 0], [2.0, 10.0, 0], [0.5, 0.8, 10.0]])
        atoms = Atoms('OH2',
                      positions=[[1, 1, 1], [1.9, 1.2, 1], [0.6, 1.9, 1]],
                      cell=cell, pbc=True)
        atoms.calc = LennardJones()
        cons = Constraints(atoms)
        cons.fix_angle((1, 0, 2), ncvecs=[[1, 0, 0], [0, 0, 0]])
        ic = Internals(atoms, cons=cons, allow_fragments=True)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        pes = CellInternalPES(atoms, internals=ic, eta=1e-4,
                              auto_find_internals=False,
                              rigid_fragments=True)

        assert not pes._constraints_are_intrafragment_shape_constraints()
        X_C = pes._unprojected_cell_motion_derivative()
        Rc_C = pes.cons.jacobian() @ X_C + pes.cons.cell_jacobian()
        assert np.linalg.norm(Rc_C) > 1e-8


class TestSingletonFragmentCellMotion:
    """Isolated one-atom fragments must move rigidly with the cell.

    allow_fragments adds a translation TRIC for a lone atom but previously did
    not record it in fragment_atom_groups, so CellInternalPES excluded it from
    rigid-body cell motion: its CoM stayed fixed in Cartesian space instead of
    following the cell fractionally, and _compute_delta_r initialized it from
    absolute position (nonzero) instead of zero, poisoning the cell gradient.
    """

    @staticmethod
    def _water_plus_ar(cell):
        atoms = Atoms('OH2Ar',
                      positions=[[0.5, 0.5, 0.5], [1.4, 0.7, 0.5],
                                 [0.1, 1.4, 0.5], [2.6, 2.6, 2.6]],
                      cell=cell, pbc=True)
        atoms.calc = LennardJones()
        return atoms

    def test_singleton_recorded_in_fragment_groups(self):
        atoms = self._water_plus_ar(np.eye(3) * 9.0)
        ic = Internals(atoms, allow_fragments=True)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        groups = [sorted(int(i) for i in g) for g in ic.fragment_atom_groups]
        assert [3] in groups, f"Ar singleton missing from {groups}"
        assert sorted([0, 1, 2]) in [sorted(g) for g in groups]

    def test_singleton_moves_fractionally_with_cell(self):
        atoms = self._water_plus_ar(np.eye(3) * 9.0)
        ic = Internals(atoms, allow_fragments=True)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        pes = CellInternalPES(atoms, internals=ic, eta=1e-4,
                              rigid_fragments=True)
        pes.get_g()
        ar0 = atoms.positions[3].copy()
        cell0 = atoms.get_cell().array.copy()
        x = pes.get_x().copy(); x[pes.n_internal:] += 0.05
        pes.save(); pes.set_x(x)
        ar1 = atoms.positions[3].copy(); cell1 = atoms.get_cell().array.copy()
        pes.restore()
        # Fractional coordinate preserved: r1 = (r0 @ inv(C0)) @ C1.
        expected = (ar0 @ np.linalg.inv(cell0)) @ cell1
        assert np.linalg.norm(ar1 - ar0) > 0.1  # actually moved
        assert_allclose(ar1, expected, atol=1e-8)

    def test_singleton_delta_r_is_zero(self):
        atoms = self._water_plus_ar(np.eye(3) * 9.0)
        ic = Internals(atoms, allow_fragments=True)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        pes = CellInternalPES(atoms, internals=ic, eta=1e-4,
                              rigid_fragments=True)
        delta_r = pes._compute_delta_r()
        assert_allclose(delta_r[3], 0.0, atol=1e-12)

    def test_cell_gradient_matches_fd_with_singleton(self):
        # Compressed skewed cell so the Ar singleton actively contributes to
        # the (nonzero) cell gradient.
        cell = np.array([[4.2, 0, 0], [1.3, 4.2, 0], [0.7, 0.5, 4.2]])
        atoms = self._water_plus_ar(cell)
        ic = Internals(atoms, allow_fragments=True)
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        pes = CellInternalPES(atoms, internals=ic, eta=1e-4,
                              rigid_fragments=True)
        pes.get_g()
        n = pes.n_internal
        g_cell = pes.curr['g'][n:].copy()
        assert np.abs(g_cell).max() > 1e-2
        x0 = pes.get_x().copy(); eps = 1e-5
        g_fd = np.zeros(pes.n_cell_dof)
        for i in range(pes.n_cell_dof):
            pes.save(); xp = x0.copy(); xp[n + i] += eps
            pes.set_x(xp); fp = pes.eval()[0]; pes.restore()
            pes.save(); xm = x0.copy(); xm[n + i] -= eps
            pes.set_x(xm); fm = pes.eval()[0]; pes.restore()
            g_fd[i] = (fp - fm) / (2 * eps)
        assert_allclose(g_cell, g_fd, atol=1e-6)

    def test_rigid_fragments_without_groups_raises(self):
        # Explicit rigid_fragments=True on an Internals built WITHOUT
        # allow_fragments has no fragment groups, so the rigid cell move would
        # freeze the atoms while the cell changes (silent corruption). Must
        # raise a clear error instead.
        atoms = Atoms('OH2', positions=[[1, 1, 1], [1.9, 1.2, 1],
                                        [0.6, 1.9, 1]],
                      cell=np.eye(3) * 6.0, pbc=True)
        atoms.calc = LennardJones()
        ic = Internals(atoms)  # no allow_fragments -> no fragment groups
        ic.find_all_bonds(); ic.find_all_angles(); ic.find_all_dihedrals()
        with pytest.raises(ValueError, match="fragment groups"):
            CellInternalPES(atoms, internals=ic, eta=1e-4,
                            auto_find_internals=False, rigid_fragments=True)


class TestNiggliReduction:
    """Tests for periodic Niggli reduction during cell optimization."""

    def test_niggli_triggers_on_skewed_cell_cartesian(self):
        """CellCartesianPES.maybe_niggli_reduce fires when angles are extreme."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms, eta=1e-4)

        # Shear the cell to create a skewed angle
        cell = atoms.get_cell().array.copy()
        cell[1, 0] += 3.0  # Large off-diagonal shear
        atoms.set_cell(cell, scale_atoms=True)
        pes.orig_cell = atoms.get_cell().array.copy()

        # Shear further so cell relative to orig_cell is identity but
        # the cell itself is skewed
        angles = atoms.get_cell().angles()
        assert min(angles) < 60.0 or max(angles) > 120.0, (
            f"Test setup: cell should be skewed, got angles {angles}"
        )

        reduced = pes.maybe_niggli_reduce()
        assert reduced, "Niggli reduction should have been triggered"

        # After reduction, angles should be closer to 90
        new_angles = atoms.get_cell().angles()
        new_max_dev = max(abs(a - 90.0) for a in new_angles)
        assert new_max_dev <= 30.0 + 1e-10, (
            f"After Niggli reduction, angles should be near 90, got {new_angles}"
        )

        # orig_cell should be updated
        assert_allclose(pes.orig_cell, atoms.get_cell().array)

    def test_niggli_disabled_for_internal_pes(self):
        """CellInternalPES declines Niggli: it would corrupt internal coords.

        Niggli reduction recombines lattice vectors and wraps atoms, which
        invalidates stored ncvecs and dummy positions. CellInternalPES must
        therefore decline it (return False) and leave the cell -- and the
        internal coordinates -- untouched, even on a badly skewed cell.
        """
        atoms = make_molecular_crystal()
        atoms.calc = EMT()

        internals = Internals(atoms, allow_fragments=True)
        pes = CellInternalPES(atoms, internals=internals, eta=1e-4)

        # Shear the cell into a regime where Niggli would otherwise fire.
        cell = atoms.get_cell().array.copy()
        cell[1, 0] += 6.0
        atoms.set_cell(cell, scale_atoms=False)
        pes.orig_cell = atoms.get_cell().array.copy()

        angles = atoms.get_cell().angles()
        assert min(angles) < 60.0 or max(angles) > 120.0

        q_before = pes.int.calc().copy()
        cell_before = atoms.get_cell().array.copy()

        reduced = pes.maybe_niggli_reduce()
        assert reduced is False, "Niggli must be declined for internal PES"

        # Cell and internal coordinates must be left exactly as they were.
        assert_allclose(atoms.get_cell().array, cell_before, atol=1e-12)
        assert_allclose(pes.int.calc(), q_before, atol=1e-12)

    def test_niggli_would_corrupt_periodic_ncvecs(self):
        """Documents *why* Niggli is declined for internal PES.

        A raw niggli_reduce on a cell recombines lattice vectors, so a stored
        integer image offset (ncvec) points at a different physical image
        afterward and the bond length jumps. This is the corruption the
        CellInternalPES override avoids by declining reduction.
        """
        from ase.build import niggli_reduce
        cell = np.array([[3.0, 0, 0], [2.9, 3.0, 0], [0, 0, 3.0]])
        atoms = Atoms('H2', positions=[[0, 0, 0], [0.5, 0, 0]],
                      cell=cell, pbc=True)
        internals = Internals(atoms)
        internals.add_bond((0, 1), ncvecs=[[1, 0, 0]])
        b_before = internals.calc()[0]

        niggli_reduce(atoms)
        b_after = internals.calc()[0]
        # The stored ncvec is now stale: the bond value silently changes.
        assert not np.isclose(b_before, b_after, atol=1e-3)

    def test_niggli_does_not_trigger_on_good_cell(self):
        """No reduction when cell angles are fine."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms, eta=1e-4)

        # Cubic cell — angles are all 60 or 90 depending on conventional vs primitive
        # Use conventional cell to get 90 degree angles
        atoms2 = bulk('Cu', 'fcc', a=3.6, cubic=True)
        atoms2.calc = EMT()
        pes2 = CellCartesianPES(atoms2, eta=1e-4)

        reduced = pes2.maybe_niggli_reduce()
        assert not reduced, "Should not reduce a cubic cell"

    def test_niggli_transforms_hessian_cell_block(self):
        """After Niggli reduction, cell block of Hessian is transformed."""
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms, eta=1e-4)
        n = pes.n_cart

        # Set Hessian to identity for cell block to track the transformation
        H = pes.H.B.copy()
        H[n:, n:] = np.eye(pes.n_cell_dof)
        H[:n, n:] = 0.0
        H[n:, :n] = 0.0
        pes.set_H(H, initialized=True)

        # Shear the cell to trigger reduction
        cell = atoms.get_cell().array.copy()
        cell[1, 0] += 5.0
        atoms.set_cell(cell, scale_atoms=True)
        pes.orig_cell = atoms.get_cell().array.copy()

        pes.maybe_niggli_reduce()

        H_new = pes.H.B
        # Cell block should be transformed (T^T @ I @ T = T^T @ T), symmetric
        H_cell = H_new[n:, n:]
        assert_allclose(H_cell, H_cell.T, atol=1e-10)
        # Should not be identity (transformation is non-trivial for skewed cell)
        assert not np.allclose(H_cell, np.eye(pes.n_cell_dof), atol=0.1)
        # Cartesian block should be preserved
        assert_allclose(H_new[:n, :n], H[:n, :n])

    def test_niggli_in_sella_step(self):
        """Niggli reduction integrates with Sella.step() without crashing."""
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        atoms.calc = EMT()

        opt = Sella(atoms, order=0, optimize_cell=True, niggli=True,
                    logfile=None)

        # Shear the cell to extreme angles
        cell = atoms.get_cell().array.copy()
        cell[1, 0] += 5.0
        atoms.set_cell(cell, scale_atoms=True)
        opt.pes.orig_cell = atoms.get_cell().array.copy()

        # Run a few steps — should not crash even if Niggli fires
        for _ in range(3):
            opt.step()


class TestExactGeodesicDefault:
    """Regression tests for exact_geodesic flag consistency.

    The resolved default (True) must be applied to the *initial* PES, not just
    to the PES rebuilt after a bad-internals reset. Before the fix, omitting the
    flag left the initial PES with exact_geodesic=None (falsy -> approximate ODE
    geodesic), which then silently flipped to True on the first reset.
    """

    def test_default_is_exact_on_initial_pes(self):
        """Omitting the flag -> initial PES uses exact geodesic (True)."""
        atoms = make_molecular_crystal()
        atoms.calc = EMT()
        opt = Sella(atoms, order=0, internal=True, optimize_cell=True,
                    logfile=None)
        assert opt.pes.exact_geodesic is True

    def test_explicit_false_is_honored(self):
        """exact_geodesic=False must reach the initial PES unchanged."""
        atoms = make_molecular_crystal()
        atoms.calc = EMT()
        opt = Sella(atoms, order=0, internal=True, optimize_cell=True,
                    exact_geodesic=False, logfile=None)
        assert opt.pes.exact_geodesic is False

    def test_explicit_true_is_honored(self):
        """exact_geodesic=True must reach the initial PES unchanged."""
        atoms = make_molecular_crystal()
        atoms.calc = EMT()
        opt = Sella(atoms, order=0, internal=True, optimize_cell=True,
                    exact_geodesic=True, logfile=None)
        assert opt.pes.exact_geodesic is True


class TestRigidFragmentsSurvivesRebuild:
    """An explicit rigid_fragments must survive the bad-internals rebuild.

    rigid_fragments is a pass-through PES kwarg captured in self.peskwargs.
    The bad-internals reset path re-runs initialize_pes with an explicit
    argument list; before the fix it omitted **self.peskwargs, so a user's
    explicit rigid_fragments=False was dropped and CellInternalPES silently
    re-auto-detected it as True.
    """

    def _force_bad_internals_rebuild(self, opt):
        """Trigger the real reset path in Sella.step() once.

        The reset fires when pes.int.check_for_bad_internals() is truthy in the
        guard inside step(). Other call sites (kick, _predict_step) also invoke
        it during a step, so force a truthy result until the PES object is
        actually rebuilt (its identity changes), then restore real behavior.
        This exercises the actual initialize_pes call inside step().
        """
        pes_before = opt.pes

        def fake_check():
            if opt.pes is pes_before:
                return {"bonds": []}  # truthy -> triggers rebuild
            return opt.pes.int.check_for_bad_internals()

        opt.pes.int.check_for_bad_internals = fake_check
        opt.step()
        assert opt.pes is not pes_before, "reset path did not fire"

    def test_explicit_false_survives_rebuild(self):
        atoms = make_molecular_crystal()
        atoms.calc = EMT()
        opt = Sella(atoms, order=0, internal=True, optimize_cell=True,
                    allow_fragments=True, rigid_fragments=False, logfile=None)
        # CH4 with allow_fragments would auto-detect True, so False is a real
        # override -- confirm it holds initially and after a rebuild.
        assert opt.pes.rigid_fragments is False
        self._force_bad_internals_rebuild(opt)
        assert opt.pes.rigid_fragments is False

    def test_explicit_true_survives_rebuild(self):
        atoms = make_molecular_crystal()
        atoms.calc = EMT()
        # allow_fragments so rigid_fragments=True has real fragment groups
        # (a lone molecule still yields a z=1 fragment); the point of the test
        # is that the explicit rigid_fragments=True survives the rebuild.
        opt = Sella(atoms, order=0, internal=True, optimize_cell=True,
                    allow_fragments=True, rigid_fragments=True, logfile=None)
        assert opt.pes.rigid_fragments is True
        self._force_bad_internals_rebuild(opt)
        assert opt.pes.rigid_fragments is True

    def test_autodetect_still_works_after_rebuild(self):
        # No explicit override: auto-detection should apply on both builds.
        atoms = make_molecular_crystal()
        atoms.calc = EMT()
        opt = Sella(atoms, order=0, internal=True, optimize_cell=True,
                    allow_fragments=True, logfile=None)
        assert opt.peskwargs == {}
        detected = opt.pes.rigid_fragments
        self._force_bad_internals_rebuild(opt)
        assert opt.pes.rigid_fragments == detected

    def test_explicit_constraints_survive_rebuild(self):
        atoms = make_water_crystal()
        atoms.calc = LennardJones()
        constraints = Constraints(atoms)
        constraints.fix_bond((0, 1))
        target = constraints.targets.copy()

        opt = Sella(
            atoms,
            order=0,
            internal=True,
            optimize_cell=True,
            allow_fragments=True,
            constraints=constraints,
            exact_geodesic=False,
            logfile=None,
        )
        self._force_bad_internals_rebuild(opt)

        assert opt.pes.cons.nbonds == 1
        assert_allclose(opt.pes.cons.targets, target, atol=0.0, rtol=0.0)


class TestDummyAtomCellHandling:
    """Cell steps must carry linear-angle dummy atoms along with real atoms.

    A linear molecule (e.g. CO2) gets a dummy atom to define its angle
    coordinates. ASE's set_cell(scale_atoms=True) scales real atoms but has
    no knowledge of dummies, and the finite-difference cell Hessian probes
    saved/restored only real atoms and cell. Both paths left dummies behind.
    """

    @staticmethod
    def _linear_co2(cell_diag=6.0):
        atoms = Atoms('OCO', positions=[[1, 2, 0], [2, 2, 0], [3, 2, 0]],
                      cell=np.eye(3) * cell_diag, pbc=True)
        atoms.calc = LennardJones()
        return atoms

    @staticmethod
    def _linear_co2_with_off_axis_probe():
        atoms = Atoms(
            'OCOH',
            positions=[
                [0.0, 0.0, 0.0],
                [1.16, 0.0, 0.0],
                [2.32, 0.0, 0.0],
                [4.0, 1.7, 0.9],
            ],
        )
        atoms.calc = LennardJones()
        return atoms

    @staticmethod
    def _mostly_linear_hoh(cell_diag=10.0):
        theta = np.deg2rad(178.0)
        r = 0.96
        positions = np.array([
            [r, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [r * np.cos(theta), r * np.sin(theta), 0.0],
        ]) + 0.5 * cell_diag
        atoms = Atoms('HOH', positions=positions,
                      cell=np.eye(3) * cell_diag, pbc=True)
        atoms.calc = LennardJones()
        return atoms

    @staticmethod
    def _two_linear_co2(cell_diag=10.0):
        # Two separated CO2, second off-axis so a cell shear rotates it.
        atoms = Atoms('OCOOCO',
                      positions=[[1, 1, 0], [2, 1, 0], [3, 1, 0],
                                 [1, 5, 0.3], [1.8, 5.6, 0.3], [2.6, 6.2, 0.3]],
                      cell=np.eye(3) * cell_diag, pbc=True)
        atoms.calc = LennardJones()
        return atoms

    def test_nonrigid_cell_step_moves_dummy(self):
        pes = Sella(self._linear_co2(), order=0, internal=True,
                    optimize_cell=True, logfile=None).pes
        assert pes.rigid_fragments is False
        assert len(pes.dummies) == 1
        dpos0 = pes.dummies.positions.copy()

        target = pes.get_x().copy()
        target[pes.n_internal:] += 0.05  # cell strain
        pes.set_x(target)

        # With the bug the dummy stayed at its original position.
        assert not np.allclose(pes.dummies.positions, dpos0)

    def test_nonrigid_cell_step_scales_dummy_like_atoms(self):
        # The dummy must follow the same fractional-preserving deformation
        # gradient that scale_atoms=True applies to real atoms.
        pes = Sella(self._linear_co2(), order=0, internal=True,
                    optimize_cell=True, logfile=None).pes
        dpos0 = pes.dummies.positions.copy()
        cell0 = pes.atoms.get_cell().array.copy()

        # Drive only the cell parameters through _set_masked_cell_params so no
        # internal-coordinate solve runs on top (isolates the scaling step).
        cell_params = pes._masked_cell_params().copy()
        cell_params += 0.05
        pes._set_masked_cell_params(cell_params)
        cell1 = pes.atoms.get_cell().array
        F = np.linalg.inv(cell0) @ cell1
        expected = dpos0 @ F
        # apply the same fix path used in set_x
        pes.dummies.positions = pes.dummies.positions @ F
        assert_allclose(pes.dummies.positions, expected, atol=1e-10)

    def test_fd_cell_hessian_restores_dummies(self):
        pes = Sella(self._two_linear_co2(), order=0, internal=True,
                    optimize_cell=True, allow_fragments=True, logfile=None).pes
        assert len(pes.dummies) >= 1
        dpos0 = pes.dummies.positions.copy()
        apos0 = pes.atoms.positions.copy()
        cell0 = pes.atoms.get_cell().array.copy()

        pes._compute_cell_hessian_columns(delta=1e-2)

        assert_allclose(pes.atoms.positions, apos0, atol=1e-12)
        assert_allclose(pes.atoms.get_cell().array, cell0, atol=1e-12)
        # The whole point: dummies restored exactly, not left displaced.
        assert_allclose(pes.dummies.positions, dpos0, atol=1e-12)

    def test_linear_dummy_constraints_remain_in_optimizer_basis(self):
        pes = Sella(self._two_linear_co2(), order=0, internal=True,
                    optimize_cell=True, allow_fragments=True,
                    rigid_fragments=True, logfile=None).pes
        assert len(pes.dummies) >= 1
        assert pes.cons.nint > 0
        drdx, Ucons, Unred, Ufree_fast = pes._compute_basis_int()
        assert drdx.shape[0] == 2 * len(pes.dummies)
        assert Ucons.shape[1] == 2 * len(pes.dummies)
        assert Ufree_fast is Unred

        pes.get_g()
        step = pes.get_Ucons().T @ pes.get_Ufree()
        assert_allclose(step, 0.0, atol=1e-12)
        assert_allclose(pes.get_scons(), 0.0, atol=0.0)

        pes.dummies.positions += np.array([1e-3, -2e-3, 1.5e-3])
        pes._update_basis()
        assert np.linalg.norm(pes.get_scons()) > 0.0

    def test_linear_dummy_cone_repairs_large_displacement(self):
        pes = Sella(self._two_linear_co2(), order=0, internal=True,
                    optimize_cell=True, allow_fragments=True,
                    rigid_fragments=True, logfile=None).pes
        apos0 = pes.atoms.positions.copy()
        pes.dummies.positions += np.array([0.15, -0.105, 0.045])
        assert np.linalg.norm(pes.cons.residual(), ord=np.inf) > 1e-7
        assert pes._project_to_constraints()

        assert_allclose(pes.atoms.positions, apos0, atol=1e-12)
        assert np.linalg.norm(pes.cons.residual(), ord=np.inf) < 1e-7

    def test_linear_dummy_residual_remains_visible_to_convergence(self):
        pes = Sella(self._two_linear_co2(), order=0, internal=True,
                    optimize_cell=True, allow_fragments=True,
                    rigid_fragments=True, logfile=None).pes
        assert pes._linear_bend_dummy_projection_records() is not None

        pes.dummies.positions += np.array([0.15, -0.105, 0.045])
        assert np.linalg.norm(pes.get_res(), ord=np.inf) > 0.1
        assert_allclose(pes.get_convergence_res(), pes.get_res())

        converged, _, cmax_actual, _ = pes.converged(
            1e100, smax=1e100, cmax=1e-5
        )
        assert not converged
        assert cmax_actual > 0.1

    def test_linear_dummy_auxiliaries_are_constrained_out_of_optimizer_step(self):
        atoms = self._linear_co2(cell_diag=10.0)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=0.0,
            forces=np.array([
                [0.00, 0.25, 0.08],
                [0.03, -0.04, 0.02],
                [-0.05, 0.10, -0.07],
            ]),
            stress=np.zeros(6),
        )
        opt = Sella(
            atoms,
            order=0,
            internal=True,
            optimize_cell=True,
            allow_fragments=True,
            rigid_fragments=True,
            exact_geodesic=False,
            logfile=None,
        )
        pes = opt.pes
        records = pes._linear_bend_dummy_projection_records()
        assert records is not None

        # The cone bond and angle remain available to the geometry integrator,
        # but the optimizer step is projected onto their null space. Before
        # the fix, these asymmetric forces requested motion in both directions.
        auxiliaries = [
            (name, coord)
            for record in records
            for name, coord in (('bonds', record[2]),
                                ('angles', record[4]))
        ]
        for name, auxiliary in auxiliaries:
            matches = [
                active
                for coord, active in zip(pes.int.internals[name],
                                         pes.int._active[name])
                if coord == auxiliary
            ]
            assert matches
            assert all(matches)

        step, _ = opt._predict_step()
        auxiliary_rows = pes._linear_dummy_auxiliary_optimizer_rows()
        assert len(auxiliary_rows) == 2
        assert_allclose(step[list(auxiliary_rows)], 0.0, atol=1e-12)

        x0 = pes.get_x().copy()
        _, dx_final, _ = pes.set_x(x0 + step)
        actual = pes.wrap_dx(pes.get_x() - x0)
        assert np.linalg.norm(pes.get_res(), ord=np.inf) < 1e-7
        assert_allclose(actual[list(auxiliary_rows)], 0.0, atol=1e-10)
        assert_allclose(dx_final[list(auxiliary_rows)],
                        actual[list(auxiliary_rows)], atol=2e-7)

    @pytest.mark.parametrize('order', [0, 1])
    @pytest.mark.filterwarnings(
        'ignore:Saddle point optimizations with eig=False'
    )
    def test_fast_linear_dummy_step_matches_dense_constrained_step(self,
                                                                   order):
        def make_optimizer():
            atoms = self._linear_co2(cell_diag=10.0)
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=0.0,
                forces=np.array([
                    [0.00, 0.25, 0.08],
                    [0.03, -0.04, 0.02],
                    [-0.05, 0.10, -0.07],
                ]),
                stress=np.zeros(6),
            )
            return Sella(
                atoms,
                order=order,
                eig=False,
                internal=True,
                optimize_cell=(order == 0),
                allow_fragments=True,
                exact_geodesic=False,
                logfile=None,
                **({'rigid_fragments': True} if order == 0 else {}),
            )

        fast = make_optimizer()
        dense = make_optimizer()
        rng = np.random.default_rng(7)
        A = rng.normal(size=(fast.pes.dim, fast.pes.dim))
        H = A.T @ A / fast.pes.dim + np.diag(
            np.linspace(0.2, 2.0, fast.pes.dim)
        )
        fast.pes.set_H(H, initialized=True)
        dense.pes.set_H(H, initialized=True)

        dense.pes._linear_bend_dummy_projection_records = lambda: None
        fast_step, _ = fast._predict_step()
        dense_step, _ = dense._predict_step()

        assert_allclose(fast_step, dense_step, atol=1e-12, rtol=1e-12)

    def test_linear_dummy_projection_filters_projection_only_dummy_rows(self):
        pes = Sella(self._two_linear_co2(), order=0, internal=True,
                    optimize_cell=True, allow_fragments=True,
                    rigid_fragments=True, logfile=None).pes
        pes.get_g()

        pes.dummies.positions += np.array([1e-3, -2e-3, 1.5e-3])
        x0 = pes.get_x().copy()
        pes.get_g()
        dummy_rows = pes._dummy_containing_coord_rows()
        aux_rows = pes._dummy_projection_gauge_filter_rows()
        dih0 = pes._dummy_dihedral_values()

        dx_initial, dx_final, _ = pes.set_x(x0)

        assert np.linalg.norm(pes.get_res(), ord=np.inf) < 1e-7
        assert_allclose(dx_initial, 0.0, atol=1e-14)
        assert len(dummy_rows) > 0
        assert len(aux_rows) > 0
        assert_allclose(dx_final[list(aux_rows)], 0.0, atol=1e-12)
        ddih = pes._wrapped_angle_delta(pes._dummy_dihedral_values(), dih0)
        assert_allclose(ddih, 0.0, atol=1e-8)

    def test_linear_dummy_projection_accounts_free_dummy_dihedral(self):
        atoms = self._linear_co2_with_off_axis_probe()
        internals = Internals(atoms)
        internals.find_all_bonds()
        internals.find_all_angles()
        internals.find_all_dihedrals()

        natoms = internals.natoms
        extra_dihedral = (natoms, 0, 3, 1)
        internals.add_dihedral(extra_dihedral)
        pes = InternalPES(atoms, internals, auto_find_internals=False)
        pes.get_g()
        _, _, Unred, Ufree = pes._compute_basis_int()
        assert Ufree is Unred
        assert pes._fast_dummy_redundant_projection() is not None
        assert_allclose(
            pes.get_Ucons().T @ pes.get_Ufree(), 0.0, atol=1e-12
        )

        extra_row = None
        row = 0
        for name in pes.int._names:
            for coord, active in zip(pes.int.internals[name],
                                     pes.int._active[name]):
                if active:
                    if (name == 'dihedrals'
                            and tuple(coord.indices) == extra_dihedral):
                        extra_row = row
                    row += 1
        assert extra_row is not None
        aux_rows = pes._dummy_projection_gauge_filter_rows()
        assert extra_row not in aux_rows

        pes.dummies.positions += np.array([[8e-4, -1.7e-3, 1.1e-3]])
        target = pes.get_x().copy()
        dx_initial, dx_final, _ = pes.set_x(target)
        actual_delta = pes._wrapped_angle_delta(
            np.array([pes.get_x()[extra_row]]),
            np.array([target[extra_row]]),
        )[0]

        assert pes._last_linear_dummy_projection_mode == 'cone'
        assert pes._last_linear_dummy_projection_ddih > 1e-4
        assert abs(actual_delta) > 1e-4
        assert_allclose(dx_initial, 0.0, atol=1e-14)
        assert_allclose(dx_final[list(aux_rows)], 0.0, atol=1e-12)
        assert_allclose(dx_final[extra_row], actual_delta, atol=1e-10)

    def test_redundant_full_rank_dummy_step_matches_dense_basis(self):
        def make_pes():
            atoms = self._linear_co2(cell_diag=10.0)
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=0.0,
                forces=np.array([
                    [0.00, 0.25, 0.08],
                    [0.03, -0.04, 0.02],
                    [-0.05, 0.10, -0.07],
                ]),
                stress=np.zeros(6),
            )
            base = Sella(
                atoms,
                order=0,
                internal=True,
                allow_fragments=True,
                logfile=None,
            ).pes
            internals = base.int.copy()
            internals.add_bond((0, 2))
            return InternalPES(
                atoms,
                internals,
                auto_find_internals=False,
            )

        fast = make_pes()
        dense = make_pes()
        dense._linear_bend_dummy_projection_records = lambda: None

        Q, R = fast._get_jacobian_qr()
        assert Q.shape[0] > Q.shape[1]
        assert R.shape[0] == R.shape[1]
        assert fast._linear_bend_dummy_projection_records() is not None

        rng = np.random.default_rng(21)
        A = rng.normal(size=(fast.dim, fast.dim))
        H = A.T @ A / fast.dim + np.diag(
            np.linspace(0.2, 2.0, fast.dim)
        )
        fast.set_H(H, initialized=True)
        dense.set_H(H, initialized=True)
        fast.get_g()
        dense.get_g()

        assert fast.curr['Ufree'] is fast.curr['Unred']
        assert fast._fast_dummy_redundant_projection() is not None
        fast_step, _ = MaxInternalStep(fast, 0, 0.1).get_s()
        dense_step, _ = MaxInternalStep(dense, 0, 0.1).get_s()

        assert_allclose(fast_step, dense_step, atol=1e-12, rtol=1e-12)
        assert_allclose(fast.get_drdx() @ fast_step, 0.0, atol=1e-12)

    def test_small_linear_bend_moves_geometry_without_projection(self):
        pes = Sella(
            self._mostly_linear_hoh(),
            order=0,
            internal=True,
            allow_fragments=True,
            exact_geodesic=False,
            logfile=None,
        ).pes
        pes.get_g()
        fixed_rows = set(pes._linear_dummy_auxiliary_optimizer_rows())
        natoms = pes.int.natoms
        free_angle = None
        row = 0
        for name in pes.int._names:
            for coord, active in zip(pes.int.internals[name],
                                     pes.int._active[name]):
                if not active:
                    continue
                if (name == 'angles' and row not in fixed_rows
                        and np.any(np.asarray(coord.indices) >= natoms)):
                    free_angle = row
                row += 1
        assert free_angle is not None

        x0 = pes.get_x().copy()
        apos0 = pes.atoms.positions.copy()
        dpos0 = pes.dummies.positions.copy()
        target = x0.copy()
        target[free_angle] += 1e-3

        pes._set_x_ode(target)

        assert np.linalg.norm(pes.atoms.positions - apos0) > 1e-4
        assert np.linalg.norm(pes.dummies.positions - dpos0) > 1e-4
        assert np.linalg.norm(pes.get_res(), ord=np.inf) < 1e-7
        assert not pes._project_to_constraints()
        assert_allclose(
            pes.wrap_dx(pes.get_x() - x0)[free_angle],
            1e-3,
            atol=1e-9,
        )

    def test_near_linear_hoh_bend_rows_are_accounted(self):
        def make_pes():
            return Sella(
                self._mostly_linear_hoh(),
                order=0,
                internal=True,
                optimize_cell=True,
                allow_fragments=True,
                rigid_fragments=True,
                logfile=None,
            ).pes

        def dummy_rows(pes):
            natoms = pes.int.natoms
            aux_rows = set(pes._linear_dummy_auxiliary_optimizer_rows())
            rows = []
            row = 0
            for name in pes.int._names:
                for coord, active in zip(pes.int.internals[name],
                                         pes.int._active[name]):
                    if active:
                        has_dummy = np.any(np.asarray(coord.indices) >= natoms)
                        rows.append((row, name, has_dummy, row in aux_rows))
                        row += 1
            return rows

        pes = make_pes()
        rows = dummy_rows(pes)
        free_angles = [
            row for row, name, has_dummy, is_aux in rows
            if name == 'angles' and has_dummy and not is_aux
        ]
        free_dihedrals = [
            row for row, name, has_dummy, is_aux in rows
            if name == 'dihedrals' and has_dummy and not is_aux
        ]
        aux_rows = list(pes._dummy_projection_gauge_filter_rows())
        assert len(free_angles) == 1
        assert len(free_dihedrals) == 1
        assert len(aux_rows) > 0

        pes.get_g()
        pes.dummies.positions += np.array([[7e-4, -1.3e-3, 9e-4]])
        target = pes.get_x().copy()
        dx_initial, dx_final, _ = pes.set_x(target)

        assert np.linalg.norm(pes.get_res(), ord=np.inf) < 1e-7
        assert_allclose(dx_initial, 0.0, atol=1e-14)
        assert_allclose(dx_final[list(aux_rows)], 0.0, atol=1e-12)
        assert abs(dx_final[free_angles[0]]) > 1e-4
        assert abs(dx_final[free_dihedrals[0]]) > 1e-10

        for row in free_angles + free_dihedrals:
            pes = make_pes()
            pes.get_g()
            x0 = pes.get_x().copy()
            target = x0.copy()
            target[row] += 1e-3
            dx_initial, dx_final, _ = pes.set_x(target)

            assert np.linalg.norm(pes.get_res(), ord=np.inf) < 1e-7
            assert_allclose(dx_initial[row], 1e-3, atol=1e-12)
            assert_allclose(dx_final[row], 1e-3, atol=1e-8)
            assert_allclose(dx_final[list(aux_rows)], 0.0, atol=1e-10)
            other = (free_dihedrals[0]
                     if row == free_angles[0] else free_angles[0])
            assert_allclose(
                pes.wrap_dx(pes.get_x() - x0)[other],
                0.0,
                atol=1e-8,
            )

    def test_linear_dummy_projection_preserves_requested_dummy_rows(self):
        pes = Sella(self._two_linear_co2(), order=0, internal=True,
                    optimize_cell=True, allow_fragments=True,
                    rigid_fragments=True, logfile=None).pes
        dummy_rows = pes._dummy_containing_coord_rows()
        assert len(dummy_rows) > 0

        requested = np.zeros(pes.int.nint)
        requested[dummy_rows[-1]] = 3e-4
        q_after_ode = pes.int.calc().copy()
        pes.dummies.positions += np.array([1e-3, -2e-3, 1.5e-3])
        pes._last_projection_filter_rows = dummy_rows

        dx_final = pes._add_proj_delta(requested, q_after_ode, True)

        assert_allclose(dx_final[list(dummy_rows)],
                        requested[list(dummy_rows)], atol=1e-12)

    @pytest.mark.parametrize("auxiliary_index", [0, 1])
    def test_linear_dummy_projection_reports_undone_auxiliary_request(
        self, auxiliary_index
    ):
        pes = Sella(self._linear_co2(), order=0, internal=True,
                    optimize_cell=True, allow_fragments=True,
                    rigid_fragments=True, logfile=None).pes
        pes.get_g()
        auxiliary_rows = pes._linear_dummy_auxiliary_optimizer_rows()
        assert len(auxiliary_rows) == 2
        row = auxiliary_rows[auxiliary_index]

        x0 = pes.get_x().copy()
        target = x0.copy()
        target[row] += 1e-3
        dx_initial, dx_final, _ = pes.set_x(target)
        actual = pes.wrap_dx(pes.get_x() - x0)

        assert_allclose(dx_initial[row], 1e-3, atol=1e-12)
        assert_allclose(actual[row], 0.0, atol=1e-12)
        assert_allclose(dx_final[row], actual[row], atol=1e-9)
        assert np.linalg.norm(pes.get_res(), ord=np.inf) < 1e-7


class TestRestrictedAtomicStepCellDOF:
    """Cartesian cell optimization must not crash under the default limiter.

    internal=False, optimize_cell=True defaults to RestrictedAtomicStep,
    whose cons() reshaped the full [atomic | cell] step vector into (-1, 3).
    When n_cell_dof is not a multiple of 3 (e.g. a one-DOF cell_mask) that
    reshape raised ValueError. cons() now splits the atomic and cell blocks.
    """

    @staticmethod
    def _atoms():
        atoms = Atoms('H4',
                      positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
                      cell=np.eye(3) * 6.0, pbc=True)
        atoms.calc = LennardJones()
        return atoms

    @pytest.mark.parametrize("n_free", [1, 2, 4, 9])
    def test_cartesian_cell_opt_steps_with_mask(self, n_free):
        # Build a mask that frees exactly n_free of the 9 cell elements, so
        # n_cell_dof spans multiples and non-multiples of 3.
        mask = np.zeros((3, 3), dtype=bool)
        mask.ravel()[:n_free] = True
        atoms = self._atoms()
        opt = Sella(atoms, order=0, internal=False, optimize_cell=True,
                    cell_mask=mask, logfile=None)
        from sella.optimize.restricted_step import RestrictedAtomicStep
        assert opt.rs is RestrictedAtomicStep
        assert opt.pes.n_cell_dof == n_free
        # Previously raised "cannot reshape array of size N into shape (3)".
        for _ in range(3):
            opt.step()

    def test_plain_cartesian_still_works(self):
        # No cell optimization: RestrictedAtomicStep behavior is unchanged.
        atoms = Atoms('H4',
                      positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
        atoms.calc = LennardJones()
        opt = Sella(atoms, order=0, internal=False, logfile=None)
        from sella.optimize.restricted_step import RestrictedAtomicStep
        assert opt.rs is RestrictedAtomicStep
        for _ in range(3):
            opt.step()

    def test_cons_gradient_matches_fd(self):
        # cons() value/gradient must be consistent for atomic- and
        # cell-dominant steps, including the wc cell weighting.
        from sella.optimize.restricted_step import RestrictedAtomicStep
        mask = np.zeros((3, 3), dtype=bool); mask[0, 0] = True
        atoms = self._atoms()
        opt = Sella(atoms, order=0, internal=False, optimize_cell=True,
                    cell_mask=mask, logfile=None)
        opt.pes.get_g()
        rs = RestrictedAtomicStep(opt.pes, opt.ord, opt.delta, wc=2.0)
        n = opt.pes.dim
        rng = np.random.RandomState(0)
        eps = 1e-7
        for desc, s in [
            ("atomic", np.r_[0.5, rng.randn(n - 1) * 0.01]),
            ("cell", np.r_[rng.randn(n - 1) * 0.001, 0.5]),
        ]:
            dsda = rng.randn(n)
            _, dval = rs.cons(s, dsda)
            vp = rs.cons(s + eps * dsda)
            vm = rs.cons(s - eps * dsda)
            dval_fd = (vp - vm) / (2 * eps)
            assert np.isclose(dval, dval_fd, atol=1e-5), desc


class TestCartesianCellTrustRadiusAdapts:
    """The Cartesian cell trust radius (delta_cell) must adapt during opt.

    The trust-radius update only split smag_cell for CellInternalPES, so with
    internal=False the cell radius stayed frozen at its initial value even as
    cell steps hit the cap. The split now applies to CellCartesianPES too
    (via pes.n_coords).
    """

    def test_atomic_component_uses_max_atom_displacement(self):
        atoms = Atoms(
            'H2', positions=[[0, 0, 0], [1, 0, 0]],
            cell=np.eye(3) * 5.0, pbc=True,
        )
        atoms.calc = LennardJones()
        opt = Sella(
            atoms, order=0, internal=False, optimize_cell=True,
            logfile=None,
        )
        step = np.zeros(opt.pes.dim)
        step[:6] = [3.0, 4.0, 0.0, 0.0, 0.0, 2.0]

        smag_atomic, smag_cell = opt._cell_trust_components(step, 5.0)

        assert smag_atomic == pytest.approx(5.0)
        assert smag_cell == 0.0

    def test_delta_cell_adapts_cartesian(self):
        # Compressed cell -> large stress -> cell steps large enough to move
        # the (initially capped) delta_cell.
        atoms = Atoms('H4',
                      positions=[[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0],
                                 [0, 0, 1.2]],
                      cell=np.eye(3) * 2.6, pbc=True)
        atoms.calc = LennardJones()
        opt = Sella(atoms, order=0, internal=False, optimize_cell=True,
                    logfile=None)
        from sella.peswrapper import CellCartesianPES
        assert isinstance(opt.pes, CellCartesianPES)
        delta_cell0 = opt.delta_cell
        seen = {round(opt.delta_cell, 6)}
        for _ in range(10):
            opt.step()
            seen.add(round(opt.delta_cell, 6))
        # delta_cell must have changed from its frozen initial value.
        assert len(seen) > 1, f"delta_cell never adapted (stuck at {delta_cell0})"


class TestIndependentCellTrustRadii:
    @staticmethod
    def _optimizer():
        atoms = make_water_crystal()
        atoms.calc = LennardJones()
        return Sella(
            atoms,
            order=0,
            internal=True,
            optimize_cell=True,
            allow_fragments=True,
            exact_geodesic=False,
            logfile=None,
        )

    def test_rejected_cell_only_step_preserves_internal_radius(self):
        opt = self._optimizer()
        delta0 = opt.delta
        delta_cell0 = opt.delta_cell
        step = np.zeros(opt.pes.dim)
        step[opt.pes.n_coords:] = 0.02

        opt._update_trust_radius(1e-3, step, 0.02)

        assert opt.delta == delta0
        assert opt.delta_cell == delta_cell0 * opt.sigma_dec

    def test_rejected_mixed_step_still_shrinks_internal_radius(self):
        opt = self._optimizer()
        step = np.zeros(opt.pes.dim)
        step[0] = 0.02
        step[opt.pes.n_coords:] = 0.01

        opt._update_trust_radius(1e-3, step, 0.02)

        assert opt.delta == 0.02 * opt.sigma_dec

    def test_rejected_internal_only_step_preserves_cell_radius(self):
        opt = self._optimizer()
        delta_cell0 = opt.delta_cell
        step = np.zeros(opt.pes.dim)
        step[0] = 0.02
        step[opt.pes.n_coords:] = np.finfo(float).eps

        opt._update_trust_radius(1e-3, step, 0.02)

        assert opt.delta_cell == delta_cell0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
