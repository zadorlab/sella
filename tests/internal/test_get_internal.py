import pytest

import numpy as np
from ase import Atoms
from ase.build import molecule

from sella.internal import Internals


def res(pos: np.ndarray, internal: Internals) -> np.ndarray:
    internal.atoms.positions = pos.reshape((-1, 3))
    return internal.calc()


def jacobian(pos: np.ndarray, internal: Internals) -> np.ndarray:
    internal.atoms.positions = pos.reshape((-1, 3))
    return internal.jacobian()


def hessian(pos: np.ndarray, internal: Internals) -> np.ndarray:
    internal.atoms.positions = pos.reshape((-1, 3))
    return internal.hessian()


@pytest.mark.parametrize("name", ['CH4', 'C6H6', 'C2H6'])
def test_get_internal(name: str) -> None:
    atoms = molecule(name)
    internal = Internals(atoms)
    internal.find_all_bonds()
    internal.find_all_angles()
    internal.find_all_dihedrals()
    jac = internal.jacobian()
    hess = internal.hessian()

    x0 = atoms.positions.ravel().copy()
    x = x0.copy()
    dx = 1e-4

    jac_numer = np.zeros_like(jac)
    hess_numer = np.zeros_like(hess)
    for i in range(len(x)):
        x[i] += dx
        atoms.positions = x.reshape((-1, 3))
        res_plus = internal.calc()
        jac_plus = internal.jacobian()
        x[i] = x0[i] - dx
        atoms.positions = x.reshape((-1, 3))
        res_minus = internal.calc()
        jac_minus = internal.jacobian()
        x[i] = x0[i]
        atoms.positions = x.reshape((-1, 3))
        jac_numer[:, i] = (internal.wrap(res_plus - res_minus)) / (2 * dx)
        hess_numer[:, i, :] = (jac_plus - jac_minus) / (2 * dx)
    np.testing.assert_allclose(jac, jac_numer, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(hess, hess_numer, rtol=1e-7, atol=1e-7)


class TestTRICs:
    """Tests for Translation-Rotation Internal Coordinates (TRICs)."""

    def test_tric_single_atom_fragment(self):
        """Test TRICs with a single-atom fragment (should not raise assertion).

        This tests the bug fix for the line ordering issue in find_all_bonds()
        where single atoms would incorrectly get rotation ICs added.
        """
        # Bi(NO3)3 cluster from the bug report - Bi is a single atom, NO3 are fragments
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )
        # Use scale=1.0 to ensure fragments are detected (not bonded via 1.25 scale)
        ints = Internals(atoms, allow_fragments=True)
        # This should not raise an assertion error even though Bi is a single atom
        ints.find_all_bonds(scale=1.0)
        ints.find_all_angles()
        ints.find_all_dihedrals()

        # Should have translations (including for the single Bi atom)
        assert len(ints.internals['translations']) > 0

        # Rotations should only be for multi-atom fragments (NO3 groups)
        # Bi should NOT have rotation ICs
        for rot in ints.internals['rotations']:
            assert len(rot.indices) >= 2, "Rotation IC added to single atom!"

    def test_tric_scale_parameter(self):
        """Test that scale parameter affects bond detection."""
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )

        # With small scale, should have fragments (TRICs added)
        ints_small = Internals(atoms, allow_fragments=True)
        ints_small.find_all_bonds(scale=1.0)
        n_trans_small = len(ints_small.internals['translations'])
        n_rot_small = len(ints_small.internals['rotations'])

        # With large scale, might connect everything (no TRICs)
        ints_large = Internals(atoms, allow_fragments=True)
        ints_large.find_all_bonds(scale=1.5)
        n_trans_large = len(ints_large.internals['translations'])
        n_rot_large = len(ints_large.internals['rotations'])

        # Smaller scale should result in more fragments (more TRICs)
        assert n_trans_small >= n_trans_large
        assert n_rot_small >= n_rot_large

    def test_tric_two_separate_molecules(self):
        """Test TRICs with two well-separated molecules."""
        # Two water molecules far apart - use explicit element list for clarity
        atoms = Atoms(
            symbols=['O', 'H', 'H', 'O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],     # O (first molecule)
                [0.96, 0.0, 0.0],    # H
                [0.0, 0.96, 0.0],    # H
                [10.0, 0.0, 0.0],    # O (second molecule, far away)
                [10.96, 0.0, 0.0],   # H
                [10.0, 0.96, 0.0],   # H
            ]
        )

        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds()
        ints.find_all_angles()

        # Should have 2 fragments, so 2 translation sets (6 coords) and 2 rotation sets (6 coords)
        assert len(ints.internals['translations']) == 6  # 3 per fragment × 2 fragments
        assert len(ints.internals['rotations']) == 6     # 3 per fragment × 2 fragments

    def test_validate_basis_with_trics(self):
        """Test that validate_basis correctly calculates DOF with TRICs."""
        # Two water molecules far apart - use explicit element list for clarity
        atoms = Atoms(
            symbols=['O', 'H', 'H', 'O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],     # O (first molecule)
                [0.96, 0.0, 0.0],    # H
                [0.0, 0.96, 0.0],    # H
                [10.0, 0.0, 0.0],    # O (second molecule, far away)
                [10.96, 0.0, 0.0],   # H
                [10.0, 0.96, 0.0],   # H
            ]
        )

        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds()
        ints.find_all_angles()

        # With TRICs, expect 3N = 18 DOF (translations+rotations span full space)
        # validate_basis should not raise warnings for TRICs
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ints.validate_basis()
            # Should not warn if TRIC DOF calculation is correct
            assert len(w) == 0, f"Unexpected warning: {w[0].message if w else 'none'}"

    def test_tric_optimization_convergence(self):
        """Test that optimization with TRICs converges (ODE doesn't fail).

        This tests the fix for the ODE convergence issue with ill-conditioned
        Jacobians that arise from TRICs.
        """
        from ase.calculators.lj import LennardJones
        from sella import Sella

        # Bi(NO3)3 cluster - a real-world TRIC test case
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )
        atoms.calc = LennardJones()

        # Use TRICs with small scale to ensure fragments are detected
        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds(scale=1.0)
        ints.find_all_angles()
        ints.find_all_dihedrals()

        # This should not raise RuntimeError about ODE convergence
        opt = Sella(atoms, internal=ints)
        # Just run a few steps to verify ODE works
        opt.run(fmax=1.0, steps=5)
