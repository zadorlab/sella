import pytest

import numpy as np
from ase import Atoms
from ase.build import molecule
from scipy.spatial.transform import Rotation as SciPyRotation

import sella.internal as internal_module
from sella.internal import (
    Bond, Angle, Dihedral, Displacement, Rotation, Constraints, Internals
)


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


class TestStarImproperDihedrals:
    @staticmethod
    def _make_star(coordination, distorted=False, extended=False):
        phis = np.linspace(0.0, 2.7, coordination)
        positions = np.zeros((coordination + 1, 3))
        positions[1:, 0] = np.cos(phis)
        positions[1:, 1] = np.sin(phis)
        if distorted:
            positions[1:, 2] = np.linspace(-0.3, 0.4, coordination) ** 2
        if extended:
            positions = np.vstack((positions, [1.0, 1.0, 0.0]))

        atoms = Atoms(
            ['Xe'] + ['H'] * (len(positions) - 1), positions=positions
        )
        internals = Internals(atoms)
        for neighbor in range(1, coordination + 1):
            internals.add_bond((0, neighbor))
        if extended:
            internals.add_bond((1, coordination + 1))
        internals.find_all_angles()
        return internals

    @pytest.mark.parametrize('coordination', [3, 4, 5, 6])
    def test_planar_star_gets_minimal_full_rank_improper_fan(
        self, coordination
    ):
        internals = self._make_star(coordination)
        assert internals.nangles == coordination * (coordination - 1) // 2

        internals.find_all_dihedrals()

        assert internals.ndihedrals == coordination - 2
        rank = np.linalg.matrix_rank(internals.jacobian(), tol=1e-8)
        assert rank == 3 * (coordination + 1) - 6

    def test_improper_fan_depends_on_graph_not_planarity(self):
        planar = self._make_star(5)
        distorted = self._make_star(5, distorted=True)

        planar.find_all_dihedrals()
        distorted.find_all_dihedrals()

        planar_dihedrals = [
            tuple(coord.indices) for coord in planar.internals['dihedrals']
        ]
        distorted_dihedrals = [
            tuple(coord.indices) for coord in distorted.internals['dihedrals']
        ]
        assert planar_dihedrals == distorted_dihedrals
        assert len(planar_dihedrals) == 3

    def test_proper_dihedrals_avoid_extra_impropers(self):
        coordination = 5
        internals = self._make_star(coordination, extended=True)

        internals.find_all_dihedrals()

        assert internals.ndihedrals == coordination - 1
        rank = np.linalg.matrix_rank(internals.jacobian(), tol=1e-8)
        assert rank == 3 * (coordination + 2) - 6


class TestTRICs:
    """Tests for Translation-Rotation Internal Coordinates (TRICs)."""

    @pytest.mark.parametrize(
        "origin,target",
        [
            (
                np.array([np.pi - 0.05, 0.0, 0.0]),
                np.array([-np.pi + 0.05, 0.0, 0.0]),
            ),
            (
                np.array([np.pi - 0.05, 0.0, 0.0]),
                np.array([0.0, np.pi - 0.05, 0.0]),
            ),
            (
                np.array([2.8, 0.2, 0.0]),
                np.array([-0.1, -2.9, 0.3]),
            ),
        ],
    )
    def test_rotation_wrap_preserves_target_orientation(self, origin, target):
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        internals = Internals(atoms)
        internals.add_rotation((0, 1, 2))

        wrapped = internals.wrap(target - origin, origin=origin)
        represented_target = origin + wrapped
        expected = SciPyRotation.from_rotvec(target).as_matrix()
        actual = SciPyRotation.from_rotvec(represented_target).as_matrix()

        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert np.linalg.norm(wrapped) <= np.linalg.norm(target - origin)

    def test_diatomic_rotation_hvp_after_cumulative_rotation(self):
        atoms = Atoms(
            'H2', positions=[[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]],
            cell=np.eye(3) * 8.0, pbc=True,
        )
        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()
        internals.find_all_angles()
        internals.find_all_dihedrals()

        reference = atoms.positions.copy()
        axis = np.array([0.3, 0.7, 0.2])
        axis /= np.linalg.norm(axis)
        center = np.array([3.0, 2.0, 1.0])
        for angle in np.linspace(0.0, 2.5, 26):
            rotation = SciPyRotation.from_rotvec(axis * angle).as_matrix()
            atoms.positions = reference @ rotation.T + center
            internals.jacobian()

        tangent = np.arange(internals.ndof, dtype=float) - 2.5
        tangent /= np.linalg.norm(tangent)
        hessians = np.asarray(internals.hessian())
        expected_hvp = np.einsum('nij,j->ni', hessians, tangent)
        analytic = internals.hessian_rdot(tangent)
        if hasattr(analytic, 'toarray'):
            analytic = analytic.toarray()
        np.testing.assert_allclose(
            analytic, expected_hvp, atol=2e-8, rtol=2e-8
        )

        positions = atoms.positions.copy()
        quaternions = [
            coord.q_prev.copy()
            for coord in internals.internals['rotations']
        ]

        direction = tangent.reshape(-1, 3)
        delta_hessian = 1e-4
        for coord, quaternion in zip(
            internals.internals['rotations'], quaternions
        ):
            coord.q_prev = quaternion.copy()
            hessian = np.asarray(
                coord.calc_hessian(internals.light_atoms)
            ).reshape(internals.ndof, internals.ndof)
            values = []
            for sign in (1.0, 0.0, -1.0):
                displaced = atoms.copy()
                displaced.positions = positions + sign * delta_hessian * direction
                trial = coord.copy()
                trial.q_prev = quaternion.copy()
                values.append(trial.calc(displaced))
            numeric_curvature = (
                values[0] - 2 * values[1] + values[2]
            ) / delta_hessian**2
            analytic_curvature = tangent @ hessian @ tangent

            np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
            np.testing.assert_allclose(
                analytic_curvature, numeric_curvature,
                atol=2e-6, rtol=2e-6,
            )

    def test_near_degenerate_rotation_hessian_stays_bounded(self):
        bend = 1e-6
        refpos = np.array([
            [-2.0, 0.0, 0.0],
            [-0.5, bend, 0.0],
            [0.7, -0.7 * bend, 0.2 * bend],
            [1.8, 0.0, 0.0],
        ])
        theta = 0.2
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        positions = refpos @ rotation_matrix.T
        positions[1] += bend * np.array([0.01, -0.02, 0.03])
        atoms = Atoms('H4', positions=positions)
        internals = Internals(atoms)
        for axis in range(3):
            internals.add_rotation(Rotation((0, 1, 2, 3), axis, refpos))

        tangent = np.arange(internals.ndof, dtype=float) - 5.5
        tangent /= np.linalg.norm(tangent)
        hessians = np.asarray(internals.hessian())
        expected = np.einsum('nij,j->ni', hessians, tangent)
        actual = internals.hessian_rdot(tangent)
        if hasattr(actual, 'toarray'):
            actual = actual.toarray()

        assert np.linalg.norm(hessians) < 1.0
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)

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

    def test_partial_singleton_constraint_keeps_free_translation_axes(self):
        atoms = Atoms(
            'HHe',
            positions=[[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
            cell=np.eye(3) * 10.0,
            pbc=True,
        )
        constraints = Constraints(atoms)
        constraints.fix_translation(0, dim=0)
        internals = Internals(
            atoms, cons=constraints, allow_fragments=True
        )

        internals.find_all_bonds()
        internals.find_all_angles()
        internals.find_all_dihedrals()

        translations = {
            (tuple(coord.indices), coord.kwargs['dim'])
            for coord in internals.internals['translations']
        }
        assert translations == {
            ((0,), 0), ((0,), 1), ((0,), 2),
            ((1,), 0), ((1,), 1), ((1,), 2),
        }
        assert np.linalg.matrix_rank(internals.jacobian()) == 6

    def test_partial_rotation_constraint_keeps_other_rotation_axes(self):
        atoms = Atoms(
            'OH2',
            positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0],
                       [0.0, 0.95, 0.0]],
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        constraints = Constraints(atoms)
        constraints.fix_rotation((0, 1, 2), axis=0)
        internals = Internals(
            atoms, cons=constraints, allow_fragments=True
        )

        internals.find_all_bonds()
        internals.find_all_angles()
        internals.find_all_dihedrals()

        axes = sorted(
            coord.kwargs['axis']
            for coord in internals.internals['rotations']
        )
        assert axes == [0, 1, 2]
        assert np.all(np.isfinite(internals.jacobian()))

    def test_rotation_gradient_initializes_and_refreshes_quaternion(self):
        atoms = Atoms(
            'OH2',
            positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0],
                       [0.0, 0.95, 0.0]],
        )
        indices = (0, 1, 2)
        refpos = atoms.positions.copy()
        rotation = Rotation(indices, 0, refpos)

        gradient0 = rotation.calc_gradient(atoms)
        assert np.all(np.isfinite(gradient0))

        atoms.positions[1] += [0.03, 0.02, 0.01]
        atoms.positions[2] += [-0.01, 0.04, -0.02]
        gradient = rotation.calc_gradient(atoms)
        fresh = Rotation(indices, 0, refpos).calc_gradient(atoms)

        np.testing.assert_allclose(gradient, fresh, atol=1e-12, rtol=1e-12)

    def test_near_degenerate_rotation_gradient_uses_stabilized_subspace(self):
        bend = 1e-6
        refpos = np.array([
            [-2.0, 0.0, 0.0],
            [-0.5, bend, 0.0],
            [0.7, -0.7 * bend, 0.2 * bend],
            [1.8, 0.0, 0.0],
        ])
        theta = 0.2
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        positions = refpos @ rotation_matrix.T
        positions[1] += bend * np.array([0.01, -0.02, 0.03])
        atoms = Atoms('H4', positions=positions)
        rotation = Rotation((0, 1, 2, 3), 0, refpos)
        rotation.calc(atoms)
        q0 = rotation.q_prev.copy()
        analytic = rotation.calc_gradient(atoms)

        delta = 1e-9
        numeric = np.zeros_like(analytic)
        for i in range(4):
            for j in range(3):
                values = []
                for sign in (1.0, -1.0):
                    displaced = atoms.copy()
                    displaced.positions[i, j] += sign * delta
                    trial = Rotation((0, 1, 2, 3), 0, refpos)
                    trial.q_prev = q0.copy()
                    values.append(trial.calc(displaced))
                numeric[i, j] = (values[0] - values[1]) / (2 * delta)

        np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=1e-5)

    def test_partial_rotation_hvp_refreshes_stabilized_quaternion(self):
        atoms = Atoms(
            'OH2',
            positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0],
                       [0.0, 0.95, 0.0]],
        )
        internals = Internals(atoms)
        internals.add_rotation((0, 1, 2), axis=0)
        coord = internals.internals['rotations'][0]
        coord.calc(internals.light_atoms)
        old_q = coord.q_prev.copy()

        atoms.positions[:] = np.array([
            [0.1, -0.2, 0.3],
            [0.8, 0.5, -0.1],
            [-0.4, 0.7, 0.2],
        ])
        tangent = np.arange(9, dtype=float) / 7.0 - 0.5

        expected_coord = coord.copy()
        expected_coord.q_prev = old_q
        hessian = np.asarray(
            expected_coord.calc_hessian(internals.light_atoms)
        ).reshape(9, 9)
        expected = hessian @ tangent
        actual = internals.hessian_rdot(tangent)
        if hasattr(actual, 'toarray'):
            actual = actual.toarray()

        np.testing.assert_allclose(actual[0], expected, atol=1e-11, rtol=1e-11)

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

    def test_mst_weld_improper_dihedral_avoids_collinear_ordering(self):
        """Improper dihedrals from welded fragments must have valid planes."""
        atoms = Atoms(
            symbols=['O', 'H', 'H', 'O', 'H', 'H'],
            positions=[
                [9.70, 5.00, 5.00],
                [10.66, 5.00, 5.00],
                [9.70, 5.90, 5.00],
                [5.00, 5.00, 5.00],
                [5.96, 5.00, 5.00],
                [5.00, 5.96, 5.00],
            ],
            cell=np.eye(3) * 10.0,
            pbc=True,
        )

        ints = Internals(atoms, allow_fragments=False)
        ints.find_all_bonds()
        ints.find_all_angles()
        ints.find_all_dihedrals()

        assert np.isfinite(ints.jacobian()).all()

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
            # Should not warn if TRIC DOF calculation is correct. Ignore
            # ResourceWarnings, which can leak in from garbage collection of
            # unclosed files opened by unrelated tests (GC timing is not
            # deterministic across the suite).
            relevant = [x for x in w
                        if not issubclass(x.category, ResourceWarning)]
            assert len(relevant) == 0, (
                f"Unexpected warning: "
                f"{relevant[0].message if relevant else 'none'}"
            )

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

    def test_periodic_multi_image_bonds(self):
        """Angle construction with two bonds connecting the same atom pair
        through different periodic images.

        Regression test for a bug where Internal.__add__ picked the wrong
        orientation, producing degenerate angles like [4, 16, 4].
        """
        atoms = Atoms(
            symbols=['F', 'C', 'C', 'C', 'Br', 'C', 'Cl', 'C', 'Br', 'C',
                     'H', 'H', 'F', 'C', 'C', 'C', 'Br', 'C', 'Cl', 'C',
                     'Br', 'C', 'H', 'H'],
            positions=[
                [0.0000, 7.4288, 8.1335],
                [0.0000, 7.4288, 0.9168],
                [0.1080, 6.2248, 1.5800],
                [0.1073, 6.2349, 2.9650],
                [0.2550, 4.5837, 3.8585],
                [0.0000, 7.4288, 3.6780],
                [0.0000, 7.4288, 5.3959],
                [4.4338, 8.6227, 2.9650],
                [4.2861, 10.2738, 3.8585],
                [4.4331, 8.6328, 1.5800],
                [0.1903, 5.3016, 1.0250],
                [4.3508, 9.5560, 1.0250],
                [2.2705, 14.8576, 0.4253],
                [2.2705, 14.8576, 7.6420],
                [2.3785, 1.2040, 6.9788],
                [2.3778, 1.1939, 5.5938],
                [2.5255, 2.8451, 4.7003],
                [2.2705, 14.8576, 4.8809],
                [2.2705, 14.8576, 3.1629],
                [2.1633, 13.6637, 5.5938],
                [2.0155, 12.0125, 4.7003],
                [2.1626, 13.6536, 6.9788],
                [2.4608, 2.1272, 7.5338],
                [2.0802, 12.7304, 7.5338],
            ],
            cell=[[4.541073, 0.0, 0.0], [0.0, 14.857576, 0.0],
                  [0.0, 0.0, 8.55882]],
            pbc=True,
        )

        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds()
        ints.find_all_angles()

        for angle in ints.internals['angles']:
            if angle.indices[0] == angle.indices[2]:
                assert not np.array_equal(
                    angle.kwargs['ncvecs'][0], angle.kwargs['ncvecs'][1]
                ), (
                    f"Degenerate angle {angle.indices} with identical "
                    f"ncvecs {angle.kwargs['ncvecs']}"
                )

    def test_periodic_pair_search_chunking_matches_full(self, monkeypatch):
        atoms = Atoms(
            'H6',
            positions=[
                [0.10, 0.10, 0.10],
                [0.72, 0.10, 0.10],
                [2.85, 0.10, 0.10],
                [1.50, 1.20, 0.20],
                [1.50, 1.82, 0.20],
                [1.50, 2.87, 0.20],
            ],
            cell=np.diag([3.0, 3.0, 3.0]),
            pbc=True,
        )
        ii, jj = np.triu_indices(len(atoms), k=0)
        labels = -np.ones(len(atoms), dtype=np.int32)
        comps = np.arange(len(atoms), dtype=np.int32)
        rcov = np.full(len(atoms), 0.37)

        def as_tuples(links):
            return [
                (i, j, tuple(np.asarray(ts, dtype=np.int32)))
                for i, j, ts in links
            ]

        monkeypatch.setattr(
            internal_module, 'PERIODIC_PAIR_CHUNK_BYTES', 1 << 30
        )
        ints_full = Internals(atoms.copy(), allow_fragments=True)
        dists_full, ts_full = ints_full._periodic_pair_distances(ii, jj)
        bonds_full = as_tuples(
            ints_full._find_bonds_vectorized(labels, 1.25, rcov)
        )
        welds_full = as_tuples(ints_full._mst_welding_bonds(comps))

        monkeypatch.setattr(internal_module, 'PERIODIC_PAIR_CHUNK_BYTES', 1)
        ints_chunked = Internals(atoms.copy(), allow_fragments=True)
        dists_chunked, ts_chunked = ints_chunked._periodic_pair_distances(ii, jj)
        bonds_chunked = as_tuples(
            ints_chunked._find_bonds_vectorized(labels, 1.25, rcov)
        )
        welds_chunked = as_tuples(ints_chunked._mst_welding_bonds(comps))

        np.testing.assert_allclose(dists_chunked, dists_full)
        np.testing.assert_array_equal(ts_chunked, ts_full)
        assert bonds_chunked == bonds_full
        assert welds_chunked == welds_full


class TestInternalEquality:
    """Regression tests for Internal.__eq__.

    A refactor once restructured the equality logic into
    ``forward_indices AND reverse_indices AND (ncvecs...)``, which only ever
    matched palindromic index sequences -- so ``Bond((0,1)) == Bond((0,1))``
    returned False. That leaked into constraint dedup (duplicate constraints
    instead of replaced targets) and degenerate constructed coordinates.
    """

    def test_identity_equality(self):
        assert Bond((0, 1)) == Bond((0, 1))
        assert Angle((0, 1, 2)) == Angle((0, 1, 2))
        assert Dihedral((0, 1, 2, 3)) == Dihedral((0, 1, 2, 3))

    def test_reversed_equality(self):
        # Internals are direction-agnostic: a coordinate equals its reverse.
        assert Bond((0, 1)) == Bond((1, 0))
        assert Angle((0, 1, 2)) == Angle((2, 1, 0))
        assert Dihedral((0, 1, 2, 3)) == Dihedral((3, 2, 1, 0))

    def test_distinct_coordinates_not_equal(self):
        assert Bond((0, 1)) != Bond((0, 2))
        assert Angle((0, 1, 2)) != Angle((0, 2, 1))
        assert Bond((0, 1)) != Angle((0, 1, 2))

    def test_ncvecs_distinguish_periodic_images(self):
        # Same atom pair through different periodic images must not compare
        # equal, or __eq__ would over-merge distinct periodic coordinates.
        b0 = Bond((0, 1), ncvecs=[[0, 0, 0]])
        b1 = Bond((0, 1), ncvecs=[[1, 0, 0]])
        assert b0 != b1
        # Reversing an image bond negates its ncvecs, so the reverse is equal.
        assert b1 == b1.reverse()

    def test_fix_bond_replaces_target(self):
        # Constraint dedup relies on __eq__: refixing the same bond must
        # replace the target, not append a second constraint.
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        cons = Constraints(atoms)
        cons.fix_bond((0, 1), target=1.0)
        cons.fix_bond((0, 1), target=2.0)
        assert len(cons.internals['bonds']) == 1
        assert cons._targets['bonds'] == [2.0]

    def test_fix_bond_replaces_target_reversed(self):
        # Refixing via the reversed index order must also dedup.
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        cons = Constraints(atoms)
        cons.fix_bond((0, 1), target=1.0)
        cons.fix_bond((1, 0), target=2.0)
        assert len(cons.internals['bonds']) == 1


class TestForbid:
    """Regression tests for forbid_* keeping parallel lists in sync.

    forbid_translation removed from internals['translations'] but left
    _active['translations'] untouched, desyncing the parallel lists (calc
    and jacobian then disagreed on shape). _forbid_internal removed from
    forbidden[name] (a no-op) instead of internals[name], so forbidding an
    already-added bond left it active while also marking it forbidden.
    """

    def test_forbid_translation_syncs_active(self):
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        ic = Internals(atoms)
        ic.add_translation(0)
        ic.forbid_translation(0)
        assert len(ic.internals['translations']) == 0
        assert ic._active['translations'] == []
        assert ic.nint == 0
        # calc and jacobian must agree on the active-coordinate count.
        assert ic.calc().shape[0] == ic.jacobian().shape[0]

    def test_forbid_bond_removes_active_internal(self):
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        ic.add_bond((1, 2))
        ic.forbid_bond((0, 1))
        assert len(ic.internals['bonds']) == 1
        assert ic._active['bonds'] == [True]
        assert ic.internals['bonds'][0] == Bond((1, 2))
        # Forbidding should also purge the dedup key and register the ban.
        assert Bond((0, 1)) in ic.forbidden['bonds']
        assert ic.calc().shape[0] == ic.jacobian().shape[0]

    def test_forbid_bond_reversed_index_order(self):
        # forbid is direction-agnostic: (1,0) must remove an added (0,1).
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        ic.forbid_bond((1, 0))
        assert len(ic.internals['bonds']) == 0
        assert ic._active['bonds'] == []

    def test_forbid_then_add_raises(self):
        # A forbidden coordinate cannot subsequently be added.
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        ic = Internals(atoms)
        ic.forbid_bond((0, 1))
        with pytest.raises(Exception):
            ic.add_bond((0, 1))


class TestDisplacementEquality:
    """Displacement.__eq__ raised KeyError('refpos') against other classes.

    Coordinate.__eq__ returns NotImplemented (truthy) for a different class,
    so the guard fell through to other.kwargs['refpos'], which doesn't exist
    on non-Displacement coordinates.
    """

    def test_displacement_vs_other_class(self):
        d = Displacement(np.array([0, 1]), np.zeros((2, 3)), np.eye(6))
        assert (d == Bond((0, 1))) is False

    def test_displacement_identity(self):
        d0 = Displacement(np.array([0, 1]), np.zeros((2, 3)), np.eye(6))
        d1 = Displacement(np.array([0, 1]), np.zeros((2, 3)), np.eye(6))
        assert d0 == d1

    def test_displacement_distinct_W(self):
        # W changes the coordinate value, so it must factor into equality.
        idx, rp = np.array([0, 1]), np.zeros((2, 3))
        d0 = Displacement(idx, rp, np.eye(6))
        d1 = Displacement(idx, rp, 2 * np.eye(6))
        assert d0 != d1

    def test_displacement_distinct_refpos(self):
        idx = np.array([0, 1])
        d0 = Displacement(idx, np.zeros((2, 3)), np.eye(6))
        d1 = Displacement(idx, np.ones((2, 3)), np.eye(6))
        assert d0 != d1


class TestReversedAddDedup:
    """add_* must reject a reversed duplicate of an existing coordinate.

    Internal.__eq__ treats reversed coordinates as equal, but _add_internal
    keyed _internals_set on the raw (orientation-sensitive) indices, so
    add_bond((1,0)) after add_bond((0,1)) slipped through as a second bond.
    The dedup key is now canonical across orientation.
    """

    def test_reversed_bond_rejected(self):
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        with pytest.raises(Exception):
            ic.add_bond((1, 0))
        assert len(ic.internals['bonds']) == 1

    def test_reversed_angle_rejected(self):
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        ic = Internals(atoms)
        ic.add_angle((0, 1, 2))
        with pytest.raises(Exception):
            ic.add_angle((2, 1, 0))
        assert len(ic.internals['angles']) == 1

    def test_reversed_dihedral_rejected(self):
        atoms = Atoms('H4',
                      positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        ic = Internals(atoms)
        ic.add_dihedral((0, 1, 2, 3))
        with pytest.raises(Exception):
            ic.add_dihedral((3, 2, 1, 0))
        assert len(ic.internals['dihedrals']) == 1

    def test_distinct_bond_still_allowed(self):
        # The canonical key must not over-merge genuinely different bonds.
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        ic.add_bond((1, 2))
        assert len(ic.internals['bonds']) == 2

    def test_reversed_forbid_purges_dedup_key(self):
        # forbid via reversed order must clear the canonical key so the
        # coordinate can be re-added later without a stale-dedup false hit.
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        ic.forbid_bond((1, 0))
        assert ic._internals_set['bonds'] == set()


class TestStructuralCacheInvalidation:
    """Topology changes must invalidate the vectorized caches.

    add_*/forbid_* mutate the coordinate lists, but the batched arrays and
    tvecs cache were reused, so calc() returned a stale-length vector and
    jacobian() crashed on the size mismatch. _invalidate_structure() now
    clears them (plus the position-keyed _cache, which a topology change can
    otherwise leave untouched when the active mask/positions are unchanged).
    """

    def test_add_bond_after_calc(self):
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        assert ic.calc().shape == (1,)
        ic.add_bond((1, 2))
        assert ic.nint == 2
        assert ic.calc().shape == (2,)
        # jacobian must not crash on a stale batched-array size mismatch.
        assert ic.jacobian().shape[0] == 2

    def test_forbid_then_add_refreshes_cache(self):
        # forbid+add can leave the active mask and positions unchanged, so
        # only _invalidate_structure (not _cache_check) refreshes the values.
        atoms = Atoms('H4',
                      positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        assert np.allclose(ic.calc(), [1.0])
        ic.forbid_bond((0, 1))
        ic.add_bond((2, 3))
        # bond (2,3) has length 2.0; a stale cache would still report 1.0.
        assert np.allclose(ic.calc(), [2.0])

    def test_add_periodic_image_bond_refreshes_tvecs(self):
        # A new periodic-image bond must not evaluate with stale (zero) tvecs.
        atoms = Atoms('H2', positions=[[0, 0, 0], [0.5, 0, 0]],
                      cell=np.eye(3) * 3.0, pbc=True)
        ic = Internals(atoms)
        ic.add_bond((0, 1), ncvecs=[[0, 0, 0]])
        assert np.allclose(ic.calc(), [0.5])
        ic.add_bond((0, 1), ncvecs=[[1, 0, 0]])
        # image bond length = |0.5 + 3.0| = 3.5; stale tvecs would give 0.5.
        vals = ic.calc()
        assert vals.shape == (2,)
        assert np.allclose(sorted(vals), [0.5, 3.5])
        assert ic.jacobian().shape[0] == 2


class TestFindAllBondsIdempotent:
    """find_all_bonds() must be idempotent with allow_fragments=True.

    Fragment translations/rotations were added unguarded, so a second
    find_all_bonds() (or passing a pre-populated fragment Internals into
    InternalPES with auto_find_internals=True) raised DuplicateInternalError.
    """

    def _two_waters(self):
        return Atoms(
            symbols=['O', 'H', 'H', 'O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0],
                [10.0, 0.0, 0.0], [10.96, 0.0, 0.0], [10.0, 0.96, 0.0],
            ],
        )

    def _linear_co2(self, pbc=False):
        atoms = Atoms(
            'OCO',
            positions=[[1.0, 2.0, 0.0],
                       [2.0, 2.0, 0.0],
                       [3.0, 2.0, 0.0]],
            cell=np.eye(3) * 6.0,
            pbc=pbc,
        )
        return atoms

    def test_repeated_find_all_bonds(self):
        ic = Internals(self._two_waters(), allow_fragments=True)
        ic.find_all_bonds()
        ic.find_all_angles()
        ic.find_all_dihedrals()
        counts1 = {k: len(v) for k, v in ic.internals.items()}
        # Second pass must not raise and must not duplicate coordinates.
        ic.find_all_bonds()
        ic.find_all_angles()
        ic.find_all_dihedrals()
        counts2 = {k: len(v) for k, v in ic.internals.items()}
        assert counts1 == counts2

    def test_internalpes_accepts_prepopulated_fragments(self):
        from ase.calculators.lj import LennardJones
        from sella.peswrapper import InternalPES
        atoms = self._two_waters()
        atoms.calc = LennardJones()
        ic = Internals(atoms, allow_fragments=True)
        ic.find_all_bonds()
        ic.find_all_angles()
        ic.find_all_dihedrals()
        # Default auto_find_internals=True re-runs find_all_bonds on the copy.
        InternalPES(atoms, ic)

    def test_internalpes_accepts_prepopulated_linear_bend_dummies(self):
        from ase.calculators.lj import LennardJones
        from sella.peswrapper import InternalPES
        atoms = self._linear_co2()
        atoms.calc = LennardJones()
        ic = Internals(atoms)
        ic.find_all_bonds()
        ic.find_all_angles()
        ic.find_all_dihedrals()
        assert ic.ndummies == 1
        # Default auto_find_internals=True re-runs find_all_bonds on the copy.
        InternalPES(atoms, ic)

    def test_pes_accepts_fragmented_linear_bend_dummies(self):
        from ase.calculators.lj import LennardJones
        from sella.peswrapper import InternalPES, CellInternalPES
        atoms = self._linear_co2(pbc=True)
        atoms.calc = LennardJones()
        ic = Internals(atoms, allow_fragments=True)
        ic.find_all_bonds()
        ic.find_all_angles()
        ic.find_all_dihedrals()
        assert ic.ndummies == 1

        for pes_cls in (InternalPES, CellInternalPES):
            pes = pes_cls(atoms, ic, auto_find_internals=True)
            assert len(pes.int.internals['translations']) == 3
            assert len(pes.int.internals['rotations']) == 3

    def test_internalpes_copy_does_not_share_linear_dummy_state(self):
        from ase.calculators.lj import LennardJones
        from sella.peswrapper import CellInternalPES
        atoms = self._linear_co2(pbc=True)
        atoms.calc = LennardJones()
        ic = Internals(atoms)
        dinds0 = ic.dinds.copy()

        pes = CellInternalPES(atoms, ic, auto_find_internals=True)

        assert pes.int.ndummies == 1
        assert ic.ndummies == 0
        np.testing.assert_array_equal(ic.dinds, dinds0)

    def test_supplied_dummy_indices_must_be_appended(self):
        atoms = self._linear_co2()
        dummies = Atoms('X', positions=[[2.0, 3.0, 0.0]])

        with pytest.raises(ValueError, match='appended dummy block'):
            Internals(
                atoms,
                dummies=dummies,
                dinds=np.array([-1, 0, -1], dtype=np.int32),
            )

    def test_dummy_constraint_dedup_keeps_metadata_aligned(self):
        atoms = self._linear_co2()
        dummies = Atoms('X', positions=[[2.0, 3.0, 0.0]])
        dinds = np.array([-1, 3, -1], dtype=np.int32)
        cons = Constraints(atoms, dummies=dummies, dinds=dinds)
        cons.fix_translation((1,), dim=0, target=1.0)
        cons.fix_translation((1, 3), dim=0, target=1.0)

        cons.add_dummy_to_internals(1)

        assert len(cons.internals['translations']) == 1
        assert len(cons._active['translations']) == 1
        assert cons._targets['translations'] == [1.0]
        assert cons._kind['translations'] == ['eq']

    def test_dummy_constraint_dedup_rejects_conflicting_targets(self):
        atoms = self._linear_co2()
        dummies = Atoms('X', positions=[[2.0, 3.0, 0.0]])
        dinds = np.array([-1, 3, -1], dtype=np.int32)
        cons = Constraints(atoms, dummies=dummies, dinds=dinds)
        cons.fix_translation((1,), dim=0, target=1.0)
        cons.fix_translation((1, 3), dim=0, target=2.0)

        with pytest.raises(
            internal_module.DuplicateConstraintError,
            match='conflicting constraints',
        ):
            cons.add_dummy_to_internals(1)

    def test_dummy_constraint_dedup_rejects_close_conflicting_targets(self):
        atoms = self._linear_co2()
        dummies = Atoms('X', positions=[[2.0, 3.0, 0.0]])
        dinds = np.array([-1, 3, -1], dtype=np.int32)
        cons = Constraints(atoms, dummies=dummies, dinds=dinds)
        cons.fix_translation((1,), dim=0, target=1.0)
        cons.fix_translation((1, 3), dim=0, target=1.0 + 1e-8)

        with pytest.raises(
            internal_module.DuplicateConstraintError,
            match='conflicting constraints',
        ):
            cons.add_dummy_to_internals(1)


class TestPeriodicConstraintUnwrap:
    """Periodic constraints must survive allow_fragments unwrapping."""

    def test_fixed_angle_ncvec_remapped_after_fragment_unwrap(self):
        cell = np.array([[10.0, 0.0, 0.0],
                         [2.0, 10.0, 0.0],
                         [0.5, 0.8, 10.0]])
        atoms = Atoms(
            'OH2',
            positions=[
                [9.70, 5.00, 5.00],
                [0.66, 5.00, 5.00],
                [9.70, 5.90, 5.00],
            ],
            cell=cell,
            pbc=True,
        )
        cons = Constraints(atoms)
        cons.fix_angle((1, 0, 2), mic=True)
        np.testing.assert_allclose(cons.residual(), [0.0], atol=1e-12)
        assert np.linalg.norm(cons.cell_jacobian()) > 0.1

        ints = Internals(atoms, cons=cons, allow_fragments=True)
        ints.find_all_bonds()

        assert np.ptp(atoms.positions, axis=0).max() < 2.0
        np.testing.assert_allclose(
            cons.internals['angles'][0].kwargs['ncvecs'], 0
        )
        np.testing.assert_allclose(cons.residual(), [0.0], atol=1e-12)
        np.testing.assert_allclose(cons.cell_jacobian(), 0.0, atol=1e-12)
        np.testing.assert_allclose(ints.cons.residual(), [0.0], atol=1e-12)

    def test_copy_does_not_share_periodic_constraint_coords(self):
        atoms = Atoms(
            'OH2',
            positions=[
                [9.70, 5.00, 5.00],
                [0.66, 5.00, 5.00],
                [9.70, 5.90, 5.00],
            ],
            cell=np.eye(3) * 10.0,
            pbc=True,
        )
        cons = Constraints(atoms)
        cons.fix_angle((1, 0, 2), mic=True)
        ints = Internals(atoms, cons=cons, allow_fragments=True)
        original = cons.internals['angles'][0]
        original_ncvecs = original.kwargs['ncvecs'].copy()

        copied = ints.copy()
        assert copied.cons.internals['angles'][0] is copied.internals['angles'][0]
        assert copied.cons.internals['angles'][0] is not original
        copied.find_all_bonds()

        np.testing.assert_allclose(original.kwargs['ncvecs'], original_ncvecs)
        np.testing.assert_allclose(
            ints.cons.internals['angles'][0].kwargs['ncvecs'],
            original_ncvecs,
        )
        np.testing.assert_allclose(
            copied.cons.internals['angles'][0].kwargs['ncvecs'], 0
        )

    def test_fixed_dihedral_ncvec_remapped_after_fragment_unwrap(self):
        atoms = Atoms(
            'H4',
            positions=[
                [3.70, 0.00, 0.00],
                [0.30, 0.00, 0.00],
                [0.90, 0.30, 0.00],
                [1.40, 0.30, 0.40],
            ],
            cell=np.eye(3) * 4.0,
            pbc=True,
        )
        cons = Constraints(atoms)
        cons.fix_dihedral((0, 1, 2, 3), mic=True)
        np.testing.assert_allclose(cons.residual(), [0.0], atol=1e-12)

        ints = Internals(atoms, cons=cons, allow_fragments=True)
        ints.find_all_bonds()

        assert np.ptp(atoms.positions, axis=0).max() < 2.0
        np.testing.assert_allclose(cons.residual(), [0.0], atol=1e-12)
        np.testing.assert_allclose(ints.cons.residual(), [0.0], atol=1e-12)


class TestFixCartesian:
    """merge_ase_constraint must read ASE FixCartesian across API versions.

    ASE changed FixCartesian around 3.23. Newer versions store constrained
    atoms in .index (an array) with a non-inverted mask (True = fixed); older
    versions (<=3.22) stored a scalar in .a and inverted the mask at
    construction (stored True = free). The user-facing convention -- mask True
    means fixed -- is identical, so both must normalize to the same result.
    The pre-fix code assumed the old .a/inverted-mask shape but then ran on
    new ASE, so a partial mask raised AttributeError and a full mask added
    nothing.
    """

    @staticmethod
    def _as_old_api(fc, a, user_mask):
        """Rewrite a real FixCartesian to the pre-3.23 attribute shape."""
        del fc.index                       # drop new-API array attr
        fc.a = a                           # scalar atom index
        fc.mask = ~np.asarray(user_mask, bool)  # stored True = free
        return fc

    def test_partial_mask(self):
        from ase.constraints import FixCartesian
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        cons = Constraints(atoms)
        cons.merge_ase_constraint(FixCartesian([0], mask=(True, False, False)))
        trans = cons.internals['translations']
        assert len(trans) == 1
        assert list(trans[0].indices) == [0]
        assert trans[0].kwargs['dim'] == 0

    def test_full_mask_multiple_atoms(self):
        from ase.constraints import FixCartesian
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        cons = Constraints(atoms)
        cons.merge_ase_constraint(
            FixCartesian([0, 2], mask=(True, True, True))
        )
        # Each atom fixed independently in all 3 dims -> 6 translations.
        assert len(cons.internals['translations']) == 6

    def test_all_relaxed_mask_adds_nothing(self):
        from ase.constraints import FixCartesian
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        cons = Constraints(atoms)
        cons.merge_ase_constraint(FixCartesian([0], mask=(False, False, False)))
        assert len(cons.internals['translations']) == 0

    def test_fixatoms_via_atoms_constraint(self):
        # FixAtoms fixes all 3 cartesian dims of each listed atom.
        from ase.constraints import FixCartesian
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        cons = Constraints(atoms)
        cons.merge_ase_constraint(FixCartesian([1], mask=(True, True, True)))
        assert len(cons.internals['translations']) == 3
        assert all(list(t.indices) == [1]
                   for t in cons.internals['translations'])

    def test_old_api_partial_mask(self):
        # Simulate ASE <=3.22: scalar .a, no .index, inverted stored mask.
        from ase.constraints import FixCartesian
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        old = self._as_old_api(FixCartesian([1], mask=(True, False, False)),
                               1, (True, False, False))
        assert not hasattr(old, 'index')
        cons = Constraints(atoms)
        cons.merge_ase_constraint(old)
        trans = cons.internals['translations']
        assert len(trans) == 1
        assert list(trans[0].indices) == [1]
        assert trans[0].kwargs['dim'] == 0

    def test_old_api_full_mask(self):
        from ase.constraints import FixCartesian
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        old = self._as_old_api(FixCartesian([2], mask=(True, True, True)),
                               2, (True, True, True))
        cons = Constraints(atoms)
        cons.merge_ase_constraint(old)
        assert len(cons.internals['translations']) == 3

    def test_old_api_all_relaxed(self):
        from ase.constraints import FixCartesian
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        old = self._as_old_api(FixCartesian([0], mask=(False, False, False)),
                               0, (False, False, False))
        cons = Constraints(atoms)
        cons.merge_ase_constraint(old)
        assert len(cons.internals['translations']) == 0


class TestFixInternalsMerge:
    """merge_ase_constraint(FixInternals) parses the ASE data format."""

    def test_fixinternals_bond(self):
        from ase.constraints import FixInternals
        atoms = Atoms('H4',
                      positions=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]])
        cons = Constraints(atoms)
        fi = FixInternals(bonds=[[1.0, [0, 1]]])
        cons.merge_ase_constraint(fi)
        assert len(cons.internals['bonds']) == 1
