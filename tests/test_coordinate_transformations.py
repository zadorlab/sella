import numpy as np

from ase.build import molecule
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator

from sella.internal import Internals
from sella.peswrapper import InternalPES


def _water_internals(atoms):
    internals = Internals(atoms)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    return internals


def test_internal_to_cartesian_step_reaches_target_coordinates():
    atoms = molecule('H2O')
    atoms.calc = LennardJones()
    pes = InternalPES(
        atoms,
        _water_internals(atoms),
        auto_find_internals=False,
        exact_geodesic=True,
    )
    initial_positions = atoms.positions.copy()
    target = pes.get_x() + np.array([0.01, -0.005, 0.02])

    pes.set_x(target)

    residual = pes.wrap_dx(pes.get_x() - target, origin=target)
    np.testing.assert_allclose(residual, 0.0, atol=2e-6)
    assert not np.allclose(atoms.positions, initial_positions)


def test_cartesian_hessian_converts_to_internal_basis():
    atoms = molecule('H2O')
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=0.0,
        forces=np.zeros((len(atoms), 3)),
    )
    pes = InternalPES(
        atoms,
        _water_internals(atoms),
        auto_find_internals=False,
    )
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(3 * len(atoms), 3 * len(atoms)))
    cartesian_hessian = matrix.T @ matrix
    jacobian = pes.int.jacobian()
    inverse_jacobian = np.linalg.pinv(jacobian)
    expected = inverse_jacobian.T @ cartesian_hessian @ inverse_jacobian

    actual = pes._convert_cartesian_hessian_to_internal(cartesian_hessian)

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
