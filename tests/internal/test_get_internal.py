import pytest

import numpy as np
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
