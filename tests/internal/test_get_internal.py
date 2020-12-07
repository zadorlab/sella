import pytest

from ase.build import molecule

from sella.internal import Internals


@pytest.mark.parametrize("name", [('CH4'), ('C6H6')])
def test_get_internal(name):
    atoms = molecule(name)
    internal = Internals(atoms)
    internal.find_all_bonds()
    internal.find_all_angles()
    internal.find_all_dihedrals()
