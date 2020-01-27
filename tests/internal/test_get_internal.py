import pytest

from ase.build import molecule

from sella.internal.get_internal import get_internal


@pytest.mark.parametrize("name", [('CH4'), ('C6H6')])
def test_get_internal(name):
    atoms = molecule(name)
    internal, constraints, dummies = get_internal(atoms)
    get_internal(atoms, conslast=constraints, dummies=dummies)
