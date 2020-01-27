import pytest

import numpy as np

from ase.build import molecule
from ase.calculators.emt import EMT

from sella.peswrapper import PES, InternalPES

@pytest.mark.parametrize("name,traj,cons",
                         [("CH4", "CH4.traj", None),
                          ("CH4", None, dict(bonds=((0, 1)))),
                          ("C6H6", None, None),
                          ])
def test_PES(name, traj, cons):
    tol = dict(atol=1e-6, rtol=1e-6)

    atoms = molecule(name)

    # EMT is *not* appropriate for molecules like this, but this is one
    # of the few calculators that is guaranteed to be available to all
    # users, and we don't need to use a physical PES to test this.
    atoms.calc = EMT()
    for MyPES in [PES, InternalPES]:
        pes = PES(atoms, trajectory=traj)

        pes.kick(0., diag=True, gamma=0.1)

        for i in range(2):
            pes.kick(-pes.get_g() * 0.01)

        assert pes.H is not None
        assert not pes.converged(0.)[0]
        assert pes.converged(1e100)
        A = pes.get_Ufree().T @ pes.get_Ucons()
        np.testing.assert_allclose(A, 0, **tol)

        pes.kick(-pes.get_g() * 0.001, diag=True, gamma=0.1)
