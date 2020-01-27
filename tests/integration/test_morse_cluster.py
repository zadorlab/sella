import pytest
import numpy as np

from ase import Atoms
from ase.calculators.morse import MorsePotential
from ase.units import kB

from sella import Sella

@pytest.mark.parametrize("internal,order",
                         [(False, 0),
                          (False, 1),
                          (True, 0),
                          (True, 1),
                          ])
def test_morse_cluster(internal, order, trajectory=None):
    rng = np.random.RandomState(42)

    nat = 4
    atoms = Atoms(['Xe'] * nat, rng.normal(size=(nat, 3), scale=2))
    # parameters from DOI: 10.1515/zna-1987-0505
    atoms.calc = MorsePotential(alpha=226.9 * kB, r0=4.73, rho0=4.73*1.099)

    opt = Sella(atoms, order=order, internal=internal, trajectory=trajectory)
    opt.run(fmax=1e-3)

    Ufree = opt.pes.get_Ufree()
    np.testing.assert_allclose(opt.pes.get_g() @ Ufree, 0, atol=5e-3)
    opt.pes.diag(gamma=0.)
    H = opt.pes.get_HL().project(Ufree)
    assert np.sum(H.evals < 0) == order, H.evals


if __name__ == '__main__':
    test_morse_cluster(True, 0, trajectory='test.traj')
    test_morse_cluster(False, 0, trajectory='test.traj')
    test_morse_cluster(True, 1, trajectory='test.traj')
    test_morse_cluster(False, 1, trajectory='test.traj')
