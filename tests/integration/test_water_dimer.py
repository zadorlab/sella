import pytest
import numpy as np

from ase import Atoms
from ase.build import molecule
from ase.collections import s22
from ase.calculators.tip3p import TIP3P, rOH, angleHOH

from sella import Sella

import matplotlib.pylab as plt

#atoms_ref = Atoms(['O', 'H', 'H'] * 2,
#                  positions=[[ 0.00,  0.00,  0.00],
#                             [-0.56,  0.38,  0.67],
#                             [ 0.77,  0.57, -0.03],
#                             [ 2.74,  0.14,  0.44],
#                             [ 2.43, -0.54, -0.15],
#                             [ 2.91, -0.31,  1.26]])
atoms_ref = molecule('H2O')
atoms_ref.cell = [3.106162559099496, 3.106162559099496, 3.106162559099496]
atoms_ref.pbc = False
atoms_ref *= (2, 2, 2)

@pytest.mark.parametrize("internal,order",
                         [(True, 0),
                          (False, 0),
                          (True, 1),
                          ])
def test_water_dimer(internal, order):
    rng = np.random.RandomState(1)

    #atoms = s22['Water_dimer']
    atoms = atoms_ref.copy()
    atoms.calc = TIP3P()
    atoms.rattle(0.01, rng=rng)

    nwater = len(atoms) // 3
    cons = dict(bonds=[], angles=[])
    for i in range(nwater):
        cons['bonds'] += [((3*i, 3*i + 1), rOH), ((3*i, 3*i + 2), rOH)]
        cons['angles'].append(((3*i + 1, 3*i, 3*i + 2), angleHOH))

    opt = Sella(atoms, constraints=cons, order=order, internal=internal, trajectory='test.traj', eta=1e-6)
    opt.run(fmax=0.00001)

    atoms.rattle()
    opt.run(fmax=0.00001)

    np.testing.assert_allclose(opt.pes.gfree, 0, atol=1e-3)
    opt.pes.diag(gamma=0.)
    assert np.sum(opt.pes.lams < 0) == order, opt.pes.lams
    #lams, vecs = np.linalg.eigh(opt.pes.Ufree.T @ opt.pes.H @ opt.pes.Ufree)
    #assert np.sum(lams < 0) == order, lams

    ##HL = opt.pes.H - opt.pes.cons.get_D(opt.pes.x.reshape((-1, 3))).ldot(opt.pes.llagrange)
    ##lams, vecs = np.linalg.eigh(opt.pes.Ufree.T @ HL @ opt.pes.Ufree)
    #assert np.sum(lams < 0) == order, lams

    #x0 = opt.pes.x.copy()
    #Ufree = opt.pes.Ufree.copy()
    #alphas = np.linspace(0., 0.1, 100)
    #es = np.zeros_like(alphas)
    #for i, alpha in enumerate(alphas):
    #    opt.pes.x = x0 + alpha * Ufree @ vecs[:, 0]
    #    es[i] = opt.pes.f
    #plt.plot(alphas, es)
    #plt.show()

if __name__ == '__main__':
    #test_water_dimer(True, 0)
    test_water_dimer(False, 0)
    #test_water_dimer(True, 1)
    #test_water_dimer(False, 1)
