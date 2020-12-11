from itertools import product
import pytest
import numpy as np

from ase import Atoms
from ase.build import molecule
from ase.calculators.tip3p import TIP3P, rOH, angleHOH

from sella import Sella, Constraints, Internals
from sella.internal import DuplicateConstraintError

water = molecule('H2O')
water.set_distance(0, 1, rOH)
water.set_distance(0, 2, rOH)
water.set_angle(1, 0, 2, angleHOH)
a = 3.106162559099496
rng = np.random.RandomState(0)

atoms_ref = Atoms()
for offsets in product(*((0, 1),) * 3):
    atoms = water.copy()
    for axis in ['x', 'y', 'z']:
        atoms.rotate(rng.random() * 360, axis)
    atoms.positions += a * np.asarray(offsets)
    atoms_ref += atoms


@pytest.mark.parametrize("internal,order",
                         [(True, 0),
                          (False, 0),
                          (True, 1),
                          (False, 1),
                          ])
def test_water_dimer(internal, order):
    internal = True
    order = 0
    rng = np.random.RandomState(1)

    atoms = atoms_ref.copy()
    atoms.calc = TIP3P()
    atoms.rattle(0.01, rng=rng)

    nwater = len(atoms) // 3
    cons = Constraints(atoms)
    for i in range(nwater):
        cons.fix_bond((3 * i, 3 * i + 1), target=rOH)
        cons.fix_bond((3 * i, 3 * i + 2), target=rOH)
        cons.fix_angle((3 * i + 1, 3 * i, 3 * i + 2), target=angleHOH)

    # Remove net translation and rotation
    try:
        cons.fix_translation()
    except DuplicateConstraintError:
        pass
    try:
        cons.fix_rotation()
    except DuplicateConstraintError:
        pass

    sella_kwargs = dict(
        order=order,
        trajectory='test.traj',
        eta=1e-6,
        delta0=1e-2,
    )
    if internal:
        sella_kwargs['internal'] = Internals(
            atoms, cons=cons, allow_fragments=True
        )
    else:
        sella_kwargs['constraints'] = cons
    opt = Sella(atoms, **sella_kwargs)

    opt.delta = 0.05
    opt.run(fmax=1e-3)
    print("First run done")

    atoms.rattle()
    opt.run(fmax=1e-3)

    Ufree = opt.pes.get_Ufree()
    g = opt.pes.get_g() @ Ufree
    np.testing.assert_allclose(g, 0, atol=1e-3)
    opt.pes.diag(gamma=1e-16)
    H = opt.pes.get_HL().project(Ufree)
    assert np.sum(H.evals < 0) == order, H.evals
