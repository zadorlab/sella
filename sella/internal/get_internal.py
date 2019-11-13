import numpy as np

from sella.constraints import get_constraints, merge_internal_constraints
from sella.internal.int_find import find_bonds, find_angles, find_dihedrals
from sella.internal.int_classes import CartToInternal

def get_internal(atoms, con_user=None, target_user=None, atol=15.,
                 intlast=None):
    if con_user is None:
        con_user = dict()

    if target_user is None:
        target_user = dict()

    atol *= np.pi / 180.

    bonds, nbonds, c10y = find_bonds(atoms)
    if intlast is None:
        dummies = None
        dinds = None
        angle_sums_old = None
    else:
        dummies = intlast.dummies
        dinds = intlast.dinds
        angle_sums_old = intlast.angle_sums

    # FIXME: This looks hideous, can it be improved?
    (angles, dummies, dinds, angle_sums,
            bcons, acons, dcons, adiffs) = find_angles(atoms, atol, bonds,
                    nbonds, c10y, dummies, dinds, angle_sums_old)

    dihedrals = find_dihedrals(atoms, atol, bonds, angles, nbonds,
                               c10y, dinds)

    bonds = np.vstack((bonds, bcons))
    angles = np.vstack((angles, acons))
    dihedrals = np.vstack((dihedrals, dcons))

    internal = CartToInternal(atoms,
                              bonds=bonds,
                              angles=angles,
                              dihedrals=dihedrals,
                              angle_sums=angle_sums,
                              angle_diffs=adiffs,
                              dummies=dummies)

    # Merge user-provided constraints with dummy atom constraints
    con, target = merge_internal_constraints(con_user, target_user, bcons,
                                             acons, dcons, adiffs)
    constraints = get_constraints(atoms, con, target, dummies, dinds,
                                  proj_trans=False, proj_rot=False)

    return internal, constraints, dummies
