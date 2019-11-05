import numpy as np

from ase.constraints import (FixCartesian, FixAtoms, FixBondLengths,
                             FixInternals)

from sella.constraints import _ase_constraints_to_dict
from sella.internal.int_find import find_bonds, find_angles, find_dihedrals
from sella.internal.int_classes import CartToInternal, Constraints, D2q

# Merges user-provided constraints dict with Atoms object constraints
def merge_user_constraints(atoms, con_user):
    if atoms.constraints:
        con = _ase_constraints_to_dict(atoms.constraints)
    else:
        return con_user

    for key, val in con_user.items():
        if key not in con:
            con[key] = val
        else:
            con[key] += val
    return con


# Merges user-provided constraints with automatically determined internal
# constraints
def merge_internal_constraints(atoms, con_user, bcons, acons, dcons, adiffs):
    con = merge_user_constraints(atoms, con_user)

    fix = []
    bonds = []
    angles = []
    dihedrals = []
    angle_diffs = []
    target = []

    # Fixed atoms
    for val in con.get('fix', []):
        if len(val) == 2:
            fix.append(np.array(val, dtype=np.int32))
            target.append(atoms.positions[val])
        else:
            for i in range(3):
                fix.append(np.array((val, i), dtype=np.int32))
                target.append(atoms.positions[val, i])

    # Bonds
    for indices, target in con.get('bonds', []):
        if isinstance(indices, tuple):
            bonds.append(np.array(indices, dtype=np.int32))
            target.append(target)
        else:
            bonds.append(np.array((indices, target), dtype=np.int32))
            targets.append(atoms.get_distance(indices, target))
    for indices in bcons:
        bonds.append(np.array(indices, dtype=np.int32))
        targets.append(1.)

    # Angles
    for arg in con.get('angles', []):
        if len(arg) == 2:
            angles.append(np.array(arg[0], dtype=np.int32))
            target.append(arg[1])
        else:
            angles.append(np.array(arg, dtype=np.int32))
            target.append(atoms.get_angle(*arg) * np.pi / 180.)
    for indices in acons:
        angles.append(np.array(indices, dtype=np.int32))
        target.append(np.pi)

    # Dihedrals
    for arg in con.get('dihedrals', []):
        if len(arg) == 2:
            dihedrals.append(np.array(arg[0], dtype=np.int32))
            target.append(arg[1])
        else:
            angles.append(np.array(arg, dtype=np.int32))
            target.append(atoms.get_dihedral(*arg) * np.pi / 180.)
    for indices in dcons:
        dihedrals.append(np.array(indices, dtype=np.int32))
        target.append(0.)

    # Angle diffs
    for indices in adiffs:
        angle_diffs.append(np.array(indices, dtype=np.int32))
        target.append(0.)

    fix = np.array(fix, dtype=np.int32)
    bonds = np.array(bonds, dtype=np.int32)
    angles = np.array(angles, dtype=np.int32)
    dihedrals = np.array(dihedrals, dtype=np.int32)
    angle_diffs = np.array(angle_diffs, dtype=np.int32)
    target = np.array(target, dtype=np.float64)

    natoms = len(atoms)
    nbonds = np.zeros(natoms, dtype=np.int32)
    angle_sums = np.empty((0, 4), dtype=np.int32)
    return Constraints(atoms, nbonds, fix, bonds, angles, dihedrals,
                       angle_sums, angle_diffs, False, False, target)

# Merges previously constructed constraints object with new automatically
# determined constraints
def merge_old_constraints(atoms, con_old, bcons, acons, dcons, adiffs):
    asdf

def get_internal(atoms, atol=15., constraints=None, intlast=None):
    atol *= np.pi / 180.
    natoms = len(atoms)

    bonds, nbonds, c10y = find_bonds(atoms)
    if intlast is None:
        dummies_old = Atoms()
        dinds = None
        angle_sums_old = None
    else:
        dummies_old = intlast.dummies
        dinds = intlast.dinds
        angle_sums_old = intlast.angle_sums
    ndummies = len(dummies_old)

    # FIXME: This looks hideous, can it be improved?
    (angles, dummies, dinds, angle_sums,
            bcons, acons, dcons, adiffs) = find_angles(atoms, atol, bonds,
                    nbonds, c10y, ndummies, dinds, angle_sums_old)
    dummies = dummies_old + dummies

    ndummies = len(dummies)

    #fix, bcons, acons, dcons, adiffs, targets = merge_constraints(atoms,
    #                                                              constraints,
    #                                                              bcons,
    #                                                              acons,
    #                                                              dcons,
    #                                                              adiffs)

    dihedrals = find_dihedrals(atoms, atol, bonds, angles, nbonds,
                               c10y, dinds)

    internal = CartToInternal(atoms,
                              bonds=bonds,
                              angles=angles,
                              dihedrals=dihedrals,
                              angle_sums=angle_sums,
                              dummies=dummies)

    return internal
    #cons = Constraints(natoms + ndummies, bcons, acons, dcons,
    #                   np.empty((0, 4), dtype=np.int32), adiffs, fix, False,
    #                   False, dummies=dummies)
    #return internal, cons
