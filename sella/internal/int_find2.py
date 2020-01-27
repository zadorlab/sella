import numpy as np
from ase import Atom, Atoms

_MAX_BONDS = 20


def get_dummy(atoms, nbonds, c10y, i, j, k, atol):
    i, k = sorted([i, k], key=lambda x: atoms.get_distance(x, j))
    for n in [i, k]:
        for m in sorted(c10y[n, :nbonds[n]],
                        key=lambda x: atoms.get_distance(x, n)):
            if m in [i, j, k]:
                continue
            if atol < atoms.get_angle(j, n, m) < 180 - atol:
                dx1 = atoms.get_distance(m, n, vector=True)
                dx1 /= np.linalg.norm(dx1)

                dx2 = atoms.get_distance(n, j, vector=True)
                dx2 /= np.linalg.norm(dx2)
                dx1 -= dx2 * (dx1 @ dx2)
                dx1 /= np.linalg.norm(dx1)
                dpos = dx1 + atoms.positions[j]
                return dpos, n, m

    dx1 = atoms.get_distance(j, i, vector=True)
    dx2 = atoms.get_distance(j, k, vector=True)
    dpos = np.cross(dx1, dx2)
    dpos_norm = np.linalg.norm(dpos)
    if dpos_norm < 1e-4:
        dim = np.argmin(np.abs(dx1))
        dpos[:] = 0.
        dpos[dim] = 1.
        dpos -= dx1 * (dpos @ dx1) / (dx1 @ dx1)
        dpos /= np.linalg.norm(dpos)
    else:
        dpos /= dpos_norm

    return dpos, i, k


def check_if_planar(dxs, atol):
    dxs /= np.linalg.norm(dxs, axis=1)[:, np.newaxis]
    U, s, VT = np.linalg.svd(dxs)
    angles = np.arccos(dxs @ VT[:, 2])
    return np.max(angles) - np.min(angles) > atol


def find_angles(atoms, atol, bonds, nbonds, c10y, dummies=None, dinds=None):
    atoms = atoms.copy()
    natoms = len(atoms)

    if dummies is None:
        dummies = Atoms()
    else:
        dummies = dummies.copy()
    ndummies = len(dummies)

    if dinds is None:
        dinds = -np.ones(natoms, dtype=np.int32)

    angles = []

    bond_constraints = []
    angle_constraints = []
    angle_sums = []
    angle_diffs = []
    dihedral_constraints = []

    for j, nj in enumerate(nbonds):
        if nj == 0:
            raise RuntimeError("Atom {} has no bonds!".format(j))
        elif nj == 2:
            i, k = c10y[j, :2]
            if atol < atoms.get_angle(i, j, k) < 180 - atol:
                angles.append([i, j, k])
            elif dinds[j] < 0:
                dinds[j] = natoms + ndummies
                dpos, a, b = get_dummy(atoms, nbonds, c10y, i, j, k, atol)
                dummies += Atom('X', dpos)

                bond_constraints.append([j, dinds[j]])
                angle_constraints.append([a, j, dinds[j]])
                dihedral_constraints.append([b, a, j, dinds[j]])

                ndummies += 1
        else:
            linear = []
            for a, i in enumerate(c10y[j, :nj-1]):
                for k in c10y[j, a+1:nj]:
                    if atol < atoms.get_angle(i, j, k) < 180 - atol:
                        angles.append([i, j, k])
                    else:
                        linear.append(sorted([i, k], key=lambda x: atoms.get_distance(x, j)))
            if dinds[j] < 0 and linear:
                dxs = atoms.positions[c10y[j, :nj]] - atoms.positions[j]
                if check_if_planar(dxs, atol):
                    dinds[j] = natoms + ndummies

                    linear = sorted(linear,
                                    key=lambda x: atoms.get_distance(x[0], j))
                    a = linear[0][0]
                    if len(linear) == 1:
                        i, k = linear[0]
                        for b in sorted(c10y[j, :nj], key=lambda x: atoms.get_distance(x, j)):
                            if b not in [i, j, k]:
                                break
                        else:
                            raise RuntimeError("???")
                    else:
                        b = linear[1][0]
                    dx1 = atoms.get_distance(j, a, vector=True)
                    dx2 = atoms.get_distance(j, b, vector=True)
                    dpos = np.cross(dx1, dx2)
                    dpos_norm = np.linalg.norm(dpos)
                    if dpos_norm < 1e-4:
                        dim = np.argmin(np.abs(dx1))
                        dpos[:] = 0.
                        dpos[dim] = 1
                        dpos -= dx1 * (dpos @ dx1) / (dx1 @ dx1)
                    else:
                        dpos /= dpos_norm
                    dummies += Atom('X', dpos + atoms.positions[j])

                    bond_constraints.append([j, dinds[j]])
                    angle_constraints += [[a, j, dinds[j]], [b, j, dinds[j]]]
                    ndummies += 1

        if dinds[j] > 0:
            allatoms = atoms + dummies
            for i in c10y[j, :nj]:
                if atol < allatoms.get_angle(i, j, dinds[j]) < 180 - atol:
                    angles.append([i, j, dinds[j]])

    allatoms = atoms + dummies
    # verify constraints are correct
    for i, j in bond_constraints:
        rij = allatoms.get_distance(i, j)
        assert abs(1 - rij) < 1e-5, rij
    for i, j, k in angle_constraints:
        aijk = allatoms.get_angle(i, j, k)
        assert abs(90 - aijk) < 1e-5, aijk
    for i, j, k, l in dihedral_constraints:
        dijkl = allatoms.get_dihedral(i, j, k, l)
        assert abs(180 - dijkl) < 1e-5, dijkl
    return (np.array(angles, dtype=np.int32).reshape((-1, 3)),
            dummies,
            dinds,
            np.array(angle_sums, dtype=np.int32).reshape((-1, 4)),
            np.array(bond_constraints, dtype=np.int32).reshape((-1, 2)),
            np.array(angle_constraints, dtype=np.int32).reshape((-1, 3)),
            np.array(dihedral_constraints, dtype=np.int32).reshape((-1, 4)),
            np.array(angle_diffs, dtype=np.int32).reshape((-1, 4)))


def find_dihedrals(atoms, atol, bonds, angles, nbonds, c10y, dinds):
    nreal = len(c10y)
    dihedrals = []
    for i, j, k in angles:
        for n, m in ((i, k), (k, i)):
            if n >= nreal:
                continue
            bonds_n = list(c10y[n, :nbonds[n]])
            if dinds[n] > 0:
                bonds_n.append(dinds[n])
            for l in bonds_n:
                if (l > m and l != j
                        and atol < atoms.get_angle(j, n, l) < 180 - atol):
                    dihedrals.append([m, j, n, l])
    return np.array(dihedrals, dtype=np.int32).reshape((-1, 4))
