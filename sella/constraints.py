import warnings

import numpy as np
from scipy.linalg import null_space

from ase.constraints import (FixCartesian, FixAtoms, FixBondLengths,
                             FixInternals)

from .cython_routines import modified_gram_schmidt
from .internal_cython import cart_to_internal


def _sort_indices(indices):
    if indices[0] > indices[-1]:
        indices = tuple(reversed(indices))
    return indices


def _ase_constraints_to_dict(constraints):
    fix = set()
    bonds = set()
    angles = set()
    dihedrals = set()
    for constraint in constraints:
        if isinstance(constraint, FixAtoms):
            fix.update(set(constraint.index))
        elif isinstance(constraint, FixCartesian):
            idx = constraint.a
            if idx in fix:
                continue
            for i, xyz in enumerate(constraint.mask):
                if xyz:
                    fix.add((idx, i))
        elif isinstance(constraint, FixBondLengths):
            bonds.update(set(constraint.pairs))
        elif isinstance(constraint, FixInternals):
            bonds.update(set(constraints.bonds))
            angles.update(set(constraints.angles))
            dihedrals.update(set(constraints.dihedrals))
        else:
            raise ValueError("Sella does not know how to handle the ASE {} "
                             "constraint object!".format(constraint))
    con = dict()
    if fix:
        con['fix'] = list(fix)
    if bonds:
        con['bonds'] = list(bonds)
    if angles:
        con['angles'] = list(angles)
    if dihedrals:
        con['dihedrals'] = list(dihedrals)
    return con


def initialize_constraints(atoms, pos0, conin, p_t, p_r):
    constraints = dict()
    nconstraints = 0

    con_ase = _ase_constraints_to_dict(atoms.constraints)
    for kind in ['bonds', 'angles', 'dihedrals']:
        for i, indices in enumerate(con_ase.get(kind, [])):
            if indices[0] > indices[-1]:
                con_ase[kind][i] = tuple(reversed(indices))

    # Make a copy, because we are going to be popping entries
    # off the object to ensure that we've parsed everything,
    # and we don't want to modify the user-provided object,
    # since they might have other plans for it!
    if conin is None:
        con = con_ase
    else:
        if con_ase:
            warnings.warn("Constraints have been specified to Sella, but "
                          "the ASE Atoms object also has constraints "
                          "attached. These constraints will be merged, "
                          "but any constraint that has been specified "
                          "in both ways will result in a runtime failure. "
                          "Consider removing all ASE constraints and "
                          "instead specify all constraints to Sella "
                          "directly.")
        con = conin.copy()
        for key, val in con_ase.items():
            if key not in con:
                con[key] = val
            else:
                con[key] += val

    # First, consider fixed atoms
    # Note that unlike other constraints, we do not allow the user
    # to specify a target for fixed atom constraints. If the user
    # wants to fix atoms to a position other than their initial
    # value, they must manually move the atoms first.
    con_f = con.pop('fix', tuple())
    fix = dict()
    # If there are any fixed atoms, this will affect how we project
    # rotations.
    rot_center = None
    rot_axes = None
    # We also keep track of which dimensions any atom has been
    # fixed in for projecting out translations. E.g. if any atom
    # has been fixed in the X direction, then we do not project
    # out overall translation in that direction.
    fixed_dims = [p_t, p_t, p_t]
    # The user can provide fixed atoms constraints either as a
    # list of atom indices, or as a list of (index, dimension)
    # tuples, or they can mix the two approaches.
    for arg in con_f:
        if isinstance(arg, (int, np.int, np.int64)):
            index = arg
            target = pos0[index]
            if rot_center is None:
                rot_center = target
            elif rot_axes is None:
                rot_axes = target - rot_center
                rot_axes /= np.linalg.norm(rot_axes)
            elif p_r:
                # If at least three atoms have been specified as
                # fixed, but the user asked for projected rotations,
                # verify that all atoms are collinear. If they
                # aren't, disable projection of rotations and warn
                # the user.
                axis = target - rot_center
                axis /= np.linalg.norm(axis)
                if 1 - np.abs(axis @ rot_axes) > 1e-2:
                    warnings.warn("At least 3 non-collinear atoms are "
                                  "fixed, but projection of rotational "
                                  "modes has been requested. This is "
                                  "not correct! Disabling projection of "
                                  "rotational modes.")
                    p_r = False

            for dim, ipos in enumerate(target):
                fix[(index, dim)] = ipos
            fixed_dims = [False, False, False]
        else:
            if p_t or p_r:
                warnings.warn("Fixing atoms in some but not all "
                              "directions while also projecting out "
                              "translational or rotational motion has "
                              "not been tested!\n "
                              "Proceed at your own risk.")
            index, dim = arg
            fix[(index, dim)] = pos0[index, dim]
            fixed_dims[dim] = False

    constraints['fix'] = fix
    nconstraints += len(fix)

    # Second, we consider translational/rotational motion.
    constraints['translations'] = dict()
    center = np.average(atoms.positions, axis=0)
    for dim, fixed in enumerate(fixed_dims):
        if fixed:
            constraints['translations'][dim] = center[dim]
    constraints['rotations'] = p_r
    nconstraints += np.sum(fixed_dims) + 3 * p_r

    if rot_center is None:
        rot_center = center
    print(center)

    # Next we look at the bonds
    con_b = con.pop('bonds', tuple())
    # The constraints will be temporarily held in the 'bonds' dict.
    bonds = dict()
    # Assume user provided constraints in the form
    # ((index_a, index_b), target_distance),
    for indices, target in con_b:
        # If "indices" is actually an int (rather than a tuple),
        # then they didn't provide a target distance, so we will
        # use the current value.
        if isinstance(indices, int):
            indices = (indices, target)
            target = atoms.get_distance(*indices)
        bonds[_sort_indices(indices)] = target

    constraints['bonds'] = bonds
    nconstraints += len(bonds)

    # Then angles and dihedrals, which are very similar
    con_a = con.pop('angles', tuple())
    angles = dict()
    for arg in con_a:
        if len(arg) == 2:
            indices = _sort_indices(arg[0])
            target = arg[1]
        else:
            indices = _sort_indices(arg)
            target = atoms.get_angle(*indices) * np.pi / 180.
        angles[_sort_indices(indices)] = target

    constraints['angles'] = angles
    nconstraints += len(angles)

    con_d = con.pop('dihedrals', tuple())
    dihedrals = dict()
    for arg in con_d:
        if len(arg) == 2:
            indices = _sort_indices(arg[0])
            target = arg[1]
        else:
            indices = _sort_indices(arg)
            target = atoms.get_dihedral(*indices) * np.pi / 180.
        dihedrals[_sort_indices(indices)] = target

    constraints['dihedrals'] = dihedrals
    nconstraints += len(dihedrals)

    # If con is still non-empty, then we didn't process all of
    # the user-provided constraints, so throw an error.
    if con:
        raise ValueError("Don't know what to do with constraint types: {}"
                         "".format(con.keys()))
    return constraints, nconstraints, rot_center, rot_axes


def calc_constr_basis(atoms, con, ncon, rot_center, rot_axes):
    d = 3 * len(atoms)
    pos = atoms.get_positions()

    # If we don't have any constraints, then this is a very
    # easy task.
    if ncon == 0:
        res = np.empty(0)
        drdx = np.empty((d, 0))
        Tm = np.eye(d)
        Tfree = np.eye(d)
        Tc = np.empty((d, 0))
        return res, drdx, Tm, Tfree, Tc

    # Calculate Tc and Tm (see above)
    res = np.zeros(ncon)  # The constraint residuals
    drdx = np.zeros((d, ncon))

    n = 0
    # For now we ignore the target value contained in the "fix"
    # constraints dict.
    del_indices = []
    for index, dim in con.get('fix', dict()).keys():
        # The residual for fixed atom constraints will always be
        # zero, because no motion that would break those constraints
        # is allowed during dynamics.
        # self._res[n] = 0  # <-- not necessary to state explicitly
        idx_flat = 3 * index + dim  # Flattened coordinate index
        # Add the index to the list of indices to delete
        del_indices.append(idx_flat)
        drdx[idx_flat, n] = 1.
        n += 1

    free_indices = list(range(d))
    for idx in sorted(del_indices, reverse=True):
        del free_indices[idx]

    Tfree = np.eye(d)[:, free_indices]

    # Early exit shortcut: if there are no other constraints, exit now
    if n == ncon:
        Tc = np.eye(d)[:, del_indices]
        Tm = Tfree.copy()
        return res, drdx, Tm, Tfree, Tc

    # Now consider translation
    tvec = np.zeros_like(pos)
    tvec[:, 0] = 1. / tvec.shape[0]
    center = np.average(pos, axis=0)
    for dim, val in con.get('translations', dict()).items():
        res[n] = center[dim] - val
        drdx[:, n] = np.roll(tvec, dim, axis=1).ravel()
        n += 1

    # And rotations
    if con.get('rotations', False):
        drdx_rot = project_rotation(pos.ravel(),
                                    rot_center,
                                    rot_axes)
        _, nrot = drdx_rot.shape
        # TODO: Figure out how to use these constraints to push the
        # structure into a consistent orientation in space. That
        # might involve reworking how rotational constraints are
        # enforced completely.
        # self._res[n : n+nrot] = ???
        drdx[:, n : n+nrot] = drdx_rot
        n += nrot

    # Bonds
    b_indices = np.array(list(con.get('bonds', dict()).keys()),
                         dtype=np.int32).reshape((-1, 2))
    # Angles
    a_indices = np.array(list(con.get('angles', dict()).keys()),
                         dtype=np.int32).reshape((-1, 3))
    # Dihedrals
    d_indices = np.array(list(con.get('dihedrals', dict()).keys()),
                         dtype=np.int32).reshape((-1, 4))

    mask = np.ones(ncon - n, dtype=np.uint8)
    r_int, drdx_int, _ = cart_to_internal(pos,
                                          b_indices,
                                          a_indices,
                                          d_indices,
                                          mask,
                                          gradient=True)

    m = 0
    for indices in b_indices:
        r_int[m] -= con['bonds'][tuple(indices)]
        m += 1

    for indices in a_indices:
        r_int[m] -= con['angles'][tuple(indices)]
        m += 1

    for indices in d_indices:
        val = r_int[m]
        target = con['dihedrals'][tuple(indices)]
        r_int[m] = (val - target + np.pi) % (2 * np.pi) - np.pi
        m += 1

    res[n:] = r_int
    drdx[:, n:] = drdx_int.T

    # If we've made it this far and there's only one constraint,
    # we don't need to work any harder to get a good orthonormal basis.
    if drdx.shape[1] == 1:
        Tc = drdx / np.linalg.norm(drdx)
    # Otherwise, we need to orthonormalize everything
    else:
        Tc = modified_gram_schmidt(drdx)

    Tm = null_space(Tc.T)

    return res, drdx, Tm, Tfree, Tc


def project_rotation(x0, center=None, axes=None):
    x = x0.reshape((-1, 3))
    if center is None:
        center = np.average(x, axis=0)
    dx = x - center

    if axes is None:
        axes = np.eye(3)
    else:
        axes = axes.reshape((-1, 3))

    rots = np.zeros((len(x0), axes.shape[0]))
    for i, axis in enumerate(axes):
        rots[:, i] = np.cross(axis, dx).ravel()

    vecs, lams, _ = np.linalg.svd(rots)
    indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
    return vecs[:, indices]
