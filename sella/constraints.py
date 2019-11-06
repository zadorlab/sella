import warnings

import numpy as np

from ase.constraints import (FixCartesian, FixAtoms, FixBondLengths,
                             FixInternals)

from sella.cython_routines import modified_gram_schmidt, simple_ortho
from sella.internal_cython import cart_to_internal
from sella.internal.int_classes import Constraints

_con_kinds = ['fix', 'bonds', 'angles', 'dihedrals',
              'angle_sums', 'angle_diffs']

def _sort_indices(indices):
    if indices[0] > indices[-1]:
        return tuple(reversed(indices))
    return tuple(indices)


def _add_check(arg, con):
    ind = _sort_indices(arg)
    ind_np = np.asarray(ind)
    for entry in con:
        if np.all(ind_np == np.asarray(entry)):
            raise ValueError("A constraint has been added multiple times.")
    con.append(ind)


def _add_fix(atoms, arg, con, target):
    if isinstance(arg, Sequence):
        n = len(arg)
        if n == 1:
            return _add_fix(atoms, arg[0], con, target)
        elif n == 2:
            if isinstance(arg[1], (int, np.integer)):
                if arg[1] >= 3:
                    raise ValueError("Invalid fix constraint:", arg)
                _add_check(tuple(arg), con)
                target.append(atoms.positions[tuple(arg)])
            else:
                raise ValueError("Invalid fix constraint:", arg)
        else:
            raise ValueError("Invalid fix constraint:", arg)
    elif isinstance(arg, (int, np.integer)):
        for i in range(3):
            _add_check((arg, i), con)
            target.append(atoms.positions[arg, i])


def _add_bond(atoms, arg, con, target):
    if not isinstance(arg, Sequence) or len(arg) != 2:
        raise ValueError("Invalid bond constraint:", arg)
    if isinstance(arg[0], Sequence):
        if len(arg[0]) != 2:
            raise ValueError("Invalid bond constraint:", arg)
        _add_check(arg[0], con)
        target.append(arg[1])
    elif isinstance(arg[0], (int, np.integer)):
        _add_check(arg, con)
        target.append(atoms.get_distance(*arg))
    else:
        raise ValueError("Invalid bond constraint:", arg)


def _add_angle(atoms, arg, con, target):
    if not isinstance(arg, Sequence):
        raise ValueError("Invalid angle constraint:", arg)
    n = len(arg)
    if n == 2:
        if not isinstance(arg[0], Sequence) or len(arg[0]) != 3:
            raise ValueError("Invalid angle constraint:", arg)
        _add_check(arg[0], con)
        target.append(arg[1] * np.pi / 180.)
    elif n == 3:
        _add_check(arg, con)
        target.append(atoms.get_angle(*arg) * np.pi / 180.)
    else:
        raise ValueError("Invalid angle constraint:", arg)


def _add_dihedral(atoms, arg, con, target):
    if not isinstance(arg, Sequence):
        raise ValueError("Invalid dihedral constraint:", arg)
    n = len(arg)
    if n == 2:
        if not isinstance(arg[0], Sequence) or len(arg[0]) != 4:
            raise ValueError("Invalid dihedral constraint:", arg)
        _add_check(arg[0], con)
        target.append(arg[1] * np.pi / 180.)
    elif n == 4:
        _add_check(arg, con)
        target.append(atoms.get_dihedral(*arg) * np.pi / 180.)
    else:
        raise ValueError("Invalid dihedral constraint:", arg)


def _add_angle_sum(atoms, arg, con, target):
    if not isinstance(arg, Sequence):
        raise ValueError("Invalid angle sum constraint:", arg)
    n = len(arg)
    if n == 2:
        if not isinstance(arg[0], Sequence) or len(arg[0]) != 4:
            raise ValueError("Invalid angle sum constraint:", arg)
        _add_check(arg[0], con)
        target.append(arg[1] * np.pi / 180.)
    elif n == 4:
        _add_check(arg, con)
        a1 = atoms.get_angle(arg[0], arg[1], arg[3])
        a2 = atoms.get_angle(arg[2], arg[1], arg[3])
        target.append((a1 + a2) * np.pi / 180.)
    else:
        raise ValueError("Invalid angle sum constraint:", arg)


def _add_angle_diff(atoms, arg, con, target):
    if not isinstance(arg, Sequence):
        raise ValueError("Invalid angle diff constraint:", arg)
    n = len(arg)
    if n == 2:
        if not isinstance(arg[0], Sequence) or len(arg[0]) != 4:
            raise ValueError("Invalid angle diff constraint:", arg)
        _add_check(arg[0], con)
        target.append(arg[1] * np.pi / 180.)
    elif n == 4:
        _add_check(arg, con)
        a1 = atoms.get_angle(arg[0], arg[1], arg[3])
        a2 = atoms.get_angle(arg[2], arg[1], arg[3])
        target.append((a1 - a2) * np.pi / 180.)
    else:
        raise ValueError("Invalid angle diff constraint:", arg)


_con_adders = dict(fix=_add_fix, bonds=_add_bond, angles=_add_angle,
                   dihedrals=_add_dihedral, angle_sums=_add_angle_sum,
                   angle_diffs=_add_angle_diff)


def merge_user_constraints(atoms, con_user=None):
    if atoms.constraints:
        con_ase = _ase_constraints_to_dict(atoms.constraints)
    else:
        con_ase = dict()

    if con_user is None:
        con_user = dict()

    con_out = {kind: [] for kind in _con_kinds}

    target = {kind: [] for kind in _con_kinds}

    for kind in _con_kinds:
        for con in [con_ase, con_user]:
            for arg in con.get(kind, []):
                _con_adders[kind](atoms, arg, con_out[kind], target[kind])

    return con_out, target


def merge_internal_constraints(con_user, target_user, bcons, acons, dcons,
                               adiffs):
    con_out = con_user.copy()
    target_out = target_user.copy()

    for arg in bcons:
        con_out['bonds'].append(_sort_indices(arg))
        target_out['bonds'].append(1.)

    for arg in acons:
        con_out['angles'].append(_sort_indices(arg))
        target_out['angles'].append(np.pi / 2.)

    for arg in dcons:
        con_out['dihedrals'].append(_sort_indices(arg))
        target_out['dihedrals'].append(0.)

    for arg in adiffs:
        con_out['angle_diffs'].append(_sort_indices(arg))
        target_out['angle_diffs'].append(0.)

    return con_out, target_out

def get_constraints(atoms, con, target, dummies=None, dinds=None):
    target_all = []
    con_arrays = dict()
    for kind in _con_kinds:
        target_all += target.get(kind, [])
        if con.get(kind, []):
            con_arrays[kind] = np.array(con[kind], dtype=np.int32)
        else:
            con_arrays[kind] = None
    target_all = np.array(target_all, dtype=np.float64)

    return Constraints(atoms, target_all, cart=con_arrays['fix'],
                       bonds=con_arrays['bonds'],
                       angles=con_arrays['angles'],
                       dihedrals=con_arrays['dihedrals'],
                       angle_sums=con_arrays['angle_sums'],
                       angle_diffs=con_arrays['angle_diffs'],
                       dummies=dummies,
                       dinds=dinds)

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
        con['bonds'] = [_sort_indices(bond) for bond in bonds]
    if angles:
        con['angles'] = [_sort_indices(angle) for angle in angles]
    if dihedrals:
        con['dihedrals'] = [_sort_indices(dihedral) for dihedral in dihedrals]
    return con


#class Constraints:
#    def __init__(self, atoms, conin, p_t=True, p_r=True):
#        # Note: We do not keep a reference to atoms!
#        con_ase = _ase_constraints_to_dict(atoms.constraints)
#        for kind in ['bonds', 'angles', 'dihedrals']:
#            for i, indices in enumerate(con_ase.get(kind, [])):
#                if indices[0] > indices[-1]:
#                    con_ase[kind][i] = tuple(reversed(indices))
#
#        if conin is None:
#            con = con_ase
#        else:
#            if con_ase:
#                warnings.warn("Constraints have been specified to Sella, but "
#                              "the ASE Atoms object also has constraints "
#                              "attached. These constraints will be merged, "
#                              "but any constraint that has been specified "
#                              "in both ways will result in a runtime failure. "
#                              "Consider removing all ASE constraints and "
#                              "instead specify all constraints to Sella "
#                              "directly.")
#            con = conin.copy()
#            for key, val in con_ase.items():
#                if key not in con:
#                    con[key] = val
#                else:
#                    con[key] += val
#
#        pos = atoms.positions
#        # First, consider fixed atoms
#        # Note that unlike other constraints, we do not allow the user
#        # to specify a target for fixed atom constraints. If the user
#        # wants to fix atoms to a position other than their initial
#        # value, they must manually move the atoms first.
#        con_f = con.pop('fix', tuple())
#        self.fix_ind = []
#        self.fix_target = []
#        # If there are any fixed atoms, this will affect how we project
#        # rotations.
#        if p_r:
#            self.rot_center = None
#            self.rot_axes = np.eye(3)
#            if len(atoms) == 2:
#                Y = (pos[1] - pos[0]).reshape((3, 1))
#                self.rot_axes = simple_ortho(np.eye(3), Y)
#        else:
#            self.rot_center = np.zeros(3)
#            self.rot_axes = np.empty((0, 3))
#        # We also keep track of which dimensions any atom has been
#        # fixed in for projecting out translations. E.g. if any atom
#        # has been fixed in the X direction, then we do not project
#        # out overall translation in that direction.
#        self.fixed_dims = [p_t, p_t, p_t]
#        # The user can provide fixed atoms constraints either as a
#        # list of atom indices, or as a list of (index, dimension)
#        # tuples, or they can mix the two approaches.
#        for arg in con_f:
#            if isinstance(arg, (int, np.int, np.int64)):
#                index = arg
#                target = pos[index]
#                # If there is only a single fixed atom, then all three
#                # rotations are still permitted, but the axes of rotation
#                # must pass through the fixed atom
#                if self.rot_center is None:
#                    self.rot_center = target
#                # If two atoms are fixed, then the only permitted rotation
#                # is along the axis defined by the two fixed atoms
#                elif self.rot_axes.shape[0] > 1:
#                    self.rot_axes = (target - self.rot_center)[np.newaxis, :]
#                    self.rot_axes /= np.linalg.norm(self.rot_axes)
#                # If at least three non-collinear atoms are fixed, then
#                # no rotation is permitted
#                elif self.rot_axes.shape[0] == 1:
#                    # If at least three atoms have been specified as
#                    # fixed, but the user asked for projected rotations,
#                    # verify that all atoms are collinear. If they
#                    # aren't, disable projection of rotations and warn
#                    # the user.
#                    axis = target - self.rot_center
#                    axis /= np.linalg.norm(axis)
#                    # A very loose threshold here to improve numerical
#                    # stability.
#                    if 1 - np.abs(axis @ self.rot_axes[0]) > 1e-2:
#                        self.rot_axes = np.empty((0, 3))
#
#                for dim, ipos in enumerate(target):
#                    self.fix_ind.append((index, dim))
#                    self.fix_target.append(ipos)
#                self.fixed_dims = [False, False, False]
#            else:
#                warnings.warn("Fixing atoms in some but not all "
#                              "directions has not been thoroughly "
#                              "tested yet. Proceed at your own risk.")
#                index, dim = arg
#                self.fix_ind.append((index, dim))
#                self.fix_target = pos[index, dim]
#                self.fixed_dims[dim] = False
#
#        # unlike bonds/angles/dihedrals below, no need to convert
#        # fix_ind to a numpy array.
#        self.nfix = len(self.fix_ind)
#
#        # Second, we consider translational/rotational motion.
#        self.trans_ind = []
#        self.trans_target = []
#        center = np.average(atoms.positions, axis=0)
#        for dim, fixed in enumerate(self.fixed_dims):
#            if fixed:
#                self.trans_ind.append(dim)
#                self.trans_target.append(center[dim])
#        self.ntrans = len(self.trans_ind)
#
#        # Rotational axes are contained in self.rot_axes
#        self.nrot = len(self.rot_axes)
#
#        if self.rot_center is None:
#            self.rot_center = center
#
#        # Next we look at the bonds
#        con_b = con.pop('bonds', tuple())
#        # The constraints will be temporarily held in the 'bonds' dict.
#        self.bonds_ind = []
#        self.bonds_target = []
#        # Assume user provided constraints in the form
#        # ((index_a, index_b), target_distance),
#        for indices, target in con_b:
#            # If "indices" is actually an int (rather than a tuple),
#            # then they didn't provide a target distance, so we will
#            # use the current value.
#            if isinstance(indices, np.integer):
#                indices = (indices, target)
#                target = atoms.get_distance(*indices)
#            self.bonds_ind.append(_sort_indices(indices))
#            self.bonds_target.append(target)
#
#        self.nbonds = len(self.bonds_ind)
#        self.bonds_ind = (np.array(self.bonds_ind, dtype=np.int32)
#                          .reshape((self.nbonds, 2)))
#        self.bonds_target = np.array(self.bonds_target, dtype=float)
#
#        # Then angles and dihedrals, which are very similar
#        con_a = con.pop('angles', tuple())
#        self.angles_ind = []
#        self.angles_target = []
#        self.angles = dict()
#        for arg in con_a:
#            if len(arg) == 2:
#                indices = _sort_indices(arg[0])
#                target = arg[1] * np.pi / 180.
#            else:
#                indices = _sort_indices(arg)
#                target = atoms.get_angle(*indices) * np.pi / 180.
#            self.angles_ind.append(_sort_indices(indices))
#            self.angles_target.append(target)
#
#        self.nangles = len(self.angles_ind)
#        self.angles_ind = (np.array(self.angles_ind, dtype=np.int32)
#                           .reshape((self.nangles, 3)))
#        self.angles_target = np.array(self.angles_target, dtype=float)
#
#        con_d = con.pop('dihedrals', tuple())
#        self.dihedrals_ind = []
#        self.dihedrals_target = []
#        for arg in con_d:
#            if len(arg) == 2:
#                indices = _sort_indices(arg[0])
#                target = arg[1] * np.pi / 180.
#            else:
#                indices = _sort_indices(arg)
#                target = atoms.get_dihedral(*indices) * np.pi / 180.
#            self.dihedrals_ind.append(_sort_indices(indices))
#            self.dihedrals_target.append(target)
#
#        self.ndihedrals = len(self.dihedrals_ind)
#        self.dihedrals_ind = (np.array(self.dihedrals_ind, dtype=np.int32)
#                              .reshape((self.ndihedrals, 4)))
#        self.dihedrals_target = np.array(self.dihedrals_target, dtype=float)
#
#        # If con is still non-empty, then we didn't process all of
#        # the user-provided constraints, so throw an error.
#        if con:
#            raise ValueError("Don't know what to do with constraint types: {}"
#                             "".format(con.keys()))
#        self.ninternal = self.nbonds + self.nangles + self.ndihedrals
#        self.nconstr = self.ninternal + self.nfix + self.ntrans + self.nrot
#
#        self.last = dict(x=np.zeros_like(pos),
#                         r=None,
#                         drdx=None,
#                         Ucons=None,
#                         Ufree=None)
#
#    def res(self, pos):
#        self._update_res(pos)
#        return self.last['r']
#
#    def drdx(self, pos):
#        self._update_res(pos)
#        return self.last['drdx']
#
#    def Ucons(self, pos):
#        self._update_res(pos)
#        return self.last['Ucons']
#
#    def Ufree(self, pos):
#        self._update_res(pos)
#        return self.last['Ufree']
#
#    def _update_res(self, pos):
#        if np.all(pos == self.last['x']):
#            return
#
#        # res is the residual vector, containing a measure of how much each
#        # constraint is being violated.
#        res = np.zeros(self.nconstr)
#
#        # drdx is the derivative of the residual w.r.t. Cartesian coordinates
#        drdx = np.zeros((len(pos.ravel()), self.nconstr))
#        res[:self.nfix] = (np.array([pos[ind] for ind in self.fix_ind])
#                           - self.fix_target)
#        n = self.nfix
#
#        mask = np.ones(self.ninternal, dtype=np.uint8)
#        p, B, _ = cart_to_internal(pos,
#                                   self.bonds_ind,
#                                   self.angles_ind,
#                                   self.dihedrals_ind,
#                                   mask,
#                                   True, False)
#
#        res[n:n+self.nbonds] = p[:self.nbonds] - self.bonds_target
#
#        n += self.nbonds
#        m = self.nbonds
#        res[n:n+self.nangles] = p[m:m+self.nangles] - self.angles_target
#
#        n += self.nangles
#        m += self.nangles
#        res_d = p[m:m+self.ndihedrals] - self.dihedrals_target
#        res[n:n+self.ndihedrals] = (res_d + np.pi) % (2 * np.pi) - np.pi
#
#        center = np.average(pos, axis=0)
#        n += self.ndihedrals
#        res[n:n+self.ntrans] = center[self.trans_ind] - self.trans_target
#
#        for i, (ind, dim) in enumerate(self.fix_ind):
#            k = 3 * ind + dim
#            drdx[k, i] = 1.
#
#        n = self.nfix
#        drdx[:, n:n+self.ninternal] = B.T
#
#        n += self.ninternal
#        if self.ntrans > 0:
#            tvec = np.zeros_like(pos)
#            tvec[:, 0] = 1.
#            tvec = tvec.ravel() / np.linalg.norm(tvec)
#            for dim in self.trans_ind:
#                drdx[:, n] = np.roll(tvec, dim)
#                n += 1
#
#        drdx[:, n:] = project_rotation(pos, center=self.rot_center,
#                                       axes=self.rot_axes)
#
#        # Ucons is an orthonormal basis spanning the constraint subspace
#        Ucons = modified_gram_schmidt(drdx)
#
#        # Ufree is an orthonormal basis spanning the subspace orthogonal to
#        # all constraints. Ucons and Ufree together form a complete basis.
#        Ufree = simple_ortho(np.eye(np.product(pos.shape)), Ucons)
#
#        self.last = dict(x=pos.copy(),
#                         r=res,
#                         drdx=drdx,
#                         Ucons=Ucons,
#                         Ufree=Ufree)
#

def project_rotation(x0, center=None, axes=None):
    x = x0.reshape((-1, 3))
    if center is None:
        center = np.average(x, axis=0)
    dx = x - center

    if axes is None:
        axes = np.eye(3)
    else:
        axes = axes.reshape((-1, 3))

    rots = np.zeros((len(x0.ravel()), axes.shape[0]))
    for i, axis in enumerate(axes):
        rots[:, i] = np.cross(axis, dx).ravel()

    vecs, lams, _ = np.linalg.svd(rots)
    indices = [i for i, lam in enumerate(lams) if abs(lam) > 1e-12]
    return vecs[:, indices]
