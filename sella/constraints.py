from typing import Dict, List
from collections import Sequence

import numpy as np

from ase import Atoms
from ase.constraints import (FixCartesian, FixAtoms, FixBondLengths,
                             FixInternals)

from sella.internal.int_classes import Constraints

_con_kinds = ['cart', 'bonds', 'angles', 'dihedrals']


def _sort_indices(indices):
    if indices[0] > indices[-1]:
        return tuple(reversed(indices))
    return tuple(indices)


def _add_check(ind, con, sort=True):
    if sort:
        ind = _sort_indices(ind)
    ind_np = np.asarray(ind)
    for entry in con:
        if np.all(ind_np == np.asarray(entry)):
            raise ValueError("A constraint has been added multiple times.")
    con.append(ind)


def _add_cart(atoms, arg, con, target):
    if isinstance(arg, Sequence):
        n = len(arg)
        if n == 1:
            return _add_cart(atoms, arg[0], con, target)
        elif n == 2:
            if isinstance(arg[1], (int, np.integer)):
                if arg[1] >= 3:
                    raise ValueError("Invalid cart constraint:", arg)
                _add_check(tuple(arg), con, sort=False)
                target.append(atoms.positions[tuple(arg)])
            else:
                raise ValueError("Invalid cart constraint:", arg)
        else:
            raise ValueError("Invalid cart constraint:", arg)
    elif isinstance(arg, (int, np.integer)):
        for i in range(3):
            _add_check((arg, i), con, sort=False)
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


_con_adders = dict(
    cart=_add_cart,
    bonds=_add_bond,
    angles=_add_angle,
    dihedrals=_add_dihedral,
)


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
        target_out['dihedrals'].append(np.pi)

    return con_out, target_out


def cons_to_dict(cons):
    # dict for combined user/generated constraints and targets
    con_out = {key: [] for key in _con_kinds}
    target_out = {key: [] for key in _con_kinds}

    tot = 0
    for kind in _con_kinds:
        for row in np.asarray(getattr(cons, kind)):
            con_out[kind].append(row)
            target_out[kind].append(cons.target[tot])
            tot += 1

    return con_out, target_out


def get_constraints(atoms, con, target, dummies=None, dinds=None,
                    proj_trans=True, proj_rot=True, **kwargs):
    target_all = []
    con_arrays = kwargs.copy()
    for kind in _con_kinds:
        target_all += target.get(kind, [])
        if con.get(kind, []):
            con_arrays[kind] = np.array(con[kind], dtype=np.int32)
        else:
            con_arrays[kind] = None
    target_all = np.array(target_all, dtype=np.float64)

    return Constraints(atoms,
                       target_all,
                       dummies=dummies,
                       dinds=dinds,
                       proj_trans=proj_trans,
                       proj_rot=proj_rot,
                       **con_arrays)


def _ase_constraints_to_dict(constraints):
    cart = set()
    bonds = set()
    angles = set()
    dihedrals = set()
    for constraint in constraints:
        if isinstance(constraint, FixAtoms):
            cart.update(set(constraint.index))
        elif isinstance(constraint, FixCartesian):
            idx = constraint.a
            if idx in cart:
                continue
            for i, xyz in enumerate(constraint.mask):
                if xyz:
                    cart.add((idx, i))
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
    if cart:
        con['cart'] = list(cart)
    if bonds:
        con['bonds'] = [_sort_indices(bond) for bond in bonds]
    if angles:
        con['angles'] = [_sort_indices(angle) for angle in angles]
    if dihedrals:
        con['dihedrals'] = [_sort_indices(dihedral) for dihedral in dihedrals]
    return con


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


class ConstraintBuilder:
    shapes = dict(cart=2, bonds=3, angles=5, dihedrals=7)

    def __init__(
        self,
        atoms: Atoms,
        con_user: Dict[str, np.ndarray] = None,
    ) -> None:
        if atoms.constraints:
            con_ase = _ase_constraints_to_dict(atoms.constraints)
        else:
            con_ase = dict()

        if con_user is None:
            con_user = dict()

        self.user_cons = {kind: [] for kind in self.shapes.keys()}
        self.user_target = {kind: [] for kind in self.shapes.keys()}

        for kind in self.shapes.keys():
            for con in [con_ase, con_user]:
                for val in con.get(kind, []):
                    _con_adders[kind](
                        atoms, val, self.user_cons[kind], self.user_target[kind]
                    )
        self.dummy_cons = {key: [] for key in self.shapes.keys()}

    def add_constraint(self, key: str, *args: List[int]) -> None:
        assert len(args) == self.shapes[key]
        self.dummy_cons[key].append(args)

    def get_constraints(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        dinds: np.ndarray = None,
        proj_trans: bool = True,
        proj_rot: bool = True,
    ) -> Constraints:
        con_out = self.user_cons.copy()
        target_out = self.user_target.copy()

        for arg in self.dummy_cons['bonds']:
            con_out['bonds'].append(_sort_indices(arg))
            target_out['bonds'].append(1.)

        for arg in self.dummy_cons['angles']:
            con_out['angles'].append(_sort_indices(arg))
            target_out['angles'].append(np.pi / 2.)

        for arg in self.dummy_cons['dihedrals']:
            con_out['dihedrals'].append(_sort_indices(arg))
            target_out['dihedrals'].append(np.pi)

        for key, val in con_out.items():
            shape = self.shapes[key]
            con_out[key] = np.array(val, dtype=np.int32).reshape((-1, shape))

        target_all = []
        for kind in _con_kinds:
            target_all += target_out.get(kind, [])
            if np.asarray(con_out[kind]).size == 0:
                con_out[kind] = None
        target_all = np.array(target_all, dtype=np.float64)

        return Constraints(
            atoms, target_all, dummies=dummies, dinds=dinds,
            proj_trans=proj_trans, proj_rot=proj_rot, **con_out,
        )
