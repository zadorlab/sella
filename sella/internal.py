from typing import (
    Tuple, Callable, Iterator, Union, TypeVar, Optional, List, Dict
)
from itertools import (
    product,
    combinations,
    combinations_with_replacement as cwr
)
from functools import partialmethod
import warnings

import numpy as np
from ase import Atom, Atoms, units
from ase.cell import Cell
from ase.geometry import complete_cell, minkowski_reduce
from ase.data import covalent_radii
from ase.constraints import (
    FixConstraint, FixAtoms, FixCom, FixBondLengths, FixCartesian, FixInternals
)

import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, custom_jvp

from sella.linalg import (
    SparseInternalJacobian, SparseInternalHessian, SparseInternalHessians
)


IVec = Tuple[int, int, int]


class NoValidInternalError(ValueError):
    pass


class DuplicateInternalError(ValueError):
    pass


class DuplicateConstraintError(DuplicateInternalError):
    pass


def _gradient(
    func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return jit(grad(func, argnums=0))


def _hessian(
    func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return jit(jacfwd(jacrev(func, argnums=0), argnums=0))


class Internal:
    nindices = None
    union = None
    diff = None

    def __init__(
        self,
        indices: Tuple[int, ...],
        ncvecs: Tuple[IVec, ...] = None
    ) -> None:
        if self.nindices is not None:
            assert len(indices) == self.nindices
        self.indices = np.array(indices, dtype=np.int32)

        if self.nindices is not None:
            if ncvecs is None:
                ncvecs = np.zeros((self.nindices - 1, 3), dtype=np.int32)
            else:
                ncvecs = np.asarray(ncvecs).reshape((self.nindices - 1, 3))
        else:
            if ncvecs is not None:
                raise ValueError(
                    "{} does not support ncvecs"
                    .format(self.__class__.__name__)
                )
            ncvecs = np.empty((0, 3), dtype=np.int32)
        self.ncvecs = ncvecs

    def reverse(self) -> 'Internal':
        return self.__class__(self.indices[::-1], -self.ncvecs[::-1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if (
            np.all(self.indices == other.indices)
            and np.all(self.ncvecs == other.ncvecs)
        ):
            return True
        srev = self.reverse()
        if (
            np.all(srev.indices == other.indices)
            and np.all(srev.ncvecs == other.ncvecs)
        ):
            return True
        return False

    def __add__(self, other: object) -> 'Internal':
        if self.union is None:
            return NotImplemented
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self == other:
            raise NoValidInternalError(
                'Cannot add {} object to itself.'
                .format(self.__class__.__name__)
            )

        for s, o in product([self, self.reverse()], [other, other.reverse()]):
            if (
                np.all(s.indices[1:] == o.indices[:-1])
                and np.all(s.ncvecs[1:] == o.ncvecs[:-1])
            ):
                new_indices = [*s.indices, o.indices[-1]]
                new_ncvecs = [*s.ncvecs, o.ncvecs[-1]]
                return self.union(new_indices, new_ncvecs)
        raise NoValidInternalError(
            '{} indices do not overlap!'.format(self.__class__.__name__)
        )

    def split(self) -> Tuple['Internal', 'Internal']:
        if self.diff is None:
            raise RuntimeError(
                "Don't know how to split a {}!".format(self.__class__.__name__)
            )
        return (
            self.diff(self.indices[:-1], self.ncvecs[:-1]),
            self.diff(self.indices[1:], self.ncvecs[1:])
        )

    def __repr__(self) -> str:
        return "{}(indices={}, ncvecs={})".format(
            self.__class__.__name__,
            tuple(self.indices),
            tuple([tuple(vec) for vec in self.ncvecs])
        )

    @staticmethod
    def _eval0(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def _eval1(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _eval2(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    def calc(self, atoms: Atoms) -> float:
        tvecs = jnp.asarray(self.ncvecs @ atoms.cell, dtype=np.float64)
        return float(self._eval0(atoms[self.indices].positions, tvecs))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        tvecs = jnp.asarray(self.ncvecs @ atoms.cell, dtype=np.float64)
        return np.array(self._eval1(atoms[self.indices].positions, tvecs))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        tvecs = jnp.asarray(self.ncvecs @ atoms.cell, dtype=np.float64)
        return np.array(self._eval2(atoms[self.indices].positions, tvecs))


def _translation(
    pos: jnp.ndarray,
    dim: int,
    tvecs: jnp.ndarray
) -> float:
    return pos[:, dim].mean()


class Translation(Internal):
    def __init__(
        self,
        indices: Tuple[int, ...],
        dim: int,
    ) -> None:
        self.indices = np.array(sorted(indices), dtype=np.int32)
        self.ncvecs = np.empty((0, 3), dtype=np.int32)
        self.dim = dim

    def reverse(self) -> 'Internal':
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        # JAX seems subtlely broken. Accessing the last element of a
        # large jax.numpy.ndarray object is something like 5 orders of
        # magnitude slower than accessing the last element of a
        # numpy.ndarray objects, so we convert to numpy.ndarray here.
        sidx = np.asarray(self.indices, dtype=np.int32)
        oidx = np.asarray(other.indices, dtype=np.int32)
        if self.dim != other.dim:
            return False
        if len(sidx) != len(oidx):
            return False
        if np.all(sidx == oidx):
            return True
        return False

    def __repr__(self) -> str:
        return "{}(indices={}, dim={})".format(
            self.__class__.__name__,
            tuple(self.indices),
            self.dim
        )

    def calc(self, atoms: Atoms) -> float:
        tvecs = jnp.asarray(self.ncvecs @ atoms.cell, dtype=np.float64)
        return float(self._eval0(
            atoms.positions[self.indices], self.dim, tvecs
        ))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        tvecs = jnp.asarray(self.ncvecs @ atoms.cell, dtype=np.float64)
        return np.array(self._eval1(
            atoms.positions[self.indices], self.dim, tvecs
            ))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        tvecs = jnp.asarray(self.ncvecs @ atoms.cell, dtype=np.float64)
        return np.array(self._eval2(
            atoms.positions[self.indices], self.dim, tvecs
        ))

    _eval0 = staticmethod(jit(_translation))
    _eval1 = staticmethod(_gradient(_translation))
    _eval2 = staticmethod(_hessian(_translation))


# Nominally, jax.numpy.linalg.eigh supports auto-differentiation,
# but if any of the eigenvalues are degenerate, the derivatives
# of *all* eigenvectors will be NaN. Worryingly, this seems to be
# the case when the molecule in question is sufficiently high-symmetry
# (e.g. methane) and has not been rotated.
#
# We are assuming here that the eigenvector of interest corresponds
# to a simple (non-degenerate) eigenvalue (though we permit the
# possibility of there being other degenerate eigenvalues).
#
# As far as I can tell, we are limited to forward-auto-differentation,
# at least for the first derivative, as backwards-mode requires
# consideration of all eigenvalues/eigenvectors, which may be undefined.
@custom_jvp
def eigh_rightmost(X):
    return jnp.linalg.eigh((X + X.T) / 2.)[1][:, -1]


@eigh_rightmost.defjvp
def eigh_rightmost_jvp(primals, tangents):
    X, = primals
    X = (X + X.T) / 2.
    dX, = tangents
    dX = (dX + dX.T) / 2.
    dim = X.shape[0]
    ws, vecs = jnp.linalg.eigh(X)
    v = vecs[:, -1]
    ldot = v.T @ dX @ v
    # Note that the last term, `-ldot * v`, does not actually contribute
    # to the first derivative since `v` is orthogonal to
    # `pinv(ws[-1] * eye(dim) - X)`. However, this term must be included for
    # the *second* derivative to be correct.
    return v, jnp.linalg.pinv(ws[-1] * jnp.eye(dim) - X) @ (dX @ v - ldot * v)


def _rotation_q(
    pos: jnp.ndarray,
    refpos: jnp.ndarray
) -> float:
    dx = pos - pos.mean(0)
    R = dx.T @ refpos
    Rtr = jnp.trace(R)
    Ftop = jnp.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]])
    F = jnp.block([
        [Rtr, Ftop[None, :]],
        [Ftop[:, None], -Rtr * jnp.eye(3) + R + R.T],
    ])
    q = eigh_rightmost(F)
    return q * jnp.sign(q[0])


# "inverse sinc" function, naive, undefined at x=1
def _asinc_naive(x):
    return jnp.arccos(x) / jnp.sqrt(1 - x**2)


# Taylor series expansion of _asinc_naive around x=1
def _asinc_taylor(x):
    y = x - 1
    return (
        1
        - y / 3
        + 2 * y**2 / 15
        - 2 * y**3 / 35
        + 8 * y**4 / 315
        - 8 * y**5 / 693
        + 16 * y**6 / 3003
        - 16 * y**7 / 6435
        + 128 * y**8 / 109395
        - 128 * y**9 / 230945
    )


def asinc(x):
    return jnp.where(
        x < 0.97,
        _asinc_naive(jnp.where(x < 0.97, x, 0.97)),
        _asinc_taylor(x)
    )


def _rotation(
    pos: jnp.ndarray,
    axis: int,
    refpos: jnp.ndarray
) -> float:
    q = _rotation_q(pos, refpos)
    return 2 * q[axis + 1] * asinc(q[0])


class Rotation(Internal):
    def __init__(
        self,
        indices: Tuple[int, ...],
        axis: int,
        refpos: np.ndarray,
    ) -> None:
        assert len(indices) >= 2
        self.indices = np.array(indices, dtype=np.int32)
        self.refpos = refpos.copy() - refpos.mean(0)
        self.axis = axis

    def reverse(self) -> 'Internal':
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.axis != other.axis:
            return False
        if len(self.indices) != len(other.indices):
            return False
        if set(self.indices) != set(other.indices):
            return False
        return True

    def __repr__(self) -> str:
        return "{}(indices={}, axis={}, refpos={})".format(
            self.__class__.__name__,
            tuple(self.indices),
            self.axis,
            self.refpos,
        )

    def calc(self, atoms: Atoms) -> float:
        return float(self._eval0(
            atoms.positions[self.indices], self.axis, self.refpos
        ))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        return np.array(self._eval1(
            atoms.positions[self.indices], self.axis, self.refpos
        ))

    def calc_hessian(self, atoms: Atoms) -> np.ndarray:
        return np.array(self._eval2(
            atoms.positions[self.indices], self.axis, self.refpos
        ))

    _eval0 = staticmethod(jit(_rotation))
    _eval1 = staticmethod(jit(jacfwd(_rotation, argnums=0)))
    _eval2 = staticmethod(jit(jacfwd(jacfwd(_rotation, argnums=0), argnums=0)))


def _displacement(
    pos: jnp.ndarray,
    refpos: jnp.ndarray,
    W: jnp.ndarray
) -> float:
    dx = (pos - refpos).ravel()
    return dx @ W @ dx


class Displacement(Internal):
    def __init__(
        self,
        indices: np.ndarray,
        refpos: np.ndarray,
        W: np.ndarray,
    ) -> None:
        self.indices = indices.copy()
        self.refpos = refpos.copy()
        self.W = W.copy()

    def reverse(self) -> 'Internal':
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return np.all(self.refpos == other.refpos)

    def __repr__(self) -> str:
        return "{}(indices={}, refpos={}, W={})".format(
            self.__class__.__name__,
            self.indices,
            self.refpos,
            self.W,
        )

    def calc(self, atoms: Atoms) -> float:
        return float(self._eval0(
            atoms.positions[self.indices], self.refpos, self.W
        ))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        return np.array(self._eval1(
            atoms.positions[self.indices], self.refpos, self.W
        ))

    def calc_hessian(self, atoms: Atoms) -> np.ndarray:
        return np.array(self._eval2(
            atoms.positions[self.indices], self.refpos, self.W
        ))

    _eval0 = staticmethod(jit(_displacement))
    _eval1 = staticmethod(jit(_gradient(_displacement)))
    _eval2 = staticmethod(jit(_hessian(_displacement)))


def _bond(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    return jnp.linalg.norm(
        pos[1] - pos[0] + tvecs[0]
    )


class Bond(Internal):
    nindices = 2
    _eval0 = staticmethod(jit(_bond))
    _eval1 = staticmethod(_gradient(_bond))
    _eval2 = staticmethod(_hessian(_bond))

    def calc_vec(self, atoms: Atoms) -> np.ndarray:
        tvecs = np.asarray(self.ncvecs @ atoms.cell, dtype=np.float64)
        i, j = self.indices
        return atoms.positions[j] - atoms.positions[i] + tvecs[0]


def _angle(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    dx1 = -(pos[1] - pos[0] + tvecs[0])
    dx2 = pos[2] - pos[1] + tvecs[1]
    return jnp.arccos(
        dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    )


class Angle(Internal):
    nindices = 3
    _eval0 = staticmethod(jit(_angle))
    _eval1 = staticmethod(_gradient(_angle))
    _eval2 = staticmethod(_hessian(_angle))


def _dihedral(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    dx1 = pos[1] - pos[0] + tvecs[0]
    dx2 = pos[2] - pos[1] + tvecs[1]
    dx3 = pos[3] - pos[2] + tvecs[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)


class Dihedral(Internal):
    nindices = 4
    _eval0 = staticmethod(jit(_dihedral))
    _eval1 = staticmethod(_gradient(_dihedral))
    _eval2 = staticmethod(_hessian(_dihedral))


Bond.union = Angle
Angle.union = Dihedral
Angle.diff = Bond
Dihedral.diff = Angle


class BaseInternals:
    _names = (
        'translations', 'bonds', 'angles', 'dihedrals', 'other', 'rotations'
    )

    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        dinds: np.ndarray = None
    ) -> None:
        self.atoms = atoms

        self._lastpos = None
        self._cache = dict()

        if dummies is None:
            if dinds is not None:
                raise ValueError('"dinds" provided, but no "dummies"!')
            dummies = Atoms()
            dinds = -np.ones(len(self.atoms), dtype=np.int32)
        else:
            if dinds is None:
                raise ValueError('"dummies" provided, but no "dinds"!')
            ndum = len(dummies)
            ndind = np.sum(dinds >= 0)
            if ndum != ndind:
                raise ValueError(
                    '{} dummy atoms were provided, but only {} dummy indices!'
                    .format(ndum, ndind)
                )
        self.dummies = dummies
        self.dinds = dinds

        self.internals = {key: [] for key in self._names}
        self._active = {key: [] for key in self._names}
        self.cell = None
        self.rcell = None
        self.op = None

    @property
    def natoms(self) -> int:
        return len(self.atoms)

    @property
    def ndummies(self) -> int:
        return len(self.dummies)

    @property
    def ntrans(self) -> int:
        return sum(self._active['translations'])

    @property
    def nbonds(self) -> int:
        return sum(self._active['bonds'])

    @property
    def nangles(self) -> int:
        return sum(self._active['angles'])

    @property
    def ndihedrals(self) -> int:
        return sum(self._active['dihedrals'])

    @property
    def nrotations(self) -> int:
        return sum(self._active['rotations'])

    @property
    def _active_mask(self) -> List[bool]:
        active = []
        for name in self._names:
            active += self._active[name]
        return active

    @property
    def _active_indices(self) -> List[int]:
        return [idx for idx, active in enumerate(self._active_mask) if active]

    @property
    def nint(self) -> int:
        return len(self._active_indices)

    @property
    def ndof(self) -> int:
        return 3 * (self.natoms + self.ndummies)

    @property
    def all_atoms(self) -> Atoms:
        return self.atoms + self.dummies

    def _cache_check(self) -> None:
        # we are comparing the current atomic positions to what they were
        # the last time a property was calculated. These positions are floats,
        # but we use a strict equality check to compare to avoid subtle bugs
        # that might occur during fine-resolution geodesic steps.
        if (
            self._lastpos is None
            or np.any(self.all_atoms.positions != self._lastpos)
        ):
            self._cache = dict()
            self._lastpos = self.all_atoms.positions.copy()

    def copy(self) -> 'BaseInternals':
        raise NotImplementedError

    def calc(self) -> np.ndarray:
        """Calculates the internal coordinate vector."""
        self._cache_check()
        if 'coords' not in self._cache:
            atoms = self.all_atoms
            self._cache['coords'] = np.array([c.calc(atoms) for c in self])
        return np.array([x for x, a in zip(self._cache['coords'], self._active_mask) if a])

    def jacobian(self) -> np.ndarray:
        """Calculates the internal coordinate Jacobian matrix."""
        self._cache_check()
        if 'jacobian' not in self._cache:
            atoms = self.all_atoms
            self._cache['jacobian'] = [
                np.array(c.calc_gradient(atoms)) for c in self
            ]
        indices = [np.array(c.indices) for c, a in zip(self, self._active_mask) if a]
        return SparseInternalJacobian(
            self.natoms + self.ndummies,
            indices,
            [j for j, a in zip(self._cache['jacobian'], self._active_mask) if a]
        ).asarray()

    def hessian(self) -> np.ndarray:
        """Calculates the Hessian matrix for each internal coordinate."""
        self._cache_check()
        if 'hessian' not in self._cache:
            atoms = self.all_atoms
            self._cache['hessian'] = [
                np.array(c.calc_hessian(atoms)) for c in self
            ]
        indices = [np.array(c.indices) for c in self]
        hessians = []
        for idx, vals, active in zip(indices, self._cache['hessian'], self._active_mask):
            if not active:
                continue
            hessians.append(SparseInternalHessian(
                self.natoms + self.ndummies, idx, vals.copy()
            ))
        return SparseInternalHessians(hessians, self.ndof)

    def wrap(self, vec: np.ndarray) -> np.ndarray:
        """Wraps an internal coord. displacement vector into a valid domain."""
        # TODO: make this more robust to the addition of other arbitrary
        # internal coordinates.
        start = 0
        for name in self._names:
            if name == 'dihedrals':
                end = start + len(self.internals[name])
                break
            start += len(self.internals[name])
        vec[start:end] = (vec[start:end] + np.pi) % (2 * np.pi) - np.pi
        return vec

    def __iter__(self) -> Iterator[Internal]:
        for name in self._names:
            for coord in self.internals[name]:
                yield coord
            #for coord, active in zip(self.internals[name], self._active[name]):
            #    if active:
            #        yield coord

    def _get_neighbors(self, dx: np.ndarray) -> Iterator[np.ndarray]:
        pbc = self.atoms.pbc
        if self.cell is None or not np.all(self.cell == self.atoms.cell):
            self.cell = self.atoms.cell.array.copy()
            rcell, self.op = minkowski_reduce(
                complete_cell(self.cell), pbc=pbc
            )
            self.rcell = Cell(rcell)
        dx_sc = dx @ self.rcell.reciprocal().T
        offset = np.zeros(3, dtype=np.int32)
        for _ in range(2):
            offset += pbc * ((dx_sc - offset) // 1.).astype(np.int32)

        for ts in product(*[np.arange(-1 * p, p + 1) for p in pbc]):
            yield (np.array(ts) - offset) @ self.op

    def _find_mic(self, indices: Tuple[int, ...]) -> np.ndarray:
        ncvecs = np.zeros((len(indices) - 1, 3), dtype=np.int32)
        if not np.any(self.atoms.pbc):
            return ncvecs

        pos = self.all_atoms.positions
        dxs = np.array([
            pos[i] - pos[j] for i, j in zip(indices[1:], indices[:-1])
        ])

        for dx, ncvec in zip(dxs, ncvecs):
            vlen = np.infty
            for neighbor in self._get_neighbors(dx):
                trial = np.linalg.norm(dx + neighbor @ self.atoms.cell)
                if trial < vlen:
                    vlen = trial
                    ncvec[:] = neighbor
        return ncvecs

    def _get_ncvecs(
        self,
        indices: Tuple[int, ...],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None
    ) -> np.ndarray:
        if ncvecs is None:
            if mic is None or not mic:
                return np.zeros((len(indices) - 1, 3), dtype=np.int32)
            else:
                return self._find_mic(indices)
        else:
            if mic:
                raise ValueError(
                    "Minimum image convention (mic) requested, but explicit "
                    "periodic vectors (ncvecs) were also provided! These "
                    "keyword arguments are mutually exclusive."
                )
            return np.asarray(
                ncvecs,
                dtype=np.int32
            ).reshape((len(indices) - 1, 3))

    def get_principal_rotation_axes(
        self,
        indices: Tuple[int, ...]
    ) -> jnp.ndarray:
        """Calculates the principal axes of rotation of a cluster of atoms."""
        indices = np.asarray(indices, dtype=np.int32)
        pos = self.all_atoms.positions
        dx = pos[indices] - pos[indices].mean(0)
        Inertia = (
            (dx * dx).sum() * jnp.eye(3)
            - (dx[:, None, :] * dx[:, :, None]).sum(0)
        )
        _, rvecs = jnp.linalg.eigh(Inertia)
        return rvecs

    def add_dummy_to_internals(
        self,
        idx: int
    ) -> None:
        didx = self.dinds[idx]
        assert didx >= 0
        for i, trans in enumerate(self.internals['translations']):
            if idx in trans.indices:
                new_indices = (*trans.indices[:-1], didx)
                new_trans = Translation(new_indices, trans.dim)
                self.internals['translations'][i] = new_trans

        for i, rot in enumerate(self.internals['rotations']):
            if idx in rot.indices[:-1]:
                new_indices = (*rot.indices[:-1], didx)
                new_rot = Rotation(
                    new_indices, rot.axis,
                    self.all_atoms[new_indices].positions
                )
                self.internals['rotations'][i] = new_rot


class Constraints(BaseInternals):
    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        dinds: np.ndarray = None,
        ignore_rotation: bool = True,
    ) -> None:
        BaseInternals.__init__(self, atoms, dummies, dinds)
        self._targets = {key: [] for key in self._names}
        self._kind = {key: [] for key in self._names}
        self.ignore_rotation = ignore_rotation
        for ase_cons in atoms.constraints:
            self.merge_ase_constraint(ase_cons)

    def copy(self) -> 'Constraints':
        new = self.__class__(
            self.atoms, self.dummies, self.dinds, self.ignore_rotation
        )
        for name in self._names:
            new.internals[name] = self.internals[name].copy()
            new._targets[name] = self._targets[name].copy()
            new._active[name] = self._active[name].copy()
            new._kind[name] = self._kind[name].copy()
        return new

    @property
    def targets(self) -> np.ndarray:
        vec = []
        for key in self._names:
            vec += self._targets[key]
        return np.array(vec, dtype=np.float64)[self._active_indices]

    def residual(self) -> np.ndarray:
        """Calculates the constraint residual vector."""
        res = self.wrap(self.calc() - self.targets)
        if self.ignore_rotation and self.nrotations:
            res[-self.nrotations:] = 0.
        return res

    def disable_satisfied_inequalities(self) -> None:
        for name in self._names:
            for i, (coord, kind, target) in enumerate(zip(
                self.internals[name], self._kind[name], self._targets[name]
            )):
                if kind == 'lt' and coord.calc(self.all_atoms) <= target:
                    active = False
                elif kind == 'gt' and coord.calc(self.all_atoms) >= target:
                    active = False
                else:
                    active = True
                self._active[name][i] = active

    def validate_inequalities(self) -> bool:
        all_valid = True
        for name in self._names:
            for i, (coord, kind, target) in enumerate(zip(
                self.internals[name], self._kind[name], self._targets[name]
            )):
                if self._active[name][i]:
                    continue
                if kind == 'lt' and coord.calc(self.all_atoms) > target:
                    self._active[name][i] = True
                    all_valid = False
                elif kind == 'gt' and coord.calc(self.all_atoms) < target:
                    self._active[name][i] = True
                    all_valid = False
        return all_valid

    def fix_rotation(
        self,
        indices: Union[Tuple[int, ...], Rotation] = None,
        axis: int = None,
    ) -> None:
        if isinstance(indices, Rotation):
            if axis is not None:
                raise ValueError(
                    "'axis' keyword cannot be used with explicit Rotation"
                )
            new = indices
        else:
            if indices is None:
                indices = np.arange(len(self.all_atoms), dtype=np.int32)
            indices = np.asarray(indices, dtype=np.int32)
            if axis is None:
                for axis in range(3):
                    self.fix_rotation(indices, axis)
                return
            new = Rotation(
                indices,
                axis,
                self.all_atoms[indices].positions
            )
        try:
            _ = self.internals['rotations'].index(new)
        except ValueError:
            self.internals['rotations'].append(new)
            self._targets['rotations'].append(0.)
            self._active['rotations'].append(True)
            self._kind['rotations'].append('eq')
        else:
            raise DuplicateConstraintError(
                "This rotation has already been constrained!"
            )

    def fix_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None,
        target: float = None,
        replace_ok: bool = True,
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Translation'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            if np.isscalar(index):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                if target is not None:
                    raise ValueError(
                        '"target" keyword requires explicit "dim"!'
                    )
                for dim in range(3):
                    self.fix_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        if target is None:
            target = new.calc(self.all_atoms)
        try:
            idx = self.internals['translations'].index(new)
        except ValueError:
            self.internals['translations'].append(new)
            self._targets['translations'].append(target)
            self._active['translations'].append(True)
            self._kind['translations'].append('eq')
        else:
            if replace_ok:
                self._targets['translations'][idx] = target
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}"
                .format(new, self._targets['translations'][idx])
            )

    def _fix_internal(
        self,
        kind: TypeVar('Internal', bound=Internal),
        name: str,
        conv: float,
        indices: Union[Tuple[int, ...], Internal],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
        target: float = None,
        comparator: str = 'eq',
        replace_ok: bool = True,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        if target is None:
            target = new.calc(self.all_atoms)
        else:
            target *= conv
        try:
            idx = self.internals[name].index(new)
        except ValueError:
            self.internals[name].append(new)
            self._targets[name].append(target)
            self._active[name].append(True)
            self._kind[name].append(comparator)
        else:
            if replace_ok:
                self._targets[name][idx] = target
                self._kind[name][idx] = comparator
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}"
                .format(new, self._targets[name][idx]) / conv
            )

    fix_bond = partialmethod(_fix_internal, Bond, 'bonds', 1.)
    fix_angle = partialmethod(_fix_internal, Angle, 'angles', np.pi / 180.)
    fix_dihedral = partialmethod(
        _fix_internal, Dihedral, 'dihedrals', np.pi / 180.
    )

    def fix_other(
        self,
        coord: Internal,
        target: float = None,
        comparator: str = 'eq',
        replace_ok: bool = True,
    ) -> None:
        if target is None:
            target = coord.calc(self.all_atoms)
        try:
            idx = self.internals['other'].index(coord)
        except ValueError:
            self.internals['other'].append(coord)
            self._targets['other'].append(target)
            self._active['other'].append(True)
            self._kind['other'].append(comparator)
        else:
            if replace_ok:
                self._targets['other'][idx] = target
                self._kind['other'][idx] = comparator
                return
            raise DuplicateCoordinateError(
                "Coordinate {} is already fixed to target {}"
                .format(coord, self._targets['other'][idx])
            )

    def merge_ase_constraint(self, ase_cons: FixConstraint) -> None:
        if isinstance(ase_cons, FixAtoms):
            for index in ase_cons.index:
                try:
                    self.fix_translation(index)
                except DuplicateConstraintError:
                    pass
        elif isinstance(ase_cons, FixCom):
            try:
                self.fix_translation()
            except DuplicateConstraintError:
                pass
        elif isinstance(ase_cons, FixBondLengths):
            for i, indices in enumerate(ase_cons.pairs):
                if ase_cons.bondlengths is None:
                    target = None
                else:
                    target = ase_cons.bondlengths[i]
                try:
                    self.fix_bond(indices, mic=True, target=target)
                except DuplicateConstraintError:
                    pass
            return
        elif isinstance(ase_cons, FixCartesian):
            for dim, relaxed in enumerate(ase_cons.mask):
                if relaxed:
                    continue
                try:
                    self.fix_translation(ase_cons.a, dim=dim)
                except DuplicateConstraintError:
                    pass
        elif isinstance(ase_cons, FixInternals):
            for ase_cons_list, adder in zip(
                (ase_cons.bonds, ase_cons.angles, ase_cons.dihedrals),
                (self.fix_bond, self.fix_angle, self.fix_dihedral),
            ):
                for target, indices in ase_cons_list:
                    try:
                        adder(indices, target=target)
                    except DuplicateInternalError:
                        pass
            if ase_cons.bondcombos:
                raise RuntimeError(
                    "Sella currently does not support combination constraints."
                )
        else:
            raise RuntimeError(
                "Sella does not currently implement the ASE {} Constraint "
                "class.".format(ase_cons.__class__.__name__)
            )


class Internals(BaseInternals):
    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        atol: float = 15.,
        dinds: np.ndarray = None,
        cons: Constraints = None,
        allow_fragments: bool = False
    ) -> None:
        BaseInternals.__init__(self, atoms, dummies, dinds)
        self.atol = atol * np.pi / 180.
        self.forbidden = {key: [] for key in self._names}
        if cons is None:
            cons = Constraints(self.atoms, self.dummies, self.dinds)
        else:
            if (
                (dummies is not None and dummies is not cons.dummies)
                or (dinds is not None and dinds is not cons.dinds)
            ):
                raise RuntimeError(
                    "Constraints has inconsistent dummy atom definitions!"
                )
            self.dummies = cons.dummies
            self.dinds = cons.dinds
        self.cons = cons

        for kind, adder in zip(self._names, (
            self.add_translation, self.add_bond, self.add_angle,
            self.add_dihedral, self.add_rotation
        )):
            for coord in self.cons.internals[kind]:
                adder(coord)
        self.allow_fragments = allow_fragments

    def copy(self) -> 'Internals':
        new = self.__class__(
            self.atoms,
            self.dummies,
            self.atol * 180. / np.pi,
            self.dinds,
            self.cons.copy(),
            self.allow_fragments,
        )
        for name in self._names:
            new.internals[name] = self.internals[name].copy()
            new.forbidden[name] = self.forbidden[name].copy()
            new._active[name] = self._active[name].copy()
        return new

    def add_rotation(
        self,
        indices: Union[Tuple[int, ...], Rotation] = None,
        axis: int = None,
    ) -> None:
        if isinstance(indices, Rotation):
            if axis is not None:
                raise ValueError(
                    "'axis' keyword cannot be used with explicit Rotation"
                )
            new = indices
        else:
            if indices is None:
                indices = np.arange(len(self.all_atoms), dtype=np.int32)
            indices = np.array(indices, dtype=np.int32)
            if axis is None:
                for axis in range(3):
                    self.add_rotation(indices, axis)
                return
            new = Rotation(
                indices,
                axis,
                self.all_atoms[indices].positions
            )
        if (
            new in self.internals['rotations']
            or new in self.forbidden['rotations']
        ):
            raise DuplicateInternalError
        self.internals['rotations'].append(new)
        self._active['rotations'].append(True)

    def add_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Cart'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            elif isinstance(index, int):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                for dim in range(3):
                    self.add_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        if (
            new in self.internals['translations']
            or new in self.forbidden['translations']
        ):
            raise DuplicateInternalError
        self.internals['translations'].append(new)
        self._active['translations'].append(True)

    def _add_internal(
        self,
        kind: TypeVar('Internal', bound=Internal),
        name: str,
        indices: Union[Tuple[int, ...], Internal],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        if (
            new in self.internals[name]
            or new in self.forbidden[name]
        ):
            raise DuplicateInternalError
        self.internals[name].append(new)
        self._active[name].append(True)

    add_bond = partialmethod(_add_internal, Bond, 'bonds')
    add_angle = partialmethod(_add_internal, Angle, 'angles')
    add_dihedral = partialmethod(_add_internal, Dihedral, 'dihedrals')

    def forbid_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Cart'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            elif isinstance(index, int):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                for dim in range(3):
                    self.forbid_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        try:
            self.internals['translations'].remove(new)
        except ValueError:
            pass
        if new not in self.forbidden['translations']:
            self.forbidden['translations'].append(new)

    def _forbid_internal(
        self,
        kind: TypeVar('Internal', bound=Internal),
        name: str,
        indices: Union[Tuple[int, ...], Internal],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        try:
            self.forbidden[name].remove(new)
        except ValueError:
            pass
        if new not in self.forbidden[name]:
            self.forbidden[name].append(new)

    forbid_bond = partialmethod(_forbid_internal, Bond, 'bonds')
    forbid_angle = partialmethod(_forbid_internal, Angle, 'angles')
    forbid_dihedral = partialmethod(_forbid_internal, Dihedral, 'dihedrals')

    @staticmethod
    def flood_fill(
        index: int,
        nbonds: np.ndarray,
        c10y: np.ndarray,
        labels: np.ndarray,
        label: int
    ) -> None:
        for j in c10y[index, :nbonds[index]]:
            if labels[j] != label:
                labels[j] = label
                Internals.flood_fill(j, nbonds, c10y, labels, label)

    def find_all_bonds(
        self,
        nbond_cart_thr: int = 6,
        max_bonds: int = 20,
    ) -> None:
        rcov = covalent_radii[self.atoms.numbers]
        nbonds = np.zeros(self.natoms, dtype=np.int32)
        labels = -np.ones(self.natoms, dtype=np.int32)
        c10y = -np.ones((self.natoms, max_bonds), dtype=np.int32)

        for bond in self.internals['bonds']:
            i, j = bond.indices
            c10y[i, nbonds[i]] = j
            nbonds[i] += 1
            c10y[j, nbonds[j]] = i
            nbonds[j] += 1

        scale = 1.25
        first_run = True
        while True:
            # use flood fill algorithm to count the number of disconnected
            # fragments
            nlabels = 0
            labels[:] = -1
            for i in range(self.natoms):
                if labels[i] == -1:
                    labels[i] = nlabels
                    self.flood_fill(i, nbonds, c10y, labels, nlabels)
                    nlabels += 1
            # if there is only one fragment, then the internal coordinates
            # are complete, and we can stop
            if nlabels == 1:
                break
            if self.allow_fragments and not first_run:
                break

            # remove labels from atoms with no bonding partners
            labels[nbonds == 0] = -1

            for i, j in cwr(range(self.natoms), 2):
                # do not add a bond between atoms belonging to the same
                # bonding network fragment
                if labels[i] == labels[j] and labels[i] != -1:
                    continue
                dx = self.atoms.positions[j] - self.atoms.positions[i]
                for ts in self._get_neighbors(dx):
                    # self-bonding is only allowed if a non-zero periodic
                    # translation vector is used
                    if i == j and np.all(ts == np.array((0, 0, 0))):
                        continue
                    dist = np.linalg.norm(dx + ts @ self.atoms.cell)
                    if dist <= scale * (rcov[i] + rcov[j]):
                        # try to add the bond. it doesn't matter if the
                        # bond already exists.
                        try:
                            self.add_bond((i, j), ts)
                        except DuplicateInternalError:
                            continue
                        c10y[i, nbonds[i]] = j
                        nbonds[i] += 1
                        c10y[j, nbonds[j]] = i
                        nbonds[j] += 1
            first_run = False
            scale *= 1.05

        if self.allow_fragments and nlabels != 1:
            assert nlabels > 1
            groups = [[] for _ in range(nlabels)]
            for i, label in enumerate(labels):
                if label == -1:
                    # A lone atom not bonded to anything else
                    self.add_translation(i)
                else:
                    groups[label].append(i)
            for group in groups:
                if not group:
                    continue
                self.add_translation(group)
                self.add_rotation(group)

    def find_all_angles(
        self,
    ) -> None:
        bonds = [[] for _ in range(self.natoms)]
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i < self.natoms:
                bonds[i].append(bond)
            if j < self.natoms:
                bonds[j].append(bond.reverse())

        for j, jbonds in enumerate(bonds):
            linear = []
            for b1, b2 in combinations(jbonds, 2):
                new = b1 + b2
                assert new.indices[1] == j, new.indices
                if self.atol < new.calc(self.atoms) < np.pi - self.atol:
                    try:
                        self.add_angle(new)
                    except DuplicateInternalError:
                        pass
                else:
                    self.forbid_angle(new)
                    linear.append((b1, b2))
            #if linear and self.dinds[j] < 0:
            if linear:
                if len(jbonds) == 2:
                    # Add a dummy atom to an atom center with only 2 bonds
                    # sort bonds from shortest to longest to ensure
                    # permutational invariance
                    b1, b2 = sorted(jbonds, key=lambda x: x.calc(self.atoms))
                    # First try to take the cross product of the two bond
                    # vectors. These two vectors are close to collinear, and
                    # may be exactly collinear, so there's a backup strategy
                    # if this results in the zero-vector.
                    if self.dinds[j] < 0:
                        self.dinds[j] = self.natoms + self.ndummies
                        dx1 = -b1.calc_vec(self.atoms)
                        dx1 /= np.linalg.norm(dx1)
                        dx2 = b2.calc_vec(self.atoms)
                        dx2 /= np.linalg.norm(dx2)
                        dpos = np.cross(dx1, dx2)
                        dpos_norm = np.linalg.norm(dpos)
                        if dpos_norm < 1e-4:
                            # the aforementioned backup strategy
                            # pick the cartesian basis vector that is maximally
                            # orthogonal with the shorter of the two displacement
                            # vectors.
                            # note: this is not rotationally invariant, but
                            # there's not much we can do about that
                            dim = np.argmin(np.abs(dx1))
                            dpos[:] = 0.
                            dpos[dim] = 1.
                            dpos -= dx1 * (dpos @ dx1)
                            dpos /= np.linalg.norm(dpos)
                        else:
                            dpos /= dpos_norm
                        # Add the dummy atom
                        dpos += self.atoms.positions[j]
                        self.dummies += Atom('X', dpos)
                    # Create and fix dummy bond
                    dbond = Bond((j, self.dinds[j]))
                    self.cons.fix_bond(dbond, replace_ok=False)
                    self.add_bond(dbond)
                    # Fix one dummy angle
                    dangle1 = b1 + dbond
                    self.cons.fix_angle(dangle1, replace_ok=False)
                    dangle2 = b2 + dbond
                    self.cons.fix_angle(dangle2, replace_ok=False)
                    # Fix the improper dihedral and update relevant internals
                    if b2.indices[1] == j:
                        b2 = b2.reverse()
                    dbond2 = Bond((self.dinds[j], b2.indices[1]), b2.ncvecs)
                    dangle3 = dbond + dbond2
                    ddihedral = dangle1 + dangle3
                    self.add_dihedral(ddihedral)
                    self.add_dummy_to_internals(j)
                    self.cons.add_dummy_to_internals(j)
                    # Add relevant angles
                    for b1 in jbonds:
                        new = b1 + dbond
                        assert new.indices[1] == j
                        angle = new.calc(self.all_atoms)
                        if self.atol < angle < np.pi - self.atol:
                            try:
                                self.add_angle(new)
                            except DuplicateInternalError:
                                pass
                        else:
                            self.forbid_angle(new)
                else:
                    for b1, b2 in linear:
                        for b3 in jbonds:
                            if b3 in (b1, b2):
                                continue
                            indices = (
                                b1.indices[1], j, b3.indices[1], b2.indices[1]
                            )
                            ncvecs = (
                                -b1.ncvecs[0],
                                b3.ncvecs[0],
                                b2.ncvecs[0] - b3.ncvecs[0]
                            )
                            try:
                                self.add_dihedral(indices, ncvecs)
                            except DuplicateInternalError:
                                pass
                            break
                        else:
                            raise RuntimeError(
                                "Unable to find improper dihedral to replace "
                                "linear angle!"
                            )

    def find_all_dihedrals(self) -> None:
        for a1, a2 in combinations(self.internals['angles'], 2):
            try:
                new = a1 + a2
            except NoValidInternalError:
                continue
            # this is a dihedral that has the same exact atom as both
            # the first and last atom.
            if (
                new.indices[0] == new.indices[3]
                and np.all(np.sum(new.ncvecs, axis=0) == np.array((0, 0, 0)))
            ):
                continue
            try:
                self.add_dihedral(new)
            except DuplicateInternalError:
                continue

    def validate_basis(self) -> None:
        jac = self.jacobian()
        U, S, VT = np.linalg.svd(jac)
        ndeloc = np.sum(S > 1e-8)
        ndof = 3 * (self.natoms + self.ndummies) - 6
        if ndeloc != ndof:
            warnings.warn(
                f'{ndeloc} coords found! Expected {ndof}.'
            )

    def check_for_bad_internals(self) -> Optional[Dict[str, List[Internal]]]:
        bad = {'bonds': [], 'angles': []}
        for a in self.internals['angles']:
            if not (self.atol < a.calc(self.all_atoms) < np.pi - self.atol):
                bad['angles'].append(a)

        for ints in bad.values():
            if ints:
                return bad
        return None

    def _h0_bond(
        self,
        bond: Bond,
        Ab: float = 0.3601,
        Bb: float = 1.944,
    ) -> float:
        idx = np.asarray(bond.indices, dtype=np.int32)
        rcov = covalent_radii[self.all_atoms.numbers[idx]].sum()
        rij = bond.calc(self.all_atoms)
        h0 = Ab * np.exp(-Bb * (rij - rcov) / units.Bohr)
        return h0 * units.Hartree / units.Bohr**2

    def _h0_angle(
        self,
        angle: Angle,
        Aa: float = 0.089,
        Ba: float = 0.11,
        Ca: float = 0.44,
        Da: float = -0.42,
    ) -> float:
        bab, bbc = angle.split()
        idxab = np.asarray(bab.indices, dtype=np.int32)
        idxbc = np.asarray(bbc.indices, dtype=np.int32)
        rcovab = covalent_radii[self.all_atoms.numbers[idxab]].sum()
        rcovbc = covalent_radii[self.all_atoms.numbers[idxbc]].sum()
        rab = bab.calc(self.all_atoms)
        rbc = bbc.calc(self.all_atoms)
        h0 = (
            Aa + Ba * np.exp(-Ca * (rab + rbc - rcovab - rcovbc) / units.Bohr)
            / (rcovab * rcovbc / units.Bohr**2)**Da
        )
        return h0 * units.Hartree

    def _h0_dihedral(
        self,
        dihedral: Dihedral,
        nbonds: np.ndarray,
        At: float = 0.0015,
        Bt: float = 14.0,
        Ct: float = 2.85,
        Dt: float = 0.57,
        Et: float = 4.00,
    ) -> float:
        _, bbc = dihedral.split()[0].split()
        idx = np.asarray(bbc.indices, dtype=np.int32)
        rcovbc = covalent_radii[self.all_atoms.numbers[idx]].sum()
        rbc = bbc.calc(self.all_atoms)
        L = nbonds[idx].sum() - 2
        h0 = (
            At + Bt * L**Dt * np.exp(-Ct * (rbc - rcovbc) / units.Bohr)
            / (rbc * rcovbc / units.Bohr**2)**Et
        )
        return h0 * units.Hartree

    def guess_hessian(self, h0cart=70.) -> np.ndarray:
        nbonds = np.zeros(len(self.all_atoms), dtype=np.int32)
        h0 = np.zeros(self.nint, dtype=np.float64)
        idx = 0
        for trans in self.internals['translations']:
            h0[idx] = h0cart
            idx += 1
        for bond in self.internals['bonds']:
            h0[idx] = self._h0_bond(bond)
            idx += 1
            # count number of bonds per atom for dihedral later
            i, j = bond.indices
            nbonds[i] += 1
            nbonds[j] += 1
        for angle in self.internals['angles']:
            h0[idx] = self._h0_angle(angle)
            idx += 1
        for dihedral in self.internals['dihedrals']:
            h0[idx] = self._h0_dihedral(dihedral, nbonds)
            idx += 1
        # remaining degrees of freedom are rotations.
        # No idea what a good curvature is for these
        h0[idx:] = 1.
        return np.diag(np.abs(h0))
