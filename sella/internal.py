from typing import (
    Tuple, Callable, Iterator, Union, TypeVar, Optional, List, Dict, Type
)
from itertools import (
    product,
    combinations,
    combinations_with_replacement as cwr
)
from functools import partialmethod
import warnings

from scipy import sparse
from scipy.linalg import svdvals
import numpy as np
from ase import Atom, Atoms, units
from ase.cell import Cell
from ase.geometry import complete_cell, minkowski_reduce
from ase.data import covalent_radii
from ase.constraints import (
    FixConstraint, FixAtoms, FixCom, FixBondLengths, FixCartesian, FixInternals
)

import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, custom_jvp, vmap, jvp, device_get

from sella.linalg import (
    SparseInternalJacobian, SparseInternalHessian, SparseInternalHessians,
    SparseInternalHessiansSkeleton
)


# =============================================================================
# Lightweight atoms-like wrapper for efficient coordinate calculations
# =============================================================================
# Creating ASE Atoms objects has significant overhead (Atoms.__init__ validates
# positions, sets up constraints, etc.). This lightweight wrapper provides just
# the positions and cell attributes needed for coordinate calculations, reducing
# Atoms.__init__ calls from ~1258 to ~266 per optimization run (~79% reduction).
# =============================================================================

class LightAtoms:
    """Lightweight wrapper providing positions and cell without Atoms overhead."""
    __slots__ = ('positions', 'cell')

    def __init__(self, positions: np.ndarray, cell: np.ndarray) -> None:
        self.positions = positions
        self.cell = cell


# =============================================================================
# Vectorized (batched) internal coordinate functions using jax.vmap
# =============================================================================
# These compute gradients/hessians for ALL coordinates of a given type at once,
# avoiding Python loop overhead. JAX's vmap automatically vectorizes over the
# batch dimension, providing significant speedup for coordinate calculations.
# =============================================================================

def _bond_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Bond length: pos shape (2, 3), tvec shape (1, 3)"""
    return jnp.linalg.norm(pos[1] - pos[0] + tvec[0])


def _angle_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Angle value: pos shape (3, 3), tvec shape (2, 3)"""
    dx1 = -(pos[1] - pos[0] + tvec[0])
    dx2 = pos[2] - pos[1] + tvec[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    # Clamp to avoid NaN from arccos
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


def _dihedral_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Dihedral angle: pos shape (4, 3), tvec shape (3, 3)"""
    dx1 = pos[1] - pos[0] + tvec[0]
    dx2 = pos[2] - pos[1] + tvec[1]
    dx3 = pos[3] - pos[2] + tvec[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)


# Batched gradient functions: input shapes (n_coords, n_atoms, 3), (n_coords, n_vecs, 3)
# Output shapes: (n_coords, n_atoms, 3)
_bond_grad_batched = jit(vmap(grad(_bond_value, argnums=0), in_axes=(0, 0)))
_angle_grad_batched = jit(vmap(grad(_angle_value, argnums=0), in_axes=(0, 0)))
_dihedral_grad_batched = jit(vmap(grad(_dihedral_value, argnums=0), in_axes=(0, 0)))

# Batched value functions
_bond_value_batched = jit(vmap(_bond_value, in_axes=(0, 0)))
_angle_value_batched = jit(vmap(_angle_value, in_axes=(0, 0)))
_dihedral_value_batched = jit(vmap(_dihedral_value, in_axes=(0, 0)))

# Batched hessian functions: output shapes (n_coords, n_atoms, 3, n_atoms, 3)
_bond_hess_batched = jit(vmap(jacfwd(grad(_bond_value, argnums=0), argnums=0), in_axes=(0, 0)))
_angle_hess_batched = jit(vmap(jacfwd(grad(_angle_value, argnums=0), argnums=0), in_axes=(0, 0)))
_dihedral_hess_batched = jit(vmap(jacfwd(grad(_dihedral_value, argnums=0), argnums=0), in_axes=(0, 0)))

# =============================================================================
# Hessian-vector product (HVP) functions using forward-over-reverse mode
# =============================================================================
# These compute H @ v directly without materializing the full Hessian matrix.
# Uses jvp(grad(f), x, v) which is O(n) instead of O(n²) for forming full Hessian.
# =============================================================================

def _bond_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single bond without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_bond_value, argnums=0), primals, tangents)
    return hvp_result


def _angle_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single angle without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_angle_value, argnums=0), primals, tangents)
    return hvp_result


def _dihedral_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single dihedral without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_dihedral_value, argnums=0), primals, tangents)
    return hvp_result


# Batched HVP functions: compute H @ v for all coords at once
# Input shapes: pos (n_coords, n_atoms, 3), tvec (n_coords, n_vecs, 3), tangent (n_coords, n_atoms, 3)
# Output shapes: (n_coords, n_atoms, 3)
_bond_hvp_batched = jit(vmap(_bond_hvp_single, in_axes=(0, 0, 0)))
_angle_hvp_batched = jit(vmap(_angle_hvp_single, in_axes=(0, 0, 0)))
_dihedral_hvp_batched = jit(vmap(_dihedral_hvp_single, in_axes=(0, 0, 0)))


# =============================================================================
# Cell-derivative functions for unit cell optimization
# =============================================================================
# These compute derivatives of internal coordinates with respect to cell matrix.
# Used for coupled atomic + cell optimization in periodic systems.
#
# The chain rule is: d(coord)/d(cell) = d(coord)/d(tvec) @ d(tvec)/d(cell)
# Since tvec = ncvec @ cell, we have d(tvec)/d(cell) = ncvec (Kronecker structure)
# =============================================================================

def _bond_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Bond length with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (1, 3) @ (3, 3) -> (1, 3)
    return jnp.linalg.norm(pos[1] - pos[0] + tvec[0])


def _angle_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Angle with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (2, 3) @ (3, 3) -> (2, 3)
    dx1 = -(pos[1] - pos[0] + tvec[0])
    dx2 = pos[2] - pos[1] + tvec[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


def _dihedral_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Dihedral angle with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (3, 3) @ (3, 3) -> (3, 3)
    dx1 = pos[1] - pos[0] + tvec[0]
    dx2 = pos[2] - pos[1] + tvec[1]
    dx3 = pos[3] - pos[2] + tvec[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)


# Single-coordinate cell gradients: output shape (3, 3) for d(coord)/d(cell)
_bond_cell_grad_single = jit(grad(_bond_with_cell, argnums=2))
_angle_cell_grad_single = jit(grad(_angle_with_cell, argnums=2))
_dihedral_cell_grad_single = jit(grad(_dihedral_with_cell, argnums=2))

# Batched cell gradients: input (n_coords, n_atoms, 3), (n_coords, n_vecs, 3), (3, 3)
# Output: (n_coords, 3, 3)
# Note: cell is NOT batched (same cell for all coords), so in_axes=(0, 0, None)
_bond_cell_grad_batched = jit(vmap(_bond_cell_grad_single, in_axes=(0, 0, None)))
_angle_cell_grad_batched = jit(vmap(_angle_cell_grad_single, in_axes=(0, 0, None)))
_dihedral_cell_grad_batched = jit(vmap(_dihedral_cell_grad_single, in_axes=(0, 0, None)))


# =============================================================================
# Block size for GPU/SIMD efficiency
# =============================================================================
# Padding arrays to multiples of BLOCK_SIZE improves GPU performance through
# better warp-level parallelism and memory coalescing. Also reduces JAX JIT
# recompilation when array sizes change.
# =============================================================================
BLOCK_SIZE = 64


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


class Coordinate:
    nindices = None
    kwargs = None

    def __init__(
        self,
        indices: Tuple[int, ...],
    ) -> None:
        if self.nindices is not None:
            assert len(indices) == self.nindices
        self.indices = np.array(indices, dtype=np.int32)
        self.kwargs = dict()

    def reverse(self) -> 'Coordinate':
        raise NotImplementedError

    def __eq__(self, other: 'Coordinate') -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self.indices) != len(other.indices):
            return False
        if np.all(self.indices == other.indices):
            return True
        return False

    def __add__(self, other: 'Coordinate') -> 'Coordinate':
        raise NotImplementedError

    def split(self) -> Tuple['Coordinate', 'Coordinate']:
        raise NotImplementedError

    def __repr__(self) -> str:
        out = [f'indices={self.indices}']
        out += [f'{key}={val}' for key, val in self.kwargs.items()]
        str_out = ', '.join(out)
        return f'{self.__class__.__name__}({str_out})'

    @staticmethod
    def _eval0(pos: jnp.ndarray, **kwargs) -> float:
        raise NotImplementedError

    @staticmethod
    def _eval1(pos: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _eval2(pos: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    def calc(self, atoms: Atoms) -> float:
        return float(self._eval0(
            atoms.positions[self.indices], **self.kwargs
        ))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        return np.array(self._eval1(
            atoms.positions[self.indices], **self.kwargs
        ))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        return np.array(self._eval2(
            atoms.positions[self.indices], **self.kwargs
        ))

    def _check_derivative(
        self, atoms: Atoms, delta: float, atol: float, order: int
    ) -> bool:
        if order == 1:
            derivative = 'Gradient'
            f0 = self.calc
            f1 = self.calc_gradient
        elif order == 2:
            derivative = 'Hessian'
            f0 = self.calc_gradient
            f1 = self.calc_hessian
        else:
            raise ValueError(f'Order {order} gradients are not implemented')

        atoms0 = atoms.copy()
        g_ref = f1(atoms0)
        g_numer = np.zeros_like(g_ref)
        atoms = atoms0.copy()
        for i, idx in enumerate(self.indices):
            for j in range(3):
                atoms.positions[idx, j] = atoms0.positions[idx, j] + delta
                fplus = f0(atoms)
                atoms.positions[idx, j] = atoms0.positions[idx, j] - delta
                fminus = f0(atoms)
                g_numer[i, j] = (fplus - fminus) / (2 * delta)
                atoms.positions[idx, j] = atoms0.positions[idx, j]
        if np.max(np.abs(g_numer - g_ref)) > atol:
            warnings.warn(f'{derivative}s for {self} failed numerical test!')
            return False
        return True

    def check_gradient(
        self, atoms: Atoms, delta: float = 1e-4, atol: float = 1e-6
    ) -> bool:
        return self._check_derivative(atoms, delta, atol, order=1)

    def check_hessian(
        self, atoms: Atoms, delta: float = 1e-4, atol: float = 1e-6
    ) -> bool:
        return self._check_derivative(atoms, delta, atol, order=2)


class Internal(Coordinate):
    union = None
    diff = None

    def __init__(
        self,
        indices: Tuple[int, ...],
        ncvecs: Tuple[IVec, ...] = None
    ) -> None:
        Coordinate.__init__(self, indices)

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
        self.kwargs['ncvecs'] = ncvecs

    def reverse(self) -> 'Internal':
        return self.__class__(self.indices[::-1], -self.kwargs['ncvecs'][::-1])

    def __eq__(self, other: object) -> bool:
        if not Coordinate.__eq__(self, other):
            return False
        srev = self.reverse()
        if not Coordinate.__eq__(srev, other):
            return False
        if np.all(self.kwargs['ncvecs'] == other.kwargs['ncvecs']):
            return True
        if np.all(srev.kwargs['ncvecs'] == other.kwargs['ncvecs']):
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
                and np.all(s.kwargs['ncvecs'][1:] == o.kwargs['ncvecs'][:-1])
            ):
                new_indices = [*s.indices, o.indices[-1]]
                new_ncvecs = [*s.kwargs['ncvecs'], o.kwargs['ncvecs'][-1]]
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
            self.diff(self.indices[:-1], self.kwargs['ncvecs'][:-1]),
            self.diff(self.indices[1:], self.kwargs['ncvecs'][1:])
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
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return float(self._eval0(atoms.positions[self.indices], tvecs))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return np.array(self._eval1(atoms.positions[self.indices], tvecs))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return np.array(self._eval2(atoms.positions[self.indices], tvecs))

    @staticmethod
    def _eval_cell_grad(
        pos: jnp.ndarray, ncvecs: jnp.ndarray, cell: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute gradient of coordinate with respect to cell matrix.

        Must be overridden in subclasses (Bond, Angle, Dihedral).
        Returns shape (3, 3) for d(coord)/d(cell).
        """
        raise NotImplementedError

    def calc_cell_gradient(self, atoms: Atoms) -> np.ndarray:
        """Compute gradient of this coordinate w.r.t. cell matrix.

        Returns:
            np.ndarray: Shape (3, 3) array of d(coord)/d(cell[i,j])
        """
        ncvecs = jnp.asarray(self.kwargs['ncvecs'], dtype=np.float64)
        cell = jnp.asarray(
            atoms.cell.array,
            dtype=np.float64
        )
        pos = jnp.asarray(atoms.positions[self.indices], dtype=np.float64)
        return np.array(self._eval_cell_grad(pos, ncvecs, cell))


def _translation(
    pos: jnp.ndarray,
    dim: int,
) -> float:
    return pos[:, dim].mean()


class Translation(Coordinate):
    def __init__(
        self,
        indices: Tuple[int, ...],
        dim: int,
    ) -> None:
        Coordinate.__init__(self, indices)
        self.kwargs['dim'] = dim

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.kwargs['dim'] != other.kwargs['dim']:
            return False
        if set(self.indices) != set(other.indices):
            return False
        return True

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
    # Use where instead of sign: sign(0)=0 in JAX which zeros the entire
    # quaternion for linear molecules, making all rotation Jacobians zero.
    return q * jnp.where(q[0] >= 0, 1.0, -1.0)


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


def _rotation_3axis_masked(
    pos: jnp.ndarray, refpos: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """All 3 rotation values for one fragment, padded + masked.

    Returns an array of length 3 (one value per rotation axis). The
    fragment is assumed centered: ``refpos`` is already the centered
    reference, padded rows are zero. ``mask`` is 1 on real atoms and 0
    on padding so the centroid divides by the actual atom count.
    """
    pos_eff = pos * mask[:, None]
    natoms = jnp.sum(mask)
    centroid = jnp.sum(pos_eff, axis=0) / natoms
    dx = (pos - centroid) * mask[:, None]
    refpos_centered = refpos * mask[:, None]
    R = dx.T @ refpos_centered
    Rtr = jnp.trace(R)
    Ftop = jnp.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]])
    F = jnp.block([
        [Rtr, Ftop[None, :]],
        [Ftop[:, None], -Rtr * jnp.eye(3) + R + R.T],
    ])
    q = eigh_rightmost(F)
    q = q * jnp.where(q[0] >= 0, 1.0, -1.0)
    a = asinc(q[0])
    return jnp.stack([2 * q[1] * a, 2 * q[2] * a, 2 * q[3] * a])


_rotation_3axis_hess_batched = jit(vmap(
    jacfwd(jacfwd(_rotation_3axis_masked, argnums=0), argnums=0),
    in_axes=(0, 0, 0)
))


_rotation_3axis_value_batched = jit(vmap(
    _rotation_3axis_masked,
    in_axes=(0, 0, 0)
))


_rotation_3axis_grad_batched = jit(vmap(
    jacfwd(_rotation_3axis_masked, argnums=0),
    in_axes=(0, 0, 0)
))


def _rotation_3axis_hvp(pos, refpos, mask, v):
    """HVP for one fragment, all 3 axes at once.

    Returns shape (3, N, 3) — the directional derivative of the
    Jacobian (3, N, 3) along v (N, 3).
    """
    primals = (pos,)
    tangents = (v,)
    _, hvp = jvp(
        lambda p: jacfwd(_rotation_3axis_masked, argnums=0)(p, refpos, mask),
        primals, tangents
    )
    return hvp


_rotation_3axis_hvp_batched_jit = jit(
    vmap(_rotation_3axis_hvp, in_axes=(0, 0, 0, 0))
)


def _rotation_hess_padded_vmap(positions, refpos_padded, mask):
    """JIT+vmap'd Hessian for all rotations of one fragment, batched.

    positions: (B, N_max, 3) — fragment positions, padded
    refpos_padded: (B, N_max, 3) — fragment refpos, centered+padded
    mask: (B, N_max) — 1 on real atoms, 0 on padding

    Returns: (B, 3, N_max, 3, N_max, 3) — per-fragment, per-axis, full
    NxN Hessian with padded rows/cols zeroed out.
    """
    return _rotation_3axis_hess_batched(positions, refpos_padded, mask)


def _rotation_hvp(
    pos: jnp.ndarray,
    axis: int,
    refpos: jnp.ndarray,
    tangent: jnp.ndarray
) -> jnp.ndarray:
    """Compute Hessian @ tangent for rotation without forming the full Hessian.

    Uses forward-over-reverse mode autodiff: jvp(grad(f)) which is O(N) instead
    of O(N²) for materializing the full Hessian via jacfwd(jacfwd(f)).
    """
    primals = (pos,)
    tangents = (tangent,)
    # grad returns gradient w.r.t. pos; jvp computes directional derivative
    _, hvp_result = jvp(
        lambda p: grad(_rotation, argnums=0)(p, axis, refpos),
        primals,
        tangents
    )
    return hvp_result


_rotation_hvp_jit = jit(_rotation_hvp, static_argnums=(1,))


class Rotation(Coordinate):
    def __init__(
        self,
        indices: Tuple[int, ...],
        axis: int,
        refpos: np.ndarray,
    ) -> None:
        assert len(indices) >= 2
        Coordinate.__init__(self, indices)
        self.kwargs['axis'] = axis
        self.kwargs['refpos'] = refpos.copy() - refpos.mean(0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.kwargs['axis'] != other.kwargs['axis']:
            return False
        if len(self.indices) != len(other.indices):
            return False
        if set(self.indices) != set(other.indices):
            return False
        if not np.allclose(self.kwargs['refpos'], other.kwargs['refpos']):
            return False
        return True

    _eval0 = staticmethod(jit(_rotation))
    _eval1 = staticmethod(jit(jacfwd(_rotation, argnums=0)))
    _eval2 = staticmethod(jit(jacfwd(jacfwd(_rotation, argnums=0), argnums=0)))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        result = np.array(self._eval2(
            atoms.positions[self.indices], **self.kwargs
        ))
        # For linear molecules, the quaternion eigenvalue problem in
        # _rotation_q has degenerate eigenvalues, producing NaN second
        # derivatives. The corresponding rotation is physically
        # meaningless (zero Jacobian), so replacing NaN with 0 is exact.
        if np.any(np.isnan(result)):
            np.nan_to_num(result, copy=False)
        return result


def _displacement(
    pos: jnp.ndarray,
    refpos: jnp.ndarray,
    W: jnp.ndarray
) -> float:
    dx = (pos - refpos).ravel()
    return dx @ W @ dx


class Displacement(Coordinate):
    def __init__(
        self,
        indices: np.ndarray,
        refpos: np.ndarray,
        W: np.ndarray,
    ) -> None:
        Coordinate.__init__(self, indices)
        self.kwargs['refpos'] = refpos.copy()
        self.kwargs['W'] = W.copy()

    def __eq__(self, other: Coordinate) -> bool:
        if not Coordinate.__eq__(self, other):
            return False
        return np.allclose(self.kwargs['refpos'], other.kwargs['refpos'])

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
    _eval_cell_grad = staticmethod(_bond_cell_grad_single)

    def calc_vec(self, atoms: Atoms) -> np.ndarray:
        tvecs = np.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        i, j = self.indices
        return atoms.positions[j] - atoms.positions[i] + tvecs[0]


def _angle(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    dx1 = -(pos[1] - pos[0] + tvecs[0])
    dx2 = pos[2] - pos[1] + tvecs[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    # Clamp to avoid NaN from arccos due to floating-point errors
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


class Angle(Internal):
    nindices = 3
    _eval0 = staticmethod(jit(_angle))
    _eval1 = staticmethod(_gradient(_angle))
    _eval2 = staticmethod(_hessian(_angle))
    _eval_cell_grad = staticmethod(_angle_cell_grad_single)


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
    _eval_cell_grad = staticmethod(_dihedral_cell_grad_single)


Bond.union = Angle
Angle.union = Dihedral
Angle.diff = Bond
Dihedral.diff = Angle


def make_internal(
    name: str,
    fun: Callable[..., float],
    nindices: int,
    use_jit: bool = True,
    jac: Callable[..., jnp.ndarray] = None,
    hess: Callable[..., jnp.ndarray] = None,
    **kwargs,
) -> Type[Coordinate]:
    if jac is None:
        jac = _gradient(fun)
    if hess is None:
        hess = _hessian(fun)

    if use_jit:
        fun = jit(fun)
        jac = jit(jac)
        hess = jit(hess)

    return type(name, (Coordinate,), dict(
        nindices=nindices,
        kwargs=kwargs,
        _eval0=staticmethod(fun),
        _eval1=staticmethod(jac),
        _eval2=staticmethod(hess)
    ))


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
        self._cache_version = 0

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

        # Cache atom count (doesn't change during optimization)
        self._natoms = len(atoms)

        self.internals = {key: [] for key in self._names}
        self._internals_set = {key: set() for key in self._names}
        self._active = {key: [] for key in self._names}
        self.cell = None
        self.rcell = None
        self._rcell_reciprocal_T = None
        self.op = None
        self._hessian_skeleton = None

        # Batched arrays for vectorized computation (built lazily)
        self._batched_arrays_valid = False

        # Lazy caches.
        self._tvecs_cache = None  # set to {'cell_hash': ..., 'tvecs': ...} on first build
        self._hvp_buf = None  # reusable buffer for hessian_rdot output

    @property
    def natoms(self) -> int:
        return self._natoms

    @property
    def ndummies(self) -> int:
        return len(self.dummies)

    @property
    def ndof(self) -> int:
        return 3 * (self._natoms + len(self.dummies))

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
    def nother(self) -> int:
        return sum(self._active['other'])

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
    def all_positions(self) -> np.ndarray:
        """Get combined positions without creating an Atoms object.

        Cached on ``self._cache['all_positions']`` so repeated reads
        within a single position evaluation reuse the same vstack.
        ``_cache_check`` clears the cache whenever positions change.
        """
        if self.ndummies == 0:
            return self.atoms.positions
        cached = self._cache.get('all_positions')
        if cached is not None:
            return cached
        merged = np.vstack([self.atoms.positions, self.dummies.positions])
        self._cache['all_positions'] = merged
        return merged

    @property
    def all_atoms(self) -> Atoms:
        return self.atoms + self.dummies

    @property
    def light_atoms(self) -> LightAtoms:
        """Get lightweight atoms-like object for coordinate calculations."""
        cell = self.atoms.cell.array
        return LightAtoms(self.all_positions, cell)

    def _cache_check(self) -> None:
        # we are comparing the current atomic positions to what they were
        # the last time a property was calculated. These positions are floats,
        # but we use a strict equality check to compare to avoid subtle bugs
        # that might occur during fine-resolution geodesic steps.
        if self.ndummies == 0:
            current_pos = self.atoms.positions
        else:
            current_pos = np.vstack([self.atoms.positions, self.dummies.positions])
        if (
            self._lastpos is None
            or np.any(current_pos != self._lastpos)
        ):
            self._cache = dict()
            self._lastpos = current_pos.copy()
            self._cache_version += 1
        # Park the freshly-merged positions in the cache so the next
        # all_positions access doesn't redo the vstack.
        if self.ndummies > 0:
            self._cache.setdefault('all_positions', self._lastpos)

    def _build_batched_arrays(self) -> None:
        """Build batched index arrays for vectorized computation.

        Arrays are padded to multiples of BLOCK_SIZE for GPU/SIMD efficiency.
        Masks are stored to filter results back to actual sizes.
        """
        if self._batched_arrays_valid:
            return

        def pad_to_block(n: int) -> int:
            """Round up to nearest multiple of BLOCK_SIZE."""
            return ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

        # Build arrays for bonds
        bonds = self.internals['bonds']
        n_bonds = len(bonds)
        if n_bonds > 0:
            n_bonds_padded = pad_to_block(n_bonds)
            # Original (unpadded) arrays for indexing
            self._bond_indices = np.array([b.indices for b in bonds], dtype=np.int32)
            self._bond_ncvecs = np.array(
                [b.kwargs['ncvecs'] for b in bonds], dtype=np.int32
            )
            # Padded arrays for batch computation
            self._bond_indices_padded = np.zeros((n_bonds_padded, 2), dtype=np.int32)
            self._bond_ncvecs_padded = np.zeros((n_bonds_padded, 1, 3), dtype=np.int32)
            self._bond_indices_padded[:n_bonds] = self._bond_indices
            self._bond_ncvecs_padded[:n_bonds] = self._bond_ncvecs
            self._bond_mask = np.zeros(n_bonds_padded, dtype=np.float64)
            self._bond_mask[:n_bonds] = 1.0
            self._n_bonds_actual = n_bonds
        else:
            self._bond_indices = np.empty((0, 2), dtype=np.int32)
            self._bond_ncvecs = np.empty((0, 1, 3), dtype=np.int32)
            self._bond_indices_padded = np.empty((0, 2), dtype=np.int32)
            self._bond_ncvecs_padded = np.empty((0, 1, 3), dtype=np.int32)
            self._bond_mask = np.empty(0, dtype=np.float64)
            self._n_bonds_actual = 0

        # Build arrays for angles
        angles = self.internals['angles']
        n_angles = len(angles)
        if n_angles > 0:
            n_angles_padded = pad_to_block(n_angles)
            self._angle_indices = np.array([a.indices for a in angles], dtype=np.int32)
            self._angle_ncvecs = np.array(
                [a.kwargs['ncvecs'] for a in angles], dtype=np.int32
            )
            self._angle_indices_padded = np.zeros((n_angles_padded, 3), dtype=np.int32)
            self._angle_ncvecs_padded = np.zeros((n_angles_padded, 2, 3), dtype=np.int32)
            self._angle_indices_padded[:n_angles] = self._angle_indices
            self._angle_ncvecs_padded[:n_angles] = self._angle_ncvecs
            self._angle_mask = np.zeros(n_angles_padded, dtype=np.float64)
            self._angle_mask[:n_angles] = 1.0
            self._n_angles_actual = n_angles
        else:
            self._angle_indices = np.empty((0, 3), dtype=np.int32)
            self._angle_ncvecs = np.empty((0, 2, 3), dtype=np.int32)
            self._angle_indices_padded = np.empty((0, 3), dtype=np.int32)
            self._angle_ncvecs_padded = np.empty((0, 2, 3), dtype=np.int32)
            self._angle_mask = np.empty(0, dtype=np.float64)
            self._n_angles_actual = 0

        # Build arrays for dihedrals
        dihedrals = self.internals['dihedrals']
        n_dihedrals = len(dihedrals)
        if n_dihedrals > 0:
            n_dihedrals_padded = pad_to_block(n_dihedrals)
            self._dihedral_indices = np.array(
                [d.indices for d in dihedrals], dtype=np.int32
            )
            self._dihedral_ncvecs = np.array(
                [d.kwargs['ncvecs'] for d in dihedrals], dtype=np.int32
            )
            self._dihedral_indices_padded = np.zeros((n_dihedrals_padded, 4), dtype=np.int32)
            self._dihedral_ncvecs_padded = np.zeros((n_dihedrals_padded, 3, 3), dtype=np.int32)
            self._dihedral_indices_padded[:n_dihedrals] = self._dihedral_indices
            self._dihedral_ncvecs_padded[:n_dihedrals] = self._dihedral_ncvecs
            self._dihedral_mask = np.zeros(n_dihedrals_padded, dtype=np.float64)
            self._dihedral_mask[:n_dihedrals] = 1.0
            self._n_dihedrals_actual = n_dihedrals
        else:
            self._dihedral_indices = np.empty((0, 4), dtype=np.int32)
            self._dihedral_ncvecs = np.empty((0, 3, 3), dtype=np.int32)
            self._dihedral_indices_padded = np.empty((0, 4), dtype=np.int32)
            self._dihedral_ncvecs_padded = np.empty((0, 3, 3), dtype=np.int32)
            self._dihedral_mask = np.empty(0, dtype=np.float64)
            self._n_dihedrals_actual = 0

        # Precompute flat column indices for direct scatter in hessian_rdot.
        # For bond (a,b), the non-zero columns in the (ndof,) output are
        # [3a, 3a+1, 3a+2, 3b, 3b+1, 3b+2].  Analogous for angles (9 cols)
        # and dihedrals (12 cols).  These are topology-dependent and
        # invalidated together with the rest of the batched arrays.
        offsets = np.arange(3)
        if self._n_bonds_actual > 0:
            bi = self._bond_indices  # (n_bonds, 2)
            self._bond_flat_cols = np.concatenate([
                bi[:, k:k+1] * 3 + offsets for k in range(2)
            ], axis=1)  # (n_bonds, 6)
        else:
            self._bond_flat_cols = np.empty((0, 6), dtype=np.intp)

        if self._n_angles_actual > 0:
            ai = self._angle_indices  # (n_angles, 3)
            self._angle_flat_cols = np.concatenate([
                ai[:, k:k+1] * 3 + offsets for k in range(3)
            ], axis=1)  # (n_angles, 9)
        else:
            self._angle_flat_cols = np.empty((0, 9), dtype=np.intp)

        if self._n_dihedrals_actual > 0:
            di = self._dihedral_indices  # (n_dihedrals, 4)
            self._dihedral_flat_cols = np.concatenate([
                di[:, k:k+1] * 3 + offsets for k in range(4)
            ], axis=1)  # (n_dihedrals, 12)
        else:
            self._dihedral_flat_cols = np.empty((0, 12), dtype=np.intp)

        # Build CSR structure for sparse hessian_rdot output.
        # Bonds/angles/dihedrals have fixed nnz per row (6/9/12).
        # Translations have zero rows. Rotations/other are dense (ndof cols).
        ndof = self.ndof
        n_trans = len(self.internals['translations'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])
        n_active = (n_trans + self._n_bonds_actual + self._n_angles_actual
                    + self._n_dihedrals_actual + n_other + n_rot)

        col_blocks = []
        nnz_per_row = []

        # Translations: zero rows
        for _ in range(n_trans):
            nnz_per_row.append(0)

        # Bonds: 6 nnz per row
        if self._n_bonds_actual > 0:
            col_blocks.append(self._bond_flat_cols.ravel())
            nnz_per_row.extend([6] * self._n_bonds_actual)

        # Angles: 9 nnz per row
        if self._n_angles_actual > 0:
            col_blocks.append(self._angle_flat_cols.ravel())
            nnz_per_row.extend([9] * self._n_angles_actual)

        # Dihedrals: 12 nnz per row
        if self._n_dihedrals_actual > 0:
            col_blocks.append(self._dihedral_flat_cols.ravel())
            nnz_per_row.extend([12] * self._n_dihedrals_actual)

        # Other/rotations: dense rows (ndof cols each)
        for _ in range(n_other + n_rot):
            col_blocks.append(np.arange(ndof))
            nnz_per_row.append(ndof)

        self._csr_indptr = np.zeros(n_active + 1, dtype=np.int32)
        np.cumsum(nnz_per_row, out=self._csr_indptr[1:])
        self._csr_indices = np.concatenate(col_blocks).astype(np.int32) if col_blocks else np.empty(0, dtype=np.int32)
        self._csr_data = np.zeros(len(self._csr_indices), dtype=np.float64)
        self._csr_n_active = n_active
        # Precompute data offset for each section
        self._csr_bond_offset = n_trans * 0  # bonds start after translations (0 nnz)
        self._csr_angle_offset = self._csr_bond_offset + self._n_bonds_actual * 6
        self._csr_dih_offset = self._csr_angle_offset + self._n_angles_actual * 9
        self._csr_other_offset = self._csr_dih_offset + self._n_dihedrals_actual * 12

        self._batched_arrays_valid = True

    def _get_cached_tvecs(self, cell: np.ndarray) -> Dict[str, np.ndarray]:
        """Get cached translation vectors for cell, computing if necessary.

        The tvecs (ncvecs @ cell) are constant for a given cell, so we cache
        them to avoid redundant matrix multiplications during ODE integration.

        Returns both unpadded tvecs (for indexing) and padded tvecs (for batch ops).
        """
        cell_hash = cell.tobytes()
        if self._tvecs_cache is not None and self._tvecs_cache['cell_hash'] == cell_hash:
            return self._tvecs_cache['tvecs']

        self._build_batched_arrays()
        tvecs = {}

        # Unpadded tvecs (for result indexing)
        if len(self._bond_indices) > 0:
            tvecs['bonds'] = self._bond_ncvecs @ cell
        else:
            tvecs['bonds'] = np.empty((0, 1, 3), dtype=np.float64)

        if len(self._angle_indices) > 0:
            tvecs['angles'] = self._angle_ncvecs @ cell
        else:
            tvecs['angles'] = np.empty((0, 2, 3), dtype=np.float64)

        if len(self._dihedral_indices) > 0:
            tvecs['dihedrals'] = self._dihedral_ncvecs @ cell
        else:
            tvecs['dihedrals'] = np.empty((0, 3, 3), dtype=np.float64)

        # Padded tvecs (for GPU-efficient batch computation)
        if len(self._bond_indices_padded) > 0:
            tvecs['bonds_padded'] = self._bond_ncvecs_padded @ cell
        else:
            tvecs['bonds_padded'] = np.empty((0, 1, 3), dtype=np.float64)

        if len(self._angle_indices_padded) > 0:
            tvecs['angles_padded'] = self._angle_ncvecs_padded @ cell
        else:
            tvecs['angles_padded'] = np.empty((0, 2, 3), dtype=np.float64)

        if len(self._dihedral_indices_padded) > 0:
            tvecs['dihedrals_padded'] = self._dihedral_ncvecs_padded @ cell
        else:
            tvecs['dihedrals_padded'] = np.empty((0, 3, 3), dtype=np.float64)

        self._tvecs_cache = {'cell_hash': cell_hash, 'tvecs': tvecs}
        return tvecs

    def _invalidate_batched_arrays(self) -> None:
        """Invalidate batched arrays (call when internals change)."""
        self._batched_arrays_valid = False

    def _compute_batched_values(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all internal coordinate values using vectorized operations.

        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        result = {}

        # Bonds - use padded arrays for consistent JAX shapes
        if self._n_bonds_actual > 0:
            bond_pos = positions[self._bond_indices_padded]  # (n_padded, 2, 3)
            values_padded = np.asarray(device_get(_bond_value_batched(bond_pos, tvecs['bonds_padded'])))
            result['bonds'] = values_padded[:self._n_bonds_actual]
        else:
            result['bonds'] = np.empty(0)

        # Angles
        if self._n_angles_actual > 0:
            angle_pos = positions[self._angle_indices_padded]  # (n_padded, 3, 3)
            values_padded = np.asarray(device_get(_angle_value_batched(angle_pos, tvecs['angles_padded'])))
            result['angles'] = values_padded[:self._n_angles_actual]
        else:
            result['angles'] = np.empty(0)

        # Dihedrals
        if self._n_dihedrals_actual > 0:
            dihedral_pos = positions[self._dihedral_indices_padded]  # (n_padded, 4, 3)
            values_padded = np.asarray(device_get(_dihedral_value_batched(dihedral_pos, tvecs['dihedrals_padded'])))
            result['dihedrals'] = values_padded[:self._n_dihedrals_actual]
        else:
            result['dihedrals'] = np.empty(0)

        return result

    def _compute_batched_gradients(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute all internal coordinate gradients using vectorized operations.

        Returns dict mapping coord type to (indices, gradients) tuples.
        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        result = {}

        # Bonds - use padded arrays
        if self._n_bonds_actual > 0:
            bond_pos = positions[self._bond_indices_padded]  # (n_padded, 2, 3)
            grads_padded = np.asarray(device_get(_bond_grad_batched(bond_pos, tvecs['bonds_padded'])))
            result['bonds'] = (self._bond_indices, grads_padded[:self._n_bonds_actual])
        else:
            result['bonds'] = (np.empty((0, 2), dtype=np.int32), np.empty((0, 2, 3)))

        # Angles
        if self._n_angles_actual > 0:
            angle_pos = positions[self._angle_indices_padded]
            grads_padded = np.asarray(device_get(_angle_grad_batched(angle_pos, tvecs['angles_padded'])))
            result['angles'] = (self._angle_indices, grads_padded[:self._n_angles_actual])
        else:
            result['angles'] = (np.empty((0, 3), dtype=np.int32), np.empty((0, 3, 3)))

        # Dihedrals
        if self._n_dihedrals_actual > 0:
            dihedral_pos = positions[self._dihedral_indices_padded]
            grads_padded = np.asarray(device_get(_dihedral_grad_batched(dihedral_pos, tvecs['dihedrals_padded'])))
            result['dihedrals'] = (self._dihedral_indices, grads_padded[:self._n_dihedrals_actual])
        else:
            result['dihedrals'] = (np.empty((0, 4), dtype=np.int32), np.empty((0, 4, 3)))

        return result

    def _compute_batched_hessians(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute all internal coordinate hessians using vectorized operations.

        Returns dict mapping coord type to (indices, hessians) tuples.
        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        result = {}

        # Bonds - use padded arrays
        if self._n_bonds_actual > 0:
            bond_pos = positions[self._bond_indices_padded]
            hess_padded = np.asarray(device_get(_bond_hess_batched(bond_pos, tvecs['bonds_padded'])))
            result['bonds'] = (self._bond_indices, hess_padded[:self._n_bonds_actual])
        else:
            result['bonds'] = (np.empty((0, 2), dtype=np.int32), np.empty((0, 2, 3, 2, 3)))

        # Angles
        if self._n_angles_actual > 0:
            angle_pos = positions[self._angle_indices_padded]
            hess_padded = np.asarray(device_get(_angle_hess_batched(angle_pos, tvecs['angles_padded'])))
            result['angles'] = (self._angle_indices, hess_padded[:self._n_angles_actual])
        else:
            result['angles'] = (np.empty((0, 3), dtype=np.int32), np.empty((0, 3, 3, 3, 3)))

        # Dihedrals
        if self._n_dihedrals_actual > 0:
            dihedral_pos = positions[self._dihedral_indices_padded]
            hess_padded = np.asarray(device_get(_dihedral_hess_batched(dihedral_pos, tvecs['dihedrals_padded'])))
            result['dihedrals'] = (self._dihedral_indices, hess_padded[:self._n_dihedrals_actual])
        else:
            result['dihedrals'] = (np.empty((0, 4), dtype=np.int32), np.empty((0, 4, 3, 4, 3)))

        return result

    def _compute_batched_cell_gradients(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all internal coordinate cell gradients using vectorized operations.

        Returns dict mapping coord type to cell gradient arrays.
        Each gradient has shape (n_coords, 3, 3) for d(coord)/d(cell).
        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        cell_jax = jnp.asarray(cell, dtype=np.float64)
        result = {}

        # Bonds - use padded arrays for consistent JAX shapes
        if self._n_bonds_actual > 0:
            bond_pos = jnp.asarray(positions[self._bond_indices_padded], dtype=np.float64)
            bond_ncvecs = jnp.asarray(self._bond_ncvecs_padded, dtype=np.float64)
            grads_padded = np.asarray(device_get(_bond_cell_grad_batched(bond_pos, bond_ncvecs, cell_jax)))
            result['bonds'] = grads_padded[:self._n_bonds_actual]
        else:
            result['bonds'] = np.empty((0, 3, 3))

        # Angles
        if self._n_angles_actual > 0:
            angle_pos = jnp.asarray(positions[self._angle_indices_padded], dtype=np.float64)
            angle_ncvecs = jnp.asarray(self._angle_ncvecs_padded, dtype=np.float64)
            grads_padded = np.asarray(device_get(_angle_cell_grad_batched(angle_pos, angle_ncvecs, cell_jax)))
            result['angles'] = grads_padded[:self._n_angles_actual]
        else:
            result['angles'] = np.empty((0, 3, 3))

        # Dihedrals
        if self._n_dihedrals_actual > 0:
            dihedral_pos = jnp.asarray(positions[self._dihedral_indices_padded], dtype=np.float64)
            dihedral_ncvecs = jnp.asarray(self._dihedral_ncvecs_padded, dtype=np.float64)
            grads_padded = np.asarray(device_get(_dihedral_cell_grad_batched(dihedral_pos, dihedral_ncvecs, cell_jax)))
            result['dihedrals'] = grads_padded[:self._n_dihedrals_actual]
        else:
            result['dihedrals'] = np.empty((0, 3, 3))

        return result

    def copy(self) -> 'BaseInternals':
        raise NotImplementedError

    def calc(self) -> np.ndarray:
        """Calculates the internal coordinate vector using vectorized operations."""
        self._cache_check()
        if 'coords' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_vals = self._compute_batched_values(positions, cell)

            # Build full coords list in order
            all_coords = []

            # Translations (not batched - usually few) - use lightweight atoms
            atoms = self.light_atoms
            for coord in self.internals['translations']:
                all_coords.append(coord.calc(atoms))

            # Bonds (batched)
            all_coords.extend(batched_vals['bonds'].tolist())

            # Angles (batched)
            all_coords.extend(batched_vals['angles'].tolist())

            # Dihedrals (batched)
            all_coords.extend(batched_vals['dihedrals'].tolist())

            # Other (not batched - heterogeneous)
            for coord in self.internals['other']:
                all_coords.append(coord.calc(atoms))

            # Rotations (batched if all 3 axes present per fragment)
            rot_vals = self._batched_rotation_values(positions)
            if rot_vals is None:
                for coord in self.internals['rotations']:
                    all_coords.append(coord.calc(atoms))
            else:
                all_coords.extend(rot_vals)

            self._cache['coords'] = np.array(all_coords)

        return np.array([
            x for x, a in zip(self._cache['coords'], self._active_mask) if a
        ])

    def jacobian(self) -> np.ndarray:
        """Calculates the internal coordinate Jacobian matrix using vectorized operations."""
        self._cache_check()

        # If a fully-built B was cached, return it directly. The cache is
        # invalidated by _cache_check whenever positions change, and the active
        # mask is stable within a single position evaluation.
        cached_B = self._cache.get('jacobian_B')
        if cached_B is not None:
            return cached_B

        if 'jacobian' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_grads = self._compute_batched_gradients(positions, cell)

            # Non-batched coords use lightweight atoms
            atoms = self.light_atoms
            trans_data = [(coord.indices, np.array(coord.calc_gradient(atoms)))
                          for coord in self.internals['translations']]
            other_data = [(coord.indices, np.array(coord.calc_gradient(atoms)))
                          for coord in self.internals['other']]
            rot_data = self._batched_rotation_gradients(positions)
            if rot_data is None:
                rot_data = [(coord.indices, np.array(coord.calc_gradient(atoms)))
                            for coord in self.internals['rotations']]

            self._cache['jacobian_batched'] = batched_grads
            self._cache['jacobian_nonbatched'] = (trans_data, other_data, rot_data)
            # Store a unique object (not a singleton) for cache identity
            self._cache['jacobian'] = object()

        # Get cached data
        batched = self._cache['jacobian_batched']
        trans_data, other_data, rot_data = self._cache['jacobian_nonbatched']

        # Get counts for each type
        n_trans = len(trans_data)
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(other_data)
        n_rot = len(rot_data)

        # Build active masks per type
        active_mask = self._active_mask
        start = 0
        trans_active = active_mask[start:start+n_trans]
        start += n_trans
        bonds_active = active_mask[start:start+n_bonds]
        start += n_bonds
        angles_active = active_mask[start:start+n_angles]
        start += n_angles
        dihedrals_active = active_mask[start:start+n_dihedrals]
        start += n_dihedrals
        other_active = active_mask[start:start+n_other]
        start += n_other
        rot_active = active_mask[start:start+n_rot]

        n_active = sum(active_mask)
        n_atoms = self.natoms + self.ndummies
        B = np.zeros((n_active, n_atoms, 3))
        row = 0

        # Translations (not batched)
        for i, (idx, jac) in enumerate(trans_data):
            if trans_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        # Bonds (batched) - vectorized scatter
        bond_indices, bond_grads = batched['bonds']
        bonds_active_arr = np.array(bonds_active, dtype=bool)
        n_active_bonds = bonds_active_arr.sum()
        if n_active_bonds > 0:
            active_bond_idx = bond_indices[bonds_active_arr]
            active_bond_grads = bond_grads[bonds_active_arr]
            # Vectorized scatter: replace loop with advanced indexing
            rows_idx = np.arange(row, row + n_active_bonds)[:, None]
            B[rows_idx, active_bond_idx] = active_bond_grads
            row += n_active_bonds

        # Angles (batched) - vectorized scatter
        angle_indices, angle_grads = batched['angles']
        angles_active_arr = np.array(angles_active, dtype=bool)
        n_active_angles = angles_active_arr.sum()
        if n_active_angles > 0:
            active_angle_idx = angle_indices[angles_active_arr]
            active_angle_grads = angle_grads[angles_active_arr]
            # Vectorized scatter
            rows_idx = np.arange(row, row + n_active_angles)[:, None]
            B[rows_idx, active_angle_idx] = active_angle_grads
            row += n_active_angles

        # Dihedrals (batched) - vectorized scatter
        dihedral_indices, dihedral_grads = batched['dihedrals']
        dihedrals_active_arr = np.array(dihedrals_active, dtype=bool)
        n_active_dihedrals = dihedrals_active_arr.sum()
        if n_active_dihedrals > 0:
            active_dih_idx = dihedral_indices[dihedrals_active_arr]
            active_dih_grads = dihedral_grads[dihedrals_active_arr]
            # Vectorized scatter
            rows_idx = np.arange(row, row + n_active_dihedrals)[:, None]
            B[rows_idx, active_dih_idx] = active_dih_grads
            row += n_active_dihedrals

        # Other (not batched)
        for i, (idx, jac) in enumerate(other_data):
            if other_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        # Rotations (not batched)
        for i, (idx, jac) in enumerate(rot_data):
            if rot_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        result = B.reshape((n_active, 3 * n_atoms))
        self._cache['jacobian_B'] = result
        return result

    def cell_jacobian(self) -> np.ndarray:
        """Compute Jacobian of internal coordinates with respect to cell matrix.

        Returns:
            np.ndarray: Shape (n_active_coords, 9) matrix where each row is
                        the flattened d(coord)/d(cell) gradient.

        Note:
            - Translations, rotations, and other non-periodic coordinates have
              zero cell derivatives (they don't depend on the cell).
            - Only bonds, angles, and dihedrals with non-zero ncvecs have
              non-zero cell derivatives.

        Raises:
            ValueError: If the system does not have periodic boundary conditions.
        """
        if not np.any(self.atoms.pbc):
            raise ValueError(
                "cell_jacobian() requires periodic boundary conditions. "
                "Set atoms.pbc = True for periodic systems."
            )

        self._cache_check()

        if 'cell_jacobian' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Compute batched cell gradients for bonds, angles, dihedrals
            cell_grads = self._compute_batched_cell_gradients(positions, cell)
            self._cache['cell_jacobian_batched'] = cell_grads
            self._cache['cell_jacobian'] = object()

        cell_grads = self._cache['cell_jacobian_batched']

        # Get counts for each type
        n_trans = len(self.internals['translations'])
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])

        # Build active masks per type
        active_mask = self._active_mask
        start = 0
        trans_active = active_mask[start:start+n_trans]
        start += n_trans
        bonds_active = active_mask[start:start+n_bonds]
        start += n_bonds
        angles_active = active_mask[start:start+n_angles]
        start += n_angles
        dihedrals_active = active_mask[start:start+n_dihedrals]
        start += n_dihedrals
        other_active = active_mask[start:start+n_other]
        start += n_other
        rot_active = active_mask[start:start+n_rot]

        n_active = sum(active_mask)
        B_cell = np.zeros((n_active, 3, 3))
        row = 0

        # Translations have zero cell derivatives (they're CoM positions)
        row += sum(trans_active)

        # Bonds
        bonds_active_arr = np.array(bonds_active, dtype=bool)
        n_active_bonds = bonds_active_arr.sum()
        if n_active_bonds > 0:
            B_cell[row:row+n_active_bonds] = cell_grads['bonds'][bonds_active_arr]
            row += n_active_bonds

        # Angles
        angles_active_arr = np.array(angles_active, dtype=bool)
        n_active_angles = angles_active_arr.sum()
        if n_active_angles > 0:
            B_cell[row:row+n_active_angles] = cell_grads['angles'][angles_active_arr]
            row += n_active_angles

        # Dihedrals
        dihedrals_active_arr = np.array(dihedrals_active, dtype=bool)
        n_active_dihedrals = dihedrals_active_arr.sum()
        if n_active_dihedrals > 0:
            B_cell[row:row+n_active_dihedrals] = cell_grads['dihedrals'][dihedrals_active_arr]
            row += n_active_dihedrals

        # Other has zero cell derivatives (custom coordinates, not periodic)
        row += sum(other_active)

        # Rotations have zero cell derivatives
        row += sum(rot_active)

        # Flatten cell matrix to 9-element vector (row-major order)
        return B_cell.reshape((n_active, 9))

    def _rotation_padded_inputs(self, positions: np.ndarray):
        """Build padded (pos, refpos, mask) batches grouped by fragment.

        Returns ``(pos_pad, ref_pad, mask, frag_indices, frag_axis_slots,
        valid)`` where:
          pos_pad/ref_pad shape (n_frags, N_max, 3),
          mask shape (n_frags, N_max),
          frag_indices: list of np.array per fragment,
          frag_axis_slots: list of [axis0_idx, axis1_idx, axis2_idx]
            per fragment (each entry is the original Rotation index),
          valid: True if all fragments have all 3 axes.
        Cached per geometry on ``self._cache``.
        """
        cached = self._cache.get('rotation_pad')
        if cached is not None:
            return cached
        rotations = self.internals['rotations']
        if not rotations:
            out = (None, None, None, [], [], True)
            self._cache['rotation_pad'] = out
            return out
        groups = {}
        for i, r in enumerate(rotations):
            key = (tuple(r.indices), r.kwargs['refpos'].tobytes())
            slot = groups.setdefault(key, [None, None, None])
            slot[r.kwargs['axis']] = i
        if any(None in slot for slot in groups.values()):
            out = (None, None, None, [], [], False)
            self._cache['rotation_pad'] = out
            return out
        n_frags = len(groups)
        n_max = max(len(r.indices) for r in rotations)
        pos_pad = np.zeros((n_frags, n_max, 3), dtype=np.float64)
        ref_pad = np.zeros((n_frags, n_max, 3), dtype=np.float64)
        mask = np.zeros((n_frags, n_max), dtype=np.float64)
        frag_indices = []
        frag_axis_slots = []
        for fi, slot in enumerate(groups.values()):
            r0 = rotations[slot[0]]
            n = len(r0.indices)
            pos_pad[fi, :n] = positions[r0.indices]
            ref_pad[fi, :n] = r0.kwargs['refpos']
            mask[fi, :n] = 1.0
            frag_indices.append(np.asarray(r0.indices))
            frag_axis_slots.append(slot)
        out = (pos_pad, ref_pad, mask, frag_indices, frag_axis_slots, True)
        self._cache['rotation_pad'] = out
        return out

    def _batched_rotation_values(self, positions: np.ndarray):
        """Per-Rotation values via one padded+vmapped JAX dispatch.

        Returns a length-N_rotations list of floats in original order,
        or None when the heterogeneous fall-back is required.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        pos_pad, ref_pad, mask, _, slots, valid = self._rotation_padded_inputs(positions)
        if not valid:
            return None
        vals_padded = np.asarray(device_get(_rotation_3axis_value_batched(
            jnp.asarray(pos_pad), jnp.asarray(ref_pad), jnp.asarray(mask)
        )))
        # vals_padded.shape == (n_frags, 3)
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            for axis, rot_idx in enumerate(slot):
                out[rot_idx] = float(vals_padded[fi, axis])
        return out

    def _batched_rotation_gradients(self, positions: np.ndarray):
        """Per-Rotation gradients via one padded+vmapped JAX dispatch.

        Returns a list of ``(indices, grad)`` tuples in original order,
        or None when the heterogeneous fall-back is required.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        pos_pad, ref_pad, mask, frag_indices, slots, valid = (
            self._rotation_padded_inputs(positions)
        )
        if not valid:
            return None
        grads_padded = np.asarray(device_get(_rotation_3axis_grad_batched(
            jnp.asarray(pos_pad), jnp.asarray(ref_pad), jnp.asarray(mask)
        )))
        # grads_padded.shape == (n_frags, 3, N_max, 3)
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            n = len(frag_indices[fi])
            for axis, rot_idx in enumerate(slot):
                out[rot_idx] = (frag_indices[fi], grads_padded[fi, axis, :n, :])
        return out

    def _batched_rotation_hessians(self, positions: np.ndarray):
        """Compute per-Rotation Hessians via one padded+vmapped JAX dispatch.

        Each fragment owns 3 Rotation objects (axes 0,1,2) that share
        the same atom indices and refpos. We group them by fragment,
        pad to the max-fragment-size across the system, run one
        vmapped call to ``_rotation_hess_padded_vmap``, and slice the
        result back into per-Rotation arrays.

        On AJEYAQ (4 fragments x 3 axes = 12 Rotations) this drops the
        rotation portion of int.hessian build from ~14ms to ~1ms.
        Returns a list of ``(indices, hess)`` tuples in the original
        per-Rotation order, matching the shape Internals.hessian
        expects.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        pos_pad, ref_pad, mask, frag_indices, slots, valid = (
            self._rotation_padded_inputs(positions)
        )
        if not valid:
            atoms = self.light_atoms
            return [(r.indices, np.array(r.calc_hessian(atoms)))
                    for r in rotations]
        hess_padded = np.array(device_get(_rotation_hess_padded_vmap(
            jnp.asarray(pos_pad), jnp.asarray(ref_pad), jnp.asarray(mask)
        )))
        # For linear molecules, the quaternion eigenvalue problem has
        # degenerate eigenvalues producing NaN second derivatives. The
        # corresponding rotation is physically meaningless (zero Jacobian),
        # so replacing NaN with 0 is exact.
        if np.any(np.isnan(hess_padded)):
            np.nan_to_num(hess_padded, copy=False)
        # hess_padded.shape == (n_frags, 3, N_max, 3, N_max, 3)
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            n = len(frag_indices[fi])
            for axis, rot_idx in enumerate(slot):
                h = hess_padded[fi, axis, :n, :, :n, :]
                out[rot_idx] = (frag_indices[fi], h)
        return out

    def _get_hessian_skeleton(self, hessians):
        """Return a cached SparseInternalHessiansSkeleton for ``hessians``.

        The skeleton holds index-derived data (per-size groupings, scatter
        indices) that depend only on which coordinates exist and which
        atom indices they touch — not on positions or Hessian values. We
        invalidate by total coord count + active mask, which jointly
        cover the mutation paths: ``add_dummy_to_internals`` /
        ``find_all_*`` / ``check_for_bad_internals`` regenerations grow
        ``self.internals``, while ``apply_inequalities`` /
        ``validate_inequalities`` flip ``self._active``.
        """
        key = (len(hessians), self.natoms + self.ndummies,
               tuple(self._active_mask))
        cached = self._hessian_skeleton
        if cached is not None and cached[0] == key:
            return cached[1]
        skeleton = SparseInternalHessiansSkeleton(hessians,
                                                  self.natoms + self.ndummies)
        self._hessian_skeleton = (key, skeleton)
        return skeleton

    def hessian(self) -> np.ndarray:
        """Calculates the Hessian matrix for each internal coordinate using vectorized operations."""
        self._cache_check()

        # Return cached SparseInternalHessians object if available
        if 'hessian_result' in self._cache:
            return self._cache['hessian_result']

        if 'hessian' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_hess = self._compute_batched_hessians(positions, cell)

            # Non-batched coords use lightweight atoms. Translation hessians are
            # identically zero (translations are linear in positions), so cache
            # one zero array per (n,) and reuse — avoids 24+ JAX calls per
            # hessian rebuild on systems with TRICs.
            atoms = self.light_atoms
            trans_data = []
            zero_cache = {}
            for coord in self.internals['translations']:
                n = len(coord.indices)
                z = zero_cache.get(n)
                if z is None:
                    z = np.zeros((n, 3, n, 3))
                    zero_cache[n] = z
                trans_data.append((coord.indices, z))
            other_data = [(coord.indices, np.array(coord.calc_hessian(atoms)))
                          for coord in self.internals['other']]
            rot_data = self._batched_rotation_hessians(positions)

            self._cache['hessian_batched'] = batched_hess
            self._cache['hessian_nonbatched'] = (trans_data, other_data, rot_data)
            # Store a unique object (not a singleton) for cache identity
            self._cache['hessian'] = object()

        # Get cached data
        batched = self._cache['hessian_batched']
        trans_data, other_data, rot_data = self._cache['hessian_nonbatched']

        # Get counts for each type
        n_trans = len(trans_data)
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(other_data)
        n_rot = len(rot_data)

        # Build active masks per type
        active_mask = self._active_mask
        start = 0
        trans_active = active_mask[start:start+n_trans]
        start += n_trans
        bonds_active = active_mask[start:start+n_bonds]
        start += n_bonds
        angles_active = active_mask[start:start+n_angles]
        start += n_angles
        dihedrals_active = active_mask[start:start+n_dihedrals]
        start += n_dihedrals
        other_active = active_mask[start:start+n_other]
        start += n_other
        rot_active = active_mask[start:start+n_rot]

        n_atoms = self.natoms + self.ndummies
        hessians = []

        # Translations (not batched). Hessian rows are stored in cached
        # nonbatched data; SparseInternalHessian only reads .vals so views are
        # safe.
        for i, (idx, hess) in enumerate(trans_data):
            if trans_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        # Bonds (batched). Fancy indexing already returns a fresh array; per-row
        # views into it are read-only consumers, so no per-coord copy is needed.
        bond_indices, bond_hess = batched['bonds']
        bonds_active_arr = np.asarray(bonds_active, dtype=bool)
        if bonds_active_arr.any():
            active_bond_idx = bond_indices[bonds_active_arr]
            active_bond_hess = bond_hess[bonds_active_arr]
            for i in range(len(active_bond_idx)):
                hessians.append(SparseInternalHessian(n_atoms, active_bond_idx[i], active_bond_hess[i]))

        # Angles (batched)
        angle_indices, angle_hess = batched['angles']
        angles_active_arr = np.asarray(angles_active, dtype=bool)
        if angles_active_arr.any():
            active_angle_idx = angle_indices[angles_active_arr]
            active_angle_hess = angle_hess[angles_active_arr]
            for i in range(len(active_angle_idx)):
                hessians.append(SparseInternalHessian(n_atoms, active_angle_idx[i], active_angle_hess[i]))

        # Dihedrals (batched)
        dihedral_indices, dihedral_hess = batched['dihedrals']
        dihedrals_active_arr = np.asarray(dihedrals_active, dtype=bool)
        if dihedrals_active_arr.any():
            active_dih_idx = dihedral_indices[dihedrals_active_arr]
            active_dih_hess = dihedral_hess[dihedrals_active_arr]
            for i in range(len(active_dih_idx)):
                hessians.append(SparseInternalHessian(n_atoms, active_dih_idx[i], active_dih_hess[i]))

        # Other (not batched)
        for i, (idx, hess) in enumerate(other_data):
            if other_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        # Rotations (not batched)
        for i, (idx, hess) in enumerate(rot_data):
            if rot_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        result = SparseInternalHessians(hessians, self.ndof,
                                        skeleton=self._get_hessian_skeleton(hessians))
        self._cache['hessian_result'] = result
        return result

    def hessian_rdot(self, v: np.ndarray):
        """Compute Hessian @ v for all internal coordinates using direct HVP.

        This computes the same result as hessian().rdot(v) but uses forward-over-reverse
        mode autodiff (jvp(grad(f))) to compute Hessian-vector products directly,
        avoiding the O(n²) cost of materializing full Hessian matrices.

        Args:
            v: Vector of shape (ndof,) to multiply with each coordinate's Hessian

        Returns:
            Sparse CSR matrix of shape (n_active_coords, ndof) where each row
            is H_i @ v. Returns dense ndarray as fallback when not all
            coordinates are active.
        """
        self._cache_check()
        positions = self.all_positions
        cell = self.atoms.cell.array
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)

        # Reshape v for easy indexing
        v_atoms = v.reshape((-1, 3))  # (n_atoms, 3)
        n_atoms = self.natoms + self.ndummies
        ndof = self.ndof  # Cache to avoid repeated property lookups

        # Get active mask and counts
        active_mask = self._active_mask
        n_trans = len(self.internals['translations'])
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])

        start = 0
        trans_active = active_mask[start:start+n_trans]
        start += n_trans
        bonds_active = np.array(active_mask[start:start+n_bonds], dtype=bool)
        start += n_bonds
        angles_active = np.array(active_mask[start:start+n_angles], dtype=bool)
        start += n_angles
        dihedrals_active = np.array(active_mask[start:start+n_dihedrals], dtype=bool)
        start += n_dihedrals
        other_active = active_mask[start:start+n_other]
        start += n_other
        rot_active = active_mask[start:start+n_rot]

        n_active = sum(active_mask)

        # Fast path: when all coords are active, use pre-built CSR structure
        use_sparse = (n_active == self._csr_n_active)

        if use_sparse:
            data = self._csr_data
            data[:] = 0
        else:
            if (self._hvp_buf is None
                    or self._hvp_buf.shape != (n_active, ndof)):
                self._hvp_buf = np.zeros((n_active, ndof))
            out = self._hvp_buf
            out[:] = 0

        row = 0  # Current write position in output

        # Translations - Hessian is zero
        n_active_trans = sum(trans_active)
        # out[row:row+n_active_trans] is already zero from the clear
        row += n_active_trans

        # Launch all JAX HVP computations, deferring device_get
        # This allows JAX to pipeline the computations before we block on transfer

        bond_jax_result = None
        bond_active_idx = None
        if bonds_active.any() and self._n_bonds_actual > 0:
            if bonds_active.all():
                bond_pos = positions[self._bond_indices_padded]
                bond_tvecs = tvecs['bonds_padded']
                v_sub = v_atoms[self._bond_indices_padded]
                bond_jax_result = _bond_hvp_batched(bond_pos, bond_tvecs, v_sub)
                bond_active_idx = self._bond_indices
            else:
                bond_active_idx = self._bond_indices[bonds_active]
                bond_pos = positions[bond_active_idx]
                bond_tvecs = tvecs['bonds'][bonds_active]
                v_sub = v_atoms[bond_active_idx]
                bond_jax_result = _bond_hvp_batched(bond_pos, bond_tvecs, v_sub)

        angle_jax_result = None
        angle_active_idx = None
        if angles_active.any() and self._n_angles_actual > 0:
            if angles_active.all():
                angle_pos = positions[self._angle_indices_padded]
                angle_tvecs = tvecs['angles_padded']
                v_sub = v_atoms[self._angle_indices_padded]
                angle_jax_result = _angle_hvp_batched(angle_pos, angle_tvecs, v_sub)
                angle_active_idx = self._angle_indices
            else:
                angle_active_idx = self._angle_indices[angles_active]
                angle_pos = positions[angle_active_idx]
                angle_tvecs = tvecs['angles'][angles_active]
                v_sub = v_atoms[angle_active_idx]
                angle_jax_result = _angle_hvp_batched(angle_pos, angle_tvecs, v_sub)

        dih_jax_result = None
        dih_active_idx = None
        if dihedrals_active.any() and self._n_dihedrals_actual > 0:
            if dihedrals_active.all():
                dih_pos = positions[self._dihedral_indices_padded]
                dih_tvecs = tvecs['dihedrals_padded']
                v_sub = v_atoms[self._dihedral_indices_padded]
                dih_jax_result = _dihedral_hvp_batched(dih_pos, dih_tvecs, v_sub)
                dih_active_idx = self._dihedral_indices
            else:
                dih_active_idx = self._dihedral_indices[dihedrals_active]
                dih_pos = positions[dih_active_idx]
                dih_tvecs = tvecs['dihedrals'][dihedrals_active]
                v_sub = v_atoms[dih_active_idx]
                dih_jax_result = _dihedral_hvp_batched(dih_pos, dih_tvecs, v_sub)

        # Launch rotation HVPs: one vmapped dispatch per fragment x 3 axes
        # when all rotations are active and all fragments have all 3 axes.
        # Otherwise fall back to per-rotation calls.
        rot_jax_results = []
        rot_indices = []
        rot_batched_result = None
        rot_batched_slots = None
        rot_batched_frag_indices = None
        all_rot_active = bool(np.asarray(rot_active, dtype=bool).all())
        if all_rot_active and self.internals['rotations']:
            pos_pad, ref_pad, mask, frag_indices, slots, valid = (
                self._rotation_padded_inputs(positions)
            )
        else:
            valid = False
        if valid:
            n_max = mask.shape[1]
            v_pad = np.zeros((len(frag_indices), n_max, 3), dtype=np.float64)
            for fi, fi_idx in enumerate(frag_indices):
                v_pad[fi, :len(fi_idx)] = v_atoms[fi_idx]
            rot_batched_result = _rotation_3axis_hvp_batched_jit(
                jnp.asarray(pos_pad), jnp.asarray(ref_pad),
                jnp.asarray(mask), jnp.asarray(v_pad),
            )
            rot_batched_slots = slots
            rot_batched_frag_indices = frag_indices
        else:
            for i, coord in enumerate(self.internals['rotations']):
                if rot_active[i]:
                    idx = np.array(coord.indices)
                    pos = positions[idx]
                    v_sub = v_atoms[idx]
                    axis = coord.kwargs['axis']
                    refpos = coord.kwargs['refpos']
                    rot_jax_results.append(
                        (_rotation_hvp_jit(pos, axis, refpos, v_sub), idx)
                    )

        # Now collect results with device_get and scatter into output

        if bond_jax_result is not None:
            hvp = np.asarray(device_get(bond_jax_result))
            if bonds_active.all():
                hvp = hvp[:self._n_bonds_actual]
            n_coords = self._n_bonds_actual if bonds_active.all() else int(bonds_active.sum())
            if use_sparse:
                off = self._csr_bond_offset
                data[off:off + n_coords * 6] = hvp.reshape(-1)
            else:
                flat_cols = self._bond_flat_cols if bonds_active.all() else self._bond_flat_cols[bonds_active]
                out[row:row+n_coords, :] = 0
                out[np.arange(row, row+n_coords)[:, None], flat_cols] = hvp.reshape(n_coords, -1)
            row += n_coords

        if angle_jax_result is not None:
            hvp = np.asarray(device_get(angle_jax_result))
            if angles_active.all():
                hvp = hvp[:self._n_angles_actual]
            n_coords = self._n_angles_actual if angles_active.all() else int(angles_active.sum())
            if use_sparse:
                off = self._csr_angle_offset
                data[off:off + n_coords * 9] = hvp.reshape(-1)
            else:
                flat_cols = self._angle_flat_cols if angles_active.all() else self._angle_flat_cols[angles_active]
                out[row:row+n_coords, :] = 0
                out[np.arange(row, row+n_coords)[:, None], flat_cols] = hvp.reshape(n_coords, -1)
            row += n_coords

        if dih_jax_result is not None:
            hvp = np.asarray(device_get(dih_jax_result))
            if dihedrals_active.all():
                hvp = hvp[:self._n_dihedrals_actual]
            n_coords = self._n_dihedrals_actual if dihedrals_active.all() else int(dihedrals_active.sum())
            if use_sparse:
                off = self._csr_dih_offset
                data[off:off + n_coords * 12] = hvp.reshape(-1)
            else:
                flat_cols = self._dihedral_flat_cols if dihedrals_active.all() else self._dihedral_flat_cols[dihedrals_active]
                out[row:row+n_coords, :] = 0
                out[np.arange(row, row+n_coords)[:, None], flat_cols] = hvp.reshape(n_coords, -1)
            row += n_coords

        # Other - use existing hessian computation (typically few coords, loop is fine)
        atoms = self.light_atoms
        off = self._csr_other_offset if use_sparse else 0
        for i, coord in enumerate(self.internals['other']):
            if other_active[i]:
                hess = np.array(coord.calc_hessian(atoms))
                idx = np.array(coord.indices)
                v_sub = v_atoms[idx]
                hvp = np.einsum('aibj,bj->ai', hess, v_sub)
                if use_sparse:
                    dense_row = np.zeros(ndof)
                    dense_row.reshape((-1, 3))[idx] = hvp
                    data[off:off + ndof] = dense_row
                    off += ndof
                else:
                    out_row = out[row].reshape((-1, 3))
                    out_row[idx] = hvp
                row += 1

        # Rotations - collect results (batched or per-rotation)
        if rot_batched_result is not None:
            hvp_padded = np.asarray(device_get(rot_batched_result))
            # hvp_padded.shape == (n_frags, 3, N_max, 3)
            # Reassemble in original Rotation order
            ordered = [None] * len(self.internals['rotations'])
            for fi, slot in enumerate(rot_batched_slots):
                n = len(rot_batched_frag_indices[fi])
                for axis, rot_idx in enumerate(slot):
                    ordered[rot_idx] = (
                        hvp_padded[fi, axis, :n, :],
                        rot_batched_frag_indices[fi],
                    )
            for i, coord in enumerate(self.internals['rotations']):
                if not rot_active[i]:
                    continue
                hvp, idx = ordered[i]
                if use_sparse:
                    dense_row = np.zeros(ndof)
                    dense_row.reshape((-1, 3))[idx] = hvp
                    data[off:off + ndof] = dense_row
                    off += ndof
                else:
                    out_row = out[row].reshape((-1, 3))
                    out_row[idx] = hvp
                row += 1
        else:
            for jax_result, idx in rot_jax_results:
                hvp = np.asarray(device_get(jax_result))
                if use_sparse:
                    dense_row = np.zeros(ndof)
                    dense_row.reshape((-1, 3))[idx] = hvp
                    data[off:off + ndof] = dense_row
                    off += ndof
                else:
                    out_row = out[row].reshape((-1, 3))
                    out_row[idx] = hvp
                row += 1

        if use_sparse:
            return sparse.csr_matrix(
                (data, self._csr_indices, self._csr_indptr),
                shape=(self._csr_n_active, ndof), copy=False,
            )
        return out[:row]

    def hessian_rdot_vecs(self, v: np.ndarray, *us: np.ndarray):
        """Compute (H_i @ v) · u for all active internal coordinates.

        Equivalent to ``[hessian_rdot(v) @ u for u in us]`` but avoids
        materializing the dense (n_coords, ndof) matrix.  Each element *i*
        of a result vector is the dot product of coordinate *i*'s
        Hessian-vector product row with *u*.

        Args:
            v: Vector of shape (ndof,) — the HVP direction.
            *us: One or more vectors of shape (ndof,) to contract with.

        Returns:
            Tuple of arrays, each of shape (n_active_coords,).
        """
        self._cache_check()
        positions = self.all_positions
        cell = self.atoms.cell.array
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)

        v_atoms = v.reshape((-1, 3))
        u_atoms_list = [u.reshape((-1, 3)) for u in us]
        n_us = len(us)

        # Active mask bookkeeping (same as hessian_rdot)
        active_mask = self._active_mask
        n_trans = len(self.internals['translations'])
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])

        start = 0
        trans_active = active_mask[start:start + n_trans]
        start += n_trans
        bonds_active = np.array(active_mask[start:start + n_bonds], dtype=bool)
        start += n_bonds
        angles_active = np.array(active_mask[start:start + n_angles], dtype=bool)
        start += n_angles
        dihedrals_active = np.array(
            active_mask[start:start + n_dihedrals], dtype=bool
        )
        start += n_dihedrals
        other_active = active_mask[start:start + n_other]
        start += n_other
        rot_active = active_mask[start:start + n_rot]

        # Collect per-coordinate-type result fragments for each u
        results = [[] for _ in range(n_us)]

        # --- Translations: Hessian is zero → dot product is zero ---
        n_active_trans = sum(trans_active)
        if n_active_trans > 0:
            z = np.zeros(n_active_trans)
            for r in results:
                r.append(z)

        # --- Bonds ---
        if bonds_active.any() and self._n_bonds_actual > 0:
            if bonds_active.all():
                bond_pos = positions[self._bond_indices_padded]
                bond_tvecs = tvecs['bonds_padded']
                v_sub = v_atoms[self._bond_indices_padded]
                hvp_padded = np.asarray(
                    device_get(_bond_hvp_batched(bond_pos, bond_tvecs, v_sub))
                )
                hvp = hvp_padded[:self._n_bonds_actual]
                active_idx = self._bond_indices
            else:
                active_idx = self._bond_indices[bonds_active]
                bond_pos = positions[active_idx]
                bond_tvecs = tvecs['bonds'][bonds_active]
                v_sub = v_atoms[active_idx]
                hvp = np.asarray(
                    device_get(_bond_hvp_batched(bond_pos, bond_tvecs, v_sub))
                )
            for k, u_atoms in enumerate(u_atoms_list):
                u_gathered = u_atoms[active_idx]  # (n_bonds, 2, 3)
                results[k].append(
                    np.sum(hvp * u_gathered, axis=(1, 2))
                )

        # --- Angles ---
        if angles_active.any() and self._n_angles_actual > 0:
            if angles_active.all():
                angle_pos = positions[self._angle_indices_padded]
                angle_tvecs = tvecs['angles_padded']
                v_sub = v_atoms[self._angle_indices_padded]
                hvp_padded = np.asarray(
                    device_get(
                        _angle_hvp_batched(angle_pos, angle_tvecs, v_sub)
                    )
                )
                hvp = hvp_padded[:self._n_angles_actual]
                active_idx = self._angle_indices
            else:
                active_idx = self._angle_indices[angles_active]
                angle_pos = positions[active_idx]
                angle_tvecs = tvecs['angles'][angles_active]
                v_sub = v_atoms[active_idx]
                hvp = np.asarray(
                    device_get(
                        _angle_hvp_batched(angle_pos, angle_tvecs, v_sub)
                    )
                )
            for k, u_atoms in enumerate(u_atoms_list):
                u_gathered = u_atoms[active_idx]  # (n_angles, 3, 3)
                results[k].append(
                    np.sum(hvp * u_gathered, axis=(1, 2))
                )

        # --- Dihedrals ---
        if dihedrals_active.any() and self._n_dihedrals_actual > 0:
            if dihedrals_active.all():
                dih_pos = positions[self._dihedral_indices_padded]
                dih_tvecs = tvecs['dihedrals_padded']
                v_sub = v_atoms[self._dihedral_indices_padded]
                hvp_padded = np.asarray(
                    device_get(
                        _dihedral_hvp_batched(
                            dih_pos, dih_tvecs, v_sub
                        )
                    )
                )
                hvp = hvp_padded[:self._n_dihedrals_actual]
                active_idx = self._dihedral_indices
            else:
                active_idx = self._dihedral_indices[dihedrals_active]
                dih_pos = positions[active_idx]
                dih_tvecs = tvecs['dihedrals'][dihedrals_active]
                v_sub = v_atoms[active_idx]
                hvp = np.asarray(
                    device_get(
                        _dihedral_hvp_batched(
                            dih_pos, dih_tvecs, v_sub
                        )
                    )
                )
            for k, u_atoms in enumerate(u_atoms_list):
                u_gathered = u_atoms[active_idx]  # (n_dihedrals, 4, 3)
                results[k].append(
                    np.sum(hvp * u_gathered, axis=(1, 2))
                )

        # --- Other (typically few coords, loop is fine) ---
        atoms = self.light_atoms
        for i, coord in enumerate(self.internals['other']):
            if other_active[i]:
                hess = np.array(coord.calc_hessian(atoms))
                idx = np.array(coord.indices)
                v_sub = v_atoms[idx]
                hvp = np.einsum('aibj,bj->ai', hess, v_sub)
                for k, u_atoms in enumerate(u_atoms_list):
                    u_sub = u_atoms[idx]
                    results[k].append(
                        np.array([np.sum(hvp * u_sub)])
                    )

        # --- Rotations ---
        for i, coord in enumerate(self.internals['rotations']):
            if rot_active[i]:
                idx = np.array(coord.indices)
                pos = positions[idx]
                v_sub = v_atoms[idx]
                axis = coord.kwargs['axis']
                refpos = coord.kwargs['refpos']
                hvp = np.asarray(
                    device_get(
                        _rotation_hvp_jit(pos, axis, refpos, v_sub)
                    )
                )
                for k, u_atoms in enumerate(u_atoms_list):
                    u_sub = u_atoms[idx]
                    results[k].append(
                        np.array([np.sum(hvp * u_sub)])
                    )

        # Concatenate fragments for each u
        out = []
        for r in results:
            if r:
                out.append(np.concatenate(r))
            else:
                out.append(np.empty(0))
        return tuple(out)

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

    def __iter__(self) -> Iterator[Coordinate]:
        for name in self._names:
            for coord in self.internals[name]:
                yield coord

    def _get_neighbors(self, dx: np.ndarray) -> Iterator[np.ndarray]:
        pbc = self.atoms.pbc
        if self.cell is None or not np.allclose(self.cell, self.atoms.cell):
            self.cell = self.atoms.cell.array.copy()
            rcell, self.op = minkowski_reduce(
                complete_cell(self.cell), pbc=pbc
            )
            self.rcell = Cell(rcell)
            self._rcell_reciprocal_T = self.rcell.reciprocal().T
        dx_sc = dx @ self._rcell_reciprocal_T
        offset = np.zeros(3, dtype=np.int32)
        for _ in range(2):
            offset += pbc * ((dx_sc - offset) // 1.).astype(np.int32)

        for ts in product(*[np.arange(-1 * p, p + 1) for p in pbc]):
            yield (np.array(ts) - offset) @ self.op

    def _find_mic(self, indices: Tuple[int, ...]) -> np.ndarray:
        ncvecs = np.zeros((len(indices) - 1, 3), dtype=np.int32)
        if not np.any(self.atoms.pbc):
            return ncvecs

        pos = self.all_positions
        dxs = np.array([
            pos[i] - pos[j] for i, j in zip(indices[1:], indices[:-1])
        ])

        for dx, ncvec in zip(dxs, ncvecs):
            vlen = np.inf
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
        pos = self.all_positions
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
        npos = len(self.all_positions)
        for i, trans in enumerate(self.internals['translations']):
            if idx in trans.indices and didx not in trans.indices:
                new_indices = (*trans.indices, didx)
                new_trans = Translation(new_indices, trans.kwargs['dim'])
                self.internals['translations'][i] = new_trans

        for i, rot in enumerate(self.internals['rotations']):
            if idx in rot.indices and didx not in rot.indices:
                new_indices = np.array((*rot.indices, didx), dtype=np.int32)
                if np.all(new_indices < npos):
                    new_rot = Rotation(
                        new_indices, rot.kwargs['axis'],
                        self.all_positions[new_indices]
                    )
                    self.internals['rotations'][i] = new_rot

    def check_all_gradients(
        self, delta: float = 1e-4, atol: float = 1e-6
    ) -> bool:
        success = True
        for coord in self:
            success &= coord.check_gradient(self.all_atoms, delta, atol)
        return success

    def check_all_hessians(
        self, delta: float = 1e-4, atol: float = 1e-6,
    ) -> bool:
        success = True
        for coord in self:
            success &= coord.check_hessian(self.all_atoms, delta, atol)
        return success


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

    def has_inequalities(self) -> bool:
        """Check if any inequality constraints (lt/gt) exist."""
        for name in self._names:
            for kind in self._kind[name]:
                if kind in ('lt', 'gt'):
                    return True
        return False

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
                self.all_positions[indices]
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
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        conv: float,
        indices: Union[Tuple[int, ...], Coordinate],
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
                .format(new, self._targets[name][idx] / conv)
            )

    fix_bond = partialmethod(_fix_internal, Bond, 'bonds', 1.)
    fix_angle = partialmethod(_fix_internal, Angle, 'angles', np.pi / 180.)
    fix_dihedral = partialmethod(
        _fix_internal, Dihedral, 'dihedrals', np.pi / 180.
    )

    def fix_other(
        self,
        coord: Coordinate,
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
            raise DuplicateConstraintError(
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
            self.add_dihedral, self.add_other, self.add_rotation
        )):
            for coord in self.cons.internals[kind]:
                adder(coord)
        self.allow_fragments = allow_fragments
        self.fragment_atom_groups = None

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
            new._internals_set[name] = self._internals_set[name].copy()
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
                self.all_positions[indices]
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
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        indices: Union[Tuple[int, ...], Coordinate],
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
        key = (tuple(new.indices), tuple(map(tuple, new.kwargs['ncvecs'])))
        if (
            key in self._internals_set[name]
            or new in self.forbidden[name]
        ):
            raise DuplicateInternalError
        self.internals[name].append(new)
        self._internals_set[name].add(key)
        self._active[name].append(True)

    add_bond = partialmethod(_add_internal, Bond, 'bonds')
    add_angle = partialmethod(_add_internal, Angle, 'angles')
    add_dihedral = partialmethod(_add_internal, Dihedral, 'dihedrals')

    def add_other(
        self,
        coord: Coordinate,
    ) -> None:
        try:
            self.internals['other'].index(coord)
        except ValueError:
            self.internals['other'].append(coord)
            self._active['other'].append(True)
        else:
            raise DuplicateInternalError()

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
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        indices: Union[Tuple[int, ...], Coordinate],
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

    def _find_bonds_vectorized(self, labels, scale, rcov):
        """Vectorized bond search across all candidate atom pairs.

        Returns a list of (i, j, ts) tuples for bonds that pass the
        distance threshold, where ts is the integer translation vector.
        """
        natoms = self.natoms
        pos = self.atoms.positions
        cell = self.atoms.cell.array
        pbc = self.atoms.pbc

        # Ensure cell/rcell/op are cached
        if self.cell is None or not np.allclose(self.cell, self.atoms.cell):
            self.cell = self.atoms.cell.array.copy()
            rcell, self.op = minkowski_reduce(
                complete_cell(self.cell), pbc=pbc
            )
            self.rcell = Cell(rcell)
            self._rcell_reciprocal_T = self.rcell.reciprocal().T

        # 1. Generate all candidate pairs (i <= j)
        ii, jj = np.triu_indices(natoms, k=0)
        # Skip pairs in the same labeled fragment
        same_frag = (labels[ii] == labels[jj]) & (labels[ii] != -1)
        keep = ~same_frag
        ii, jj = ii[keep], jj[keep]

        if len(ii) == 0:
            return []

        # 2. All pairwise displacements
        dx = pos[jj] - pos[ii]  # (n_pairs, 3)

        # 3. Pair-dependent offsets (vectorized _get_neighbors logic)
        dx_sc = dx @ self._rcell_reciprocal_T
        offset = np.zeros(dx_sc.shape, dtype=np.int32)
        for _ in range(2):
            offset += (pbc * ((dx_sc - offset) // 1.)).astype(np.int32)

        # 4. Base translation vectors from PBC dimensions
        ranges = [np.arange(-1 * p, p + 1) for p in pbc]
        base_ts = np.array(
            list(product(*ranges)), dtype=np.int32
        )  # (n_ts, 3)

        # 5. Shifted translations and Cartesian vectors
        shifted = base_ts[None, :, :] - offset[:, None, :]  # (n_pairs, n_ts, 3)
        tvecs_cart = (shifted @ self.op) @ cell  # (n_pairs, n_ts, 3)

        # 6. Distances
        dists = np.linalg.norm(
            dx[:, None, :] + tvecs_cart, axis=2
        )  # (n_pairs, n_ts)

        # 7. Covalent radius threshold
        thresholds = scale * (rcov[ii] + rcov[jj])
        bond_mask = dists <= thresholds[:, None]

        # 8. Exclude self-bonds (i==j) with zero translation
        self_bond = (ii == jj)
        zero_ts = np.all(shifted @ self.op == 0, axis=2)
        bond_mask &= ~(self_bond[:, None] & zero_ts)

        # 9. Collect hits
        pair_idx, ts_idx = np.nonzero(bond_mask)
        op = self.op
        results = []
        for k in range(len(pair_idx)):
            p = pair_idx[k]
            t = ts_idx[k]
            ts = (shifted[p, t] @ op).astype(np.int32)
            results.append((int(ii[p]), int(jj[p]), ts))
        return results

    def _wrap_fragment_positions(self, group, cumshifts):
        """Shift atom positions so fragment atoms are contiguous across PBC.

        BFS from first atom in group, using bond ncvecs to bring each
        bonded neighbor into the same periodic image. Accumulates shifts
        along bond chains so molecules spanning multiple cell boundaries
        are fully contracted. Records cumulative shifts in cumshifts dict
        for subsequent ncvec correction.
        """
        group_set = set(group)
        cell = np.asarray(self.atoms.cell)

        adj = {i: [] for i in group}
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i in group_set and j in group_set:
                ncvec = bond.kwargs['ncvecs'][0]
                adj[i].append((j, ncvec))
                adj[j].append((i, -ncvec))

        anchor = group[0]
        cumshifts[anchor] = np.zeros(3, dtype=int)
        queue = [anchor]
        while queue:
            i = queue.pop(0)
            for j, ncvec in adj[i]:
                if j in cumshifts:
                    continue
                cumshifts[j] = ncvec + cumshifts[i]
                self.atoms.positions[j] += cumshifts[j] @ cell
                queue.append(j)

    def find_all_bonds(
        self,
        nbond_cart_thr: int = 6,
        max_bonds: int = 20,
        scale: float = 1.25,
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

            # Remove labels from atoms with no bonding partners.
            # This must happen BEFORE the allow_fragments break, otherwise
            # single atoms will retain fragment labels and cause rotation ICs
            # to be incorrectly added to single-atom groups.
            labels[nbonds == 0] = -1

            if self.allow_fragments and not first_run:
                break

            candidates = self._find_bonds_vectorized(
                labels, scale, rcov
            )
            for i, j, ts in candidates:
                try:
                    self.add_bond((i, j), ts)
                except DuplicateInternalError:
                    continue
                if nbonds[i] < max_bonds and nbonds[j] < max_bonds:
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
            cumshifts = {}
            self.fragment_atom_groups = []
            for group in groups:
                if not group:
                    continue
                self._wrap_fragment_positions(group, cumshifts)
                self.fragment_atom_groups.append(np.array(group, dtype=np.int32))
                self.add_translation(group)
                self.add_rotation(group)

            # Update bond ncvecs to match the new wrapped positions.
            # ncvec_new = ncvec_old - cumshift[j] + cumshift[i]
            zero = np.zeros(3, dtype=int)
            for bond in self.internals['bonds']:
                i, j = bond.indices
                shift_i = cumshifts.get(i, zero)
                shift_j = cumshifts.get(j, zero)
                if np.any(shift_i != 0) or np.any(shift_j != 0):
                    bond.kwargs['ncvecs'] = np.array(
                        [bond.kwargs['ncvecs'][0] - shift_j + shift_i]
                    )

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
                            # orthogonal with the shorter of the two
                            # displacement vectors.
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
                        self._batched_arrays_valid = False
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
                    dbond2 = Bond(
                        (self.dinds[j], b2.indices[1]), b2.kwargs['ncvecs']
                    )
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
                                -b1.kwargs['ncvecs'][0],
                                b3.kwargs['ncvecs'][0],
                                b2.kwargs['ncvecs'][0] - b3.kwargs['ncvecs'][0]
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
        # First, find proper dihedrals from angle combinations.
        # Group angles by their bond edges so we only try pairs that
        # share a bond (required for __add__ to succeed).
        edge_to_angles = {}
        for angle in self.internals['angles']:
            i, j, k = angle.indices
            for edge_key in ((min(i, j), max(i, j)), (min(j, k), max(j, k))):
                edge_to_angles.setdefault(edge_key, []).append(angle)

        seen_pairs = set()
        for angles_on_edge in edge_to_angles.values():
            for a1, a2 in combinations(angles_on_edge, 2):
                pair_key = (id(a1), id(a2))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                try:
                    new = a1 + a2
                except NoValidInternalError:
                    continue
                # this is a dihedral that has the same exact atom as both
                # the first and last atom.
                if (
                    new.indices[0] == new.indices[3]
                    and np.all(
                        np.sum(new.kwargs['ncvecs'], axis=0)
                        == np.array((0, 0, 0))
                    )
                ):
                    continue
                try:
                    self.add_dihedral(new)
                except DuplicateInternalError:
                    continue

        # Second, add improper dihedrals for atoms with 3 or 4 neighbors that don't
        # have any proper dihedral passing through them. This is needed because:
        # 1. At planar geometries, bond/angle derivatives vanish for out-of-plane motion
        # 2. Even starting non-planar, the geometry may planarize during optimization
        # 3. Improper dihedrals capture the out-of-plane (umbrella) mode


        # Note this does add some redundancy to the internals but it also makes it
        # so that the Jacobian is well-conditioned in the case of planar systems,
        # such as nitrate.
        #
        # We only add impropers when no proper dihedral exists through the atom,
        # which avoids excessive unnecessary additional internals.

        # First, find which atoms have proper dihedrals through them
        dihedral_centers = set()
        for d, a in zip(self.internals['dihedrals'], self._active['dihedrals']):
            if a:
                # Positions 1 and 2 are the "central" atoms of a dihedral
                dihedral_centers.add(int(d.indices[1]))
                dihedral_centers.add(int(d.indices[2]))

        # Build neighbor list
        neighbors = [[] for _ in range(self.natoms)]
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i < self.natoms:
                neighbors[i].append((int(j), bond.kwargs['ncvecs'][0]))
            if j < self.natoms:
                neighbors[j].append((int(i), -bond.kwargs['ncvecs'][0]))

        for center in range(self.natoms):
            # Consider atoms with 3 or 4 neighbors that lack proper dihedrals.
            # - 3 neighbors: at planar geometries (e.g., NO3, sp2 carbons), the
            #   3 angles sum to 360°, creating linear dependency.
            # - 4 neighbors: at square planar geometries (e.g., Pt(II)), the
            #   4 cis angles sum to 360°, similar issue. For tetrahedral, the
            #   improper is redundant but harmless (pseudo-inverse handles it).
            # - 5+ neighbors: rare, and typically have proper dihedrals anyway.
            if len(neighbors[center]) not in (3, 4):
                continue

            # Skip if this atom already has proper dihedrals through it
            if center in dihedral_centers:
                continue

            # Add improper dihedral: neighbors[0]-center-neighbors[1]-neighbors[2]
            n0, ncvec0 = neighbors[center][0]
            n1, ncvec1 = neighbors[center][1]
            n2, ncvec2 = neighbors[center][2]
            # Improper dihedral indices: (n0, center, n1, n2)
            # The ncvecs connect consecutive atoms in the dihedral
            imp_ncvecs = (
                -ncvec0,  # from n0 to center
                ncvec1,   # from center to n1
                ncvec2 - ncvec1,  # from n1 to n2
            )
            try:
                self.add_dihedral((n0, center, n1, n2), imp_ncvecs)
            except DuplicateInternalError:
                pass

    def validate_basis(self) -> None:
        jac = self.jacobian()
        S = svdvals(jac)
        ndeloc = np.sum(S > 1e-8)

        # If TRICs (translations/rotations) are present, they span the full
        # 3N DOF. Otherwise, 6 DOF are removed for global translation/rotation.
        has_trics = (len(self.internals['translations']) > 0 or
                     len(self.internals['rotations']) > 0)
        if has_trics:
            ndof = 3 * (self.natoms + self.ndummies)
        else:
            ntot = self.natoms + self.ndummies
            has_periodic_bonds = any(
                np.any(bond.kwargs['ncvecs'] != 0)
                for bond in self.internals['bonds']
            )
            if has_periodic_bonds:
                ndof = 3 * ntot
            elif ntot <= 1:
                ndof = 0
            elif ntot == 2:
                ndof = 1
            else:
                ndof = 3 * ntot - 6

        if ndeloc != ndof:
            warnings.warn(
                f'{ndeloc} coords found! Expected {ndof}.'
            )

    def check_for_bad_internals(self) -> Optional[Dict[str, List[Coordinate]]]:
        """Check for angles that are too close to 0 or pi (linear).

        Uses vectorized computation for efficiency.
        """
        bad = {'bonds': [], 'angles': []}

        angles = self.internals['angles']
        if not angles:
            return None

        # Use vectorized computation to check all angles at once
        # Use padded arrays for consistent JAX shapes (avoids recompilation)
        self._build_batched_arrays()
        if self._n_angles_actual > 0:
            positions = self.all_positions
            cell = self.atoms.cell.array
            tvecs = self._get_cached_tvecs(cell)
            angle_pos = positions[self._angle_indices_padded]
            angle_vals_padded = np.asarray(_angle_value_batched(angle_pos, tvecs['angles_padded']))
            angle_vals = angle_vals_padded[:self._n_angles_actual]

            # Find bad angles
            bad_mask = ~((self.atol < angle_vals) & (angle_vals < np.pi - self.atol))
            if np.any(bad_mask):
                bad_indices = np.where(bad_mask)[0]
                for idx in bad_indices:
                    bad['angles'].append(angles[idx])

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
        dummy_set = set(range(self.natoms, self.natoms + self.ndummies))
        for dihedral in self.internals['dihedrals']:
            if any(j in dummy_set for j in dihedral.indices):
                h0[idx] = 0.5 * units.Hartree
            else:
                h0[idx] = self._h0_dihedral(dihedral, nbonds)
            idx += 1
        # remaining degrees of freedom are rotations.
        # No idea what a good curvature is for these
        h0[idx:] = 1.
        return np.diag(np.abs(h0))
